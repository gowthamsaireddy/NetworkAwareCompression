"""BUSI Dataset manager for sender application.

Handles directory scanning, image and mask identification, manifest generation,
and ground-truth mask combination without modifying source files.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from common.logging_config import setup_logger

logger = setup_logger("DatasetManager")


@dataclass
class DatasetItem:
    """Represents an image item and its associated ground-truth mask(s)."""
    image_id: str
    image_path: Path
    category: str  # 'benign', 'malignant', 'normal'
    mask_paths: List[Path]
    file_size_bytes: int
    width: int
    height: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "image_path": str(self.image_path),
            "category": self.category,
            "mask_paths": [str(p) for p in self.mask_paths],
            "file_size_bytes": self.file_size_bytes,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetItem":
        return cls(
            image_id=str(data["image_id"]),
            image_path=Path(data["image_path"]),
            category=str(data["category"]),
            mask_paths=[Path(p) for p in data.get("mask_paths", [])],
            file_size_bytes=int(data["file_size_bytes"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


class DatasetManager:
    """Scans, verifies, and indexes BUSI ultrasound image collections."""

    VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir: Optional[Path] = dataset_dir
        self.items: List[DatasetItem] = []

    def set_dataset_directory(self, path: Path) -> bool:
        """Sets dataset root directory. Returns True if valid directory."""
        if path.is_dir():
            self.dataset_dir = path
            return True
        return False

    def scan_dataset(self) -> List[DatasetItem]:
        """Scans dataset root directory for images and corresponding ground-truth masks.

        Supports standard BUSI directory hierarchies:
        - Root with 'benign', 'malignant', 'normal' subfolders
        - Nested root e.g. Dataset_BUSI_with_mask/benign/
        - Flat directory structure with category naming patterns

        Never modifies source images or masks.
        """
        self.items.clear()
        if not self.dataset_dir or not self.dataset_dir.exists():
            logger.warning(f"Scan aborted: dataset directory does not exist ({self.dataset_dir})")
            return self.items

        # First, search for all image files recursively
        all_image_files: List[Path] = [
            f for f in self.dataset_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in self.VALID_IMAGE_EXTS
        ]

        # Separate masks from ultrasound images
        # BUSI masks always include "_mask" in the stem
        mask_map: Dict[str, List[Path]] = {}
        candidate_images: List[Path] = []

        for f in all_image_files:
            stem_lower = f.stem.lower()
            if "_mask" in stem_lower:
                # Extract the base image stem by stripping '_mask' and any trailing suffix like '_1'
                base_stem = re.sub(r"_mask.*$", "", f.stem, flags=re.IGNORECASE)
                parent_key = f"{f.parent.name.lower()}/{base_stem}"
                mask_map.setdefault(parent_key, []).append(f)
            else:
                candidate_images.append(f)

        # Process each ultrasound image
        for img_path in sorted(candidate_images, key=lambda p: (p.parent.name, p.name)):
            parent_name = img_path.parent.name.lower()
            # Determine category from parent folder or file stem
            if "benign" in parent_name or "benign" in img_path.stem.lower():
                category = "benign"
            elif "malignant" in parent_name or "malignant" in img_path.stem.lower():
                category = "malignant"
            elif "normal" in parent_name or "normal" in img_path.stem.lower():
                category = "normal"
            else:
                category = parent_name or "unknown"

            image_id = f"{category}/{img_path.stem}"
            parent_key = f"{parent_name}/{img_path.stem}"
            mask_paths = sorted(mask_map.get(parent_key, []))

            # Query dimensions cleanly via header inspection
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                logger.warning(f"Could not read image header for {img_path}: {e}")
                width, height = 0, 0

            item = DatasetItem(
                image_id=image_id,
                image_path=img_path,
                category=category,
                mask_paths=mask_paths,
                file_size_bytes=img_path.stat().st_size,
                width=width,
                height=height,
            )
            self.items.append(item)

        logger.info(f"Dataset scan complete: found {len(self.items)} images across {self.dataset_dir}")
        return self.items

    def get_item_by_id(self, image_id: str) -> Optional[DatasetItem]:
        """Retrieve dataset item by unique image_id."""
        for item in self.items:
            if item.image_id == image_id:
                return item
        return None

    def filter_by_category(self, category: Optional[str] = None) -> List[DatasetItem]:
        """Filters items by category (e.g., 'benign', 'malignant', 'normal'). None returns all."""
        if not category or category.lower() == "all":
            return list(self.items)
        target = category.lower()
        return [item for item in self.items if item.category == target]

    def load_original_image(self, item: DatasetItem, as_rgb: bool = True) -> np.ndarray:
        """Loads original uncompressed image without altering source file."""
        if not item.image_path.exists():
            raise FileNotFoundError(f"Image not found at {item.image_path}")

        with Image.open(item.image_path) as img:
            if as_rgb and img.mode != "RGB":
                img = img.convert("RGB")
            return np.array(img)

    def load_ground_truth_mask(self, item: DatasetItem) -> Optional[np.ndarray]:
        """Loads and combines ground-truth masks into a single binary mask (0 or 255).

        If an image has multiple masks (e.g. multi-lesion cases), they are combined using bitwise OR.
        Ground-truth masks are NEVER compressed.
        Returns None if no masks exist (e.g., normal cases without lesions).
        """
        if not item.mask_paths:
            return None

        valid_masks = [p for p in item.mask_paths if p.exists()]
        if not valid_masks:
            return None

        combined: Optional[np.ndarray] = None
        for mask_path in valid_masks:
            with Image.open(mask_path) as m_img:
                m_arr = np.array(m_img.convert("L"))
                # Binarize mask: >0 becomes 255
                bin_mask = (m_arr > 0).astype(np.uint8) * 255

                if combined is None:
                    combined = bin_mask
                else:
                    if combined.shape != bin_mask.shape:
                        logger.warning(
                            f"Mask shape mismatch in multi-mask image {item.image_id}: {combined.shape} vs {bin_mask.shape}"
                        )
                    else:
                        combined = np.maximum(combined, bin_mask)

        return combined

    def export_manifest(self, output_path: Path) -> Path:
        """Exports dataset manifest to a JSON file for reproducible logging."""
        manifest_data = {
            "dataset_root": str(self.dataset_dir),
            "total_images": len(self.items),
            "category_counts": {
                "benign": len(self.filter_by_category("benign")),
                "malignant": len(self.filter_by_category("malignant")),
                "normal": len(self.filter_by_category("normal")),
            },
            "items": [item.to_dict() for item in self.items],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        logger.info(f"Manifest exported to {output_path}")
        return output_path

    def load_manifest(self, manifest_path: Path) -> List[DatasetItem]:
        """Loads dataset items from an exported manifest file."""
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.items = [DatasetItem.from_dict(d) for d in data.get("items", [])]
        if "dataset_root" in data and data["dataset_root"]:
            self.dataset_dir = Path(data["dataset_root"])
        logger.info(f"Loaded {len(self.items)} items from manifest {manifest_path}")
        return self.items
