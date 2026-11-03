"""BUSI Dataset manager for sender application.

Handles directory scanning, image and mask identification, manifest generation,
without altering original image or mask files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DatasetItem:
    """Represents an image item and its associated ground-truth mask(s)."""
    image_id: str
    image_path: Path
    category: str  # 'benign', 'malignant', 'normal'
    mask_paths: List[Path]
    file_size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None


class DatasetManager:
    """Scans and manages BUSI ultrasound image collections."""

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
        """Scans the dataset directory and identifies images and masks.

        Note: Phase 3 will implement full BUSI file pattern matching.
        """
        self.items.clear()
        if not self.dataset_dir or not self.dataset_dir.exists():
            return self.items

        # Look for category folders (benign, malignant, normal)
        for category_dir in [d for d in self.dataset_dir.iterdir() if d.is_dir()]:
            category = category_dir.name.lower()
            # Scan files that are images and not masks
            for file_path in category_dir.glob("*.png"):
                if "_mask" not in file_path.stem:
                    image_id = f"{category}/{file_path.stem}"
                    # Find all associated masks (some images have multiple masks: _mask.png, _mask_1.png)
                    mask_paths = list(category_dir.glob(f"{file_path.stem}_mask*.png"))
                    item = DatasetItem(
                        image_id=image_id,
                        image_path=file_path,
                        category=category,
                        mask_paths=mask_paths,
                        file_size_bytes=file_path.stat().st_size,
                    )
                    self.items.append(item)

        return self.items

    def get_item_by_id(self, image_id: str) -> Optional[DatasetItem]:
        """Retrieve dataset item by unique image_id."""
        for item in self.items:
            if item.image_id == image_id:
                return item
        return None

