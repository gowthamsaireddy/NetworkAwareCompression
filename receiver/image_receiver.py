"""Receiver-side image storage and retrieval."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil
from typing import List, Optional, Union
from shared.metadata import TransmissionMetadata

MIN_FREE_DISK_BYTES = 50 * 1024 * 1024  # 50 MB safety margin


@dataclass
class ReceivedImageItem:
    """Represents a successfully received, integrity-checked image."""
    image_id: str
    experiment_id: str
    file_path: Path
    metadata: TransmissionMetadata
    received_timestamp: str


class ImageReceiverManager:
    """Manages storage and indexing of received compressed images."""

    def __init__(self, storage_dir: Union[Path, str]):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.received_items: List[ReceivedImageItem] = []

    def save_image(
        self,
        metadata: TransmissionMetadata,
        image_bytes: bytes,
        timestamp: Optional[str] = None,
    ) -> Path:
        """Saves verified image binary to disk with deterministic naming and security validation."""
        if not timestamp:
            timestamp = datetime.now().isoformat()

        # Check storage space
        try:
            free_bytes = shutil.disk_usage(self.storage_dir).free
            if free_bytes < (len(image_bytes) + MIN_FREE_DISK_BYTES):
                raise OSError(f"Insufficient disk space ({free_bytes // (1024*1024)} MB available).")
        except Exception as e:
            if isinstance(e, OSError):
                raise

        # Sanitize filenames to prevent path traversal while preserving parentheses in image ids
        clean_img_id = metadata.image_id.replace("..", "_").replace("/", "_").replace("\\", "_")
        safe_img_id = re.sub(r"[^\w\s\(\)\.-]", "_", clean_img_id).strip()
        clean_exp_id = metadata.experiment_id.replace("..", "_").replace("/", "_").replace("\\", "_")
        safe_exp_id = re.sub(r"[^\w\s\(\)\.-]", "_", clean_exp_id).strip()
        filename = f"{safe_img_id}_Q{metadata.compression_quality}_{safe_exp_id}.jpg"

        target_path = (self.storage_dir / filename).resolve()
        storage_resolved = self.storage_dir.resolve()
        if not str(target_path).startswith(str(storage_resolved)):
            raise ValueError(f"Path traversal detected: {metadata.image_id}")

        with open(target_path, "wb") as f:
            f.write(image_bytes)

        # Write metadata JSON sidecar to preserve 100% exact metadata across restarts
        meta_sidecar = target_path.with_suffix(".json")
        try:
            with open(meta_sidecar, "w", encoding="utf-8") as f_meta:
                f_meta.write(metadata.to_json())
        except Exception:
            pass

        item = ReceivedImageItem(
            image_id=metadata.image_id,
            experiment_id=metadata.experiment_id,
            file_path=target_path,
            metadata=metadata,
            received_timestamp=timestamp,
        )
        self.received_items.append(item)
        return target_path

    def get_all(self) -> List[ReceivedImageItem]:
        return self.received_items

