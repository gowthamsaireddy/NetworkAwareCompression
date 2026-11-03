"""Receiver-side image storage and retrieval."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from shared.metadata import TransmissionMetadata


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

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.received_items: List[ReceivedImageItem] = []

    def save_image(
        self,
        metadata: TransmissionMetadata,
        image_bytes: bytes,
        timestamp: str,
    ) -> Path:
        """Saves verified image binary to disk with deterministic naming."""
        safe_img_id = metadata.image_id.replace("/", "_").replace("\\", "_")
        filename = f"{safe_img_id}_Q{metadata.compression_quality}_{metadata.experiment_id}.jpg"
        target_path = self.storage_dir / filename

        with open(target_path, "wb") as f:
            f.write(image_bytes)

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

