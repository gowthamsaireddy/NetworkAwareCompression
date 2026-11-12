"""Receiver Image Viewer & Inspection View."""

from pathlib import Path
import re
from typing import List, Optional, Tuple
from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from common.config import DATA_DIR
from shared.constants import PROTOCOL_VERSION, STATUS_NO_DATA
from shared.metadata import TransmissionMetadata
from shared.protocol import compute_sha256
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class ReceiverImageView(QWidget):
    """Detailed visual inspection and navigation of received ultrasound images."""

    image_selected = pyqtSignal(object, object)  # Emits (TransmissionMetadata, Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.received_items: List[Tuple[TransmissionMetadata, Path]] = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Received Image Viewer",
            "Inspect received compressed ultrasound frames and metadata",
        )
        layout.addWidget(header)

        # Splitter: Left navigation list/table, Right preview & metadata
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left list of received images
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Received Image Catalog:"))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Image ID", "Quality", "Size (Bytes)", "Format"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        left_layout.addWidget(self.table)

        splitter.addWidget(left_widget)

        # Right inspection panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_box = ImagePreviewBox("Received Image Preview", STATUS_NO_DATA)
        right_layout.addWidget(self.preview_box, stretch=1)

        # Metadata cards
        meta_grid = QGridLayout()
        meta_grid.setSpacing(8)

        self.card_dimensions = MetricBox("Image Dimensions", STATUS_NO_DATA)
        self.card_quality = MetricBox("Compression Quality", STATUS_NO_DATA)
        self.card_orig_size = MetricBox("Original Size", STATUS_NO_DATA)
        self.card_comp_size = MetricBox("Compressed Size", STATUS_NO_DATA)

        meta_grid.addWidget(self.card_dimensions, 0, 0)
        meta_grid.addWidget(self.card_quality, 0, 1)
        meta_grid.addWidget(self.card_orig_size, 1, 0)
        meta_grid.addWidget(self.card_comp_size, 1, 1)

        right_layout.addLayout(meta_grid)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, stretch=1)

        # Table click handler
        self.table.itemSelectionChanged.connect(self._on_table_selection)

    def add_received_image(self, metadata: TransmissionMetadata, save_path: Path):
        """Adds a verified received image to the inspection catalog."""
        self.received_items.append((metadata, save_path))
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(metadata.image_id))
        self.table.setItem(row, 1, QTableWidgetItem(f"Q{metadata.compression_quality}"))
        self.table.setItem(row, 2, QTableWidgetItem(str(metadata.image_byte_length)))
        self.table.setItem(row, 3, QTableWidgetItem(metadata.image_format))

        # Select the newly added row
        self.table.selectRow(row)

    def load_existing_received_images(self, search_dir: Optional[Path] = None) -> int:
        """Scans disk storage for existing received images and indexes them."""
        target_dir = search_dir or (DATA_DIR / "received")
        if not target_dir.exists():
            return 0

        loaded_count = 0
        pattern = re.compile(r"^(.*)_Q(\d+)_(.*)\.jpg$")

        for img_path in sorted(target_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime):
            # Prevent duplicate indexing
            if any(path == img_path for _, path in self.received_items):
                continue

            match = pattern.match(img_path.name)
            if not match:
                continue

            safe_img_id, q_str, exp_id = match.groups()
            quality = int(q_str)

            # Restore original image_id category/name structure
            image_id = safe_img_id
            for prefix in ("benign_", "malignant_", "normal_"):
                if safe_img_id.startswith(prefix):
                    cat = prefix[:-1]
                    name = safe_img_id[len(prefix):]
                    image_id = f"{cat}/{name}"
                    break

            try:
                raw_bytes = img_path.read_bytes()
                comp_size = len(raw_bytes)
                sha = compute_sha256(raw_bytes)

                with Image.open(img_path) as pil_img:
                    w, h = pil_img.size

                # Attempt to determine original file size if source exists
                orig_size = comp_size
                for cand in (DATA_DIR / "BUSI").rglob("*.png"):
                    if "_mask" not in cand.name and Path(image_id).stem in cand.name:
                        orig_size = cand.stat().st_size
                        break

                meta = TransmissionMetadata(
                    protocol_version=PROTOCOL_VERSION,
                    experiment_id=exp_id,
                    image_id=image_id,
                    compression_quality=quality,
                    image_width=w,
                    image_height=h,
                    image_format="JPEG",
                    original_file_size=orig_size,
                    compressed_file_size=comp_size,
                    image_byte_length=comp_size,
                    sha256_checksum=sha,
                )

                self.add_received_image(meta, img_path)
                loaded_count += 1
            except Exception:
                continue

        return loaded_count

    def _on_table_selection(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        if 0 <= row < len(self.received_items):
            metadata, save_path = self.received_items[row]
            if save_path.exists():
                pixmap = QPixmap(str(save_path))
                caption = (
                    f"{metadata.image_id} | Q{metadata.compression_quality} | "
                    f"{metadata.image_width}×{metadata.image_height} | {metadata.image_byte_length:,} bytes"
                )
                self.preview_box.set_pixmap(pixmap, caption=caption)

            self.card_dimensions.set_value(f"{metadata.image_width} × {metadata.image_height}")
            self.card_quality.set_value(f"Q{metadata.compression_quality}")
            self.card_orig_size.set_value(f"{metadata.original_file_size:,} B")
            self.card_comp_size.set_value(f"{metadata.compressed_file_size:,} B")
            self.image_selected.emit(metadata, save_path)
