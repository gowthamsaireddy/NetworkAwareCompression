"""Sender JPEG Compression & Image Quality Analysis View."""

from io import BytesIO
from pathlib import Path
import time
from typing import Optional
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from sender.compression import CompressionResult, ImageCompressor
from sender.dataset_manager import DatasetItem
from shared.constants import (
    COMPRESSION_QUALITIES,
    PROTOCOL_VERSION,
    STATUS_NOT_CALCULATED,
)
from shared.metadata import TransmissionMetadata
from shared.protocol import compute_sha256
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class SenderCompressionView(QWidget):
    """Interactive variable-rate JPEG compression and distortion evaluation."""

    compression_completed = pyqtSignal(object, bytes)  # Emits (TransmissionMetadata, image_bytes)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item: Optional[DatasetItem] = None
        self.current_image_path: Optional[Path] = None
        self.current_result: Optional[CompressionResult] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "JPEG Compression & Quality Evaluation",
            "Perform standardized JPEG compression (Q30, Q50, Q70, Q90) and evaluate PSNR / SSIM fidelity",
        )
        layout.addWidget(header)

        # Control panel: Quality selection & Execute button
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("CardFrame")
        ctrl_layout = QHBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(16, 12, 16, 12)

        self.browse_single_btn = QPushButton("Select Image File")
        self.browse_single_btn.setObjectName("SecondaryBtn")
        ctrl_layout.addWidget(self.browse_single_btn)

        ctrl_layout.addWidget(QLabel("Select Compression Quality:"))

        self.btn_group = QButtonGroup(self)
        self.radio_q30 = QRadioButton("Q30 (High Compression)")
        self.radio_q50 = QRadioButton("Q50 (Medium-High)")
        self.radio_q70 = QRadioButton("Q70 (Balanced)")
        self.radio_q90 = QRadioButton("Q90 (High Quality)")
        self.radio_q70.setChecked(True)

        self.btn_group.addButton(self.radio_q30, 30)
        self.btn_group.addButton(self.radio_q50, 50)
        self.btn_group.addButton(self.radio_q70, 70)
        self.btn_group.addButton(self.radio_q90, 90)

        ctrl_layout.addWidget(self.radio_q30)
        ctrl_layout.addWidget(self.radio_q50)
        ctrl_layout.addWidget(self.radio_q70)
        ctrl_layout.addWidget(self.radio_q90)
        ctrl_layout.addStretch()

        self.compress_btn = QPushButton("Compress Current Image")
        ctrl_layout.addWidget(self.compress_btn)

        layout.addWidget(ctrl_frame)

        # Two Image Previews: Original vs Compressed
        previews_layout = QHBoxLayout()
        self.original_preview = ImagePreviewBox("Original Uncompressed Image", "No image selected")
        self.compressed_preview = ImagePreviewBox("JPEG Compressed Image", "Not compressed yet")
        previews_layout.addWidget(self.original_preview)
        previews_layout.addWidget(self.compressed_preview)
        layout.addLayout(previews_layout, stretch=1)

        # Metrics display grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        self.card_orig_size = MetricBox("Original Size", STATUS_NOT_CALCULATED)
        self.card_comp_size = MetricBox("Compressed Size", STATUS_NOT_CALCULATED)
        self.card_ratio = MetricBox("Compression Ratio", STATUS_NOT_CALCULATED)
        self.card_reduction = MetricBox("Size Reduction %", STATUS_NOT_CALCULATED)
        self.card_psnr = MetricBox("PSNR (dB)", STATUS_NOT_CALCULATED)
        self.card_ssim = MetricBox("SSIM", STATUS_NOT_CALCULATED)

        metrics_grid.addWidget(self.card_orig_size, 0, 0)
        metrics_grid.addWidget(self.card_comp_size, 0, 1)
        metrics_grid.addWidget(self.card_ratio, 0, 2)
        metrics_grid.addWidget(self.card_reduction, 1, 0)
        metrics_grid.addWidget(self.card_psnr, 1, 1)
        metrics_grid.addWidget(self.card_ssim, 1, 2)

        layout.addLayout(metrics_grid)

        # Connect actions
        self.browse_single_btn.clicked.connect(self._browse_single_image)
        self.compress_btn.clicked.connect(self._handle_compress)

    def load_dataset_item(self, item: DatasetItem):
        """Loads a DatasetItem selected from DatasetView."""
        self.current_item = item
        self.current_image_path = item.image_path
        if item.image_path.exists():
            pix = QPixmap(str(item.image_path))
            self.original_preview.set_pixmap(
                pix,
                caption=f"ID: {item.image_id} | {item.width}×{item.height} | {item.file_size_bytes:,} bytes",
            )
            self.compressed_preview.clear_image("Not compressed yet")
            self._reset_metrics()

    def _browse_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Ultrasound Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            p = Path(file_path)
            if "_mask" in p.stem.lower():
                QMessageBox.warning(
                    self,
                    "Mask Selected",
                    "The selected file appears to be a ground-truth mask. Ground-truth masks must NEVER be compressed.",
                )
                return

            with Image.open(p) as img:
                w, h = img.size

            item = DatasetItem(
                image_id=f"custom/{p.stem}",
                image_path=p,
                category="custom",
                mask_paths=[],
                file_size_bytes=p.stat().st_size,
                width=w,
                height=h,
            )
            self.load_dataset_item(item)

    def _reset_metrics(self):
        self.card_orig_size.set_value(STATUS_NOT_CALCULATED)
        self.card_comp_size.set_value(STATUS_NOT_CALCULATED)
        self.card_ratio.set_value(STATUS_NOT_CALCULATED)
        self.card_reduction.set_value(STATUS_NOT_CALCULATED)
        self.card_psnr.set_value(STATUS_NOT_CALCULATED)
        self.card_ssim.set_value(STATUS_NOT_CALCULATED)

    def _handle_compress(self):
        if not self.current_image_path or not self.current_image_path.exists():
            QMessageBox.warning(self, "No Image", "Please select an ultrasound image from Dataset Management first.")
            return

        quality = self.btn_group.checkedId()
        if quality <= 0:
            quality = 70

        try:
            res = ImageCompressor.compress_image(self.current_image_path, quality)
            self.current_result = res

            # Convert decoded RGB array to QPixmap
            h, w, ch = res.decoded_array.shape
            bytes_per_line = ch * w
            q_img = QImage(
                res.decoded_array.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )
            comp_pix = QPixmap.fromImage(q_img)

            caption = (
                f"JPEG Q{res.quality} | {res.width}×{res.height} | {res.compressed_size_bytes:,} bytes "
                f"({res.compression_ratio:.2f}:1)"
            )
            self.compressed_preview.set_pixmap(comp_pix, caption=caption)

            # Update genuine measured metric cards
            self.card_orig_size.set_value(f"{res.original_size_bytes:,} B")
            self.card_comp_size.set_value(f"{res.compressed_size_bytes:,} B")
            self.card_ratio.set_value(f"{res.compression_ratio:.2f} : 1")
            self.card_reduction.set_value(f"{res.size_reduction_pct:.2f}%")
            self.card_psnr.set_value(f"{res.psnr:.2f} dB" if res.psnr != float("inf") else "inf")
            self.card_ssim.set_value(f"{res.ssim:.4f}")

            # Construct transmission metadata and emit for TransmissionView
            img_id = self.current_item.image_id if self.current_item else f"custom/{self.current_image_path.stem}"
            sha = compute_sha256(res.compressed_bytes)
            metadata = TransmissionMetadata(
                protocol_version=PROTOCOL_VERSION,
                experiment_id=f"EXP_{int(time.time())}",
                image_id=img_id,
                compression_quality=quality,
                image_width=res.width,
                image_height=res.height,
                image_format="JPEG",
                original_file_size=res.original_size_bytes,
                compressed_file_size=res.compressed_size_bytes,
                image_byte_length=res.compressed_size_bytes,
                sha256_checksum=sha,
            )
            self.compression_completed.emit(metadata, res.compressed_bytes)

        except Exception as e:
            QMessageBox.critical(self, "Compression Error", f"Failed to compress image: {e}")
