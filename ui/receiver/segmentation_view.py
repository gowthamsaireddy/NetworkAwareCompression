"""Receiver Segmentation Inference View."""

from pathlib import Path
from typing import Optional
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from receiver.model_manager import ModelManager
from receiver.segmentation import SegmentationEngine, SegmentationResult
from shared.constants import STATUS_NOT_CALCULATED
from shared.metadata import TransmissionMetadata
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class ReceiverSegmentationView(QWidget):
    """Executes frozen U-Net inference to generate predicted lesion masks and visual overlays."""

    segmentation_completed = pyqtSignal(object, object)  # Emits (SegmentationResult, TransmissionMetadata)

    def __init__(self, model_manager: Optional[ModelManager] = None, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager or ModelManager()
        self.engine = SegmentationEngine(self.model_manager)

        self.current_image_path: Optional[Path] = None
        self.current_metadata: Optional[TransmissionMetadata] = None
        self.current_result: Optional[SegmentationResult] = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Lesion Segmentation Inference",
            "Generate binary lesion masks and overlays using frozen U-Net model",
        )
        layout.addWidget(header)

        # Control row
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("CardFrame")
        ctrl_layout = QHBoxLayout(ctrl_frame)

        self.run_btn = QPushButton("Run Segmentation on Selected Image")
        self.run_batch_btn = QPushButton("Run Batch Inference on All Received Images")
        self.run_batch_btn.setObjectName("SecondaryBtn")

        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addWidget(self.run_batch_btn)
        ctrl_layout.addStretch()

        layout.addWidget(ctrl_frame)

        # Previews: Received Image, Predicted Binary Mask, Overlay
        previews_layout = QHBoxLayout()
        self.recv_preview = ImagePreviewBox("Received Image", "No image selected")
        self.pred_mask_preview = ImagePreviewBox("Predicted Lesion Mask (256×256)", "Not generated")
        self.overlay_preview = ImagePreviewBox("Lesion Overlay", "Not generated")

        previews_layout.addWidget(self.recv_preview)
        previews_layout.addWidget(self.pred_mask_preview)
        previews_layout.addWidget(self.overlay_preview)

        layout.addLayout(previews_layout, stretch=1)

        # Inference metrics row
        metrics_row = QHBoxLayout()
        self.card_latency = MetricBox("Inference Latency", STATUS_NOT_CALCULATED)
        self.card_resolution = MetricBox("Processing Resolution", "256 × 256")
        self.card_lesion_detected = MetricBox("Lesion Area (Pixels)", STATUS_NOT_CALCULATED)

        metrics_row.addWidget(self.card_latency)
        metrics_row.addWidget(self.card_resolution)
        metrics_row.addWidget(self.card_lesion_detected)

        layout.addLayout(metrics_row)

        self.run_btn.clicked.connect(self._handle_run_segmentation)

    def set_active_image(self, image_path: Path, metadata: TransmissionMetadata):
        """Sets active image for segmentation inference."""
        self.current_image_path = image_path
        self.current_metadata = metadata
        if image_path.exists():
            pix = QPixmap(str(image_path))
            self.recv_preview.set_pixmap(
                pix,
                caption=f"{metadata.image_id} | Q{metadata.compression_quality} | {metadata.image_width}×{metadata.image_height}",
            )
            self.pred_mask_preview.clear_image("Ready to segment")
            self.overlay_preview.clear_image("Ready to segment")
            self.card_latency.set_value(STATUS_NOT_CALCULATED)
            self.card_lesion_detected.set_value(STATUS_NOT_CALCULATED)

    def _handle_run_segmentation(self):
        if not self.current_image_path or not self.current_image_path.exists():
            QMessageBox.warning(self, "No Image", "Please select an image from the Image Viewer first.")
            return

        if not self.model_manager.is_loaded or self.model_manager.model is None:
            QMessageBox.warning(
                self,
                "No Model Loaded",
                "Please load a pre-trained frozen U-Net model from the Model Management tab first.",
            )
            return

        img_id = self.current_metadata.image_id if self.current_metadata else self.current_image_path.stem
        success, result, msg = self.engine.run_inference(self.current_image_path, img_id)

        if not success or result is None:
            QMessageBox.critical(self, "Inference Failed", msg)
            return

        self.current_result = result

        # Display predicted mask
        h, w = result.predicted_mask.shape
        q_mask = QImage(result.predicted_mask.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.pred_mask_preview.set_pixmap(QPixmap.fromImage(q_mask), caption=f"256 × 256 Binary Mask")

        # Display lesion overlay
        oh, ow, oc = result.overlay_image.shape
        bytes_per_line = oc * ow
        q_overlay = QImage(
            result.overlay_image.data,
            ow,
            oh,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        self.overlay_preview.set_pixmap(QPixmap.fromImage(q_overlay), caption="Lesion Boundary Overlay")

        # Update cards
        self.card_latency.set_value(f"{result.inference_time_ms:.2f} ms")
        self.card_lesion_detected.set_value(f"{result.lesion_area_pixels:,} px")

        if self.current_metadata is not None:
            self.segmentation_completed.emit(result, self.current_metadata)
