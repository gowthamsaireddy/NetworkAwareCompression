"""Receiver Segmentation Inference View."""

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from shared.constants import STATUS_NOT_CALCULATED
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class ReceiverSegmentationView(QWidget):
    """Executes frozen U-Net inference to generate predicted lesion masks and visual overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
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

