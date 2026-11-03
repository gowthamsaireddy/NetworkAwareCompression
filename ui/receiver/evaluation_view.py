"""Receiver Segmentation Evaluation View (Dice & IoU)."""

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from shared.constants import STATUS_NOT_CALCULATED
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class ReceiverEvaluationView(QWidget):
    """Comparative evaluation against verified ground-truth masks (Dice and IoU metrics)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Evaluation & Ground-Truth Comparison",
            "Evaluate segmentation accuracy against verified ground-truth masks using genuine Dice and IoU metrics",
        )
        layout.addWidget(header)

        # Action bar
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("CardFrame")
        ctrl_layout = QHBoxLayout(ctrl_frame)

        self.eval_btn = QPushButton("Evaluate Current Image Against Ground Truth")
        self.eval_all_btn = QPushButton("Evaluate Entire Batch")
        self.eval_all_btn.setObjectName("SecondaryBtn")

        ctrl_layout.addWidget(self.eval_btn)
        ctrl_layout.addWidget(self.eval_all_btn)
        ctrl_layout.addStretch()

        layout.addWidget(ctrl_frame)

        # 5-Panel Display:
        # Original Image | Compressed Image | Ground-Truth Mask | Predicted Mask | Overlay
        preview_grid = QGridLayout()
        preview_grid.setSpacing(10)

        self.orig_box = ImagePreviewBox("Original Uncompressed", "No image")
        self.comp_box = ImagePreviewBox("Received Compressed", "No image")
        self.gt_box = ImagePreviewBox("Ground-Truth Mask", "No ground truth")
        self.pred_box = ImagePreviewBox("Predicted Mask", "No prediction")
        self.overlay_box = ImagePreviewBox("Ground-Truth vs Prediction Overlay", "No overlay")

        preview_grid.addWidget(self.orig_box, 0, 0)
        preview_grid.addWidget(self.comp_box, 0, 1)
        preview_grid.addWidget(self.gt_box, 0, 2)
        preview_grid.addWidget(self.pred_box, 1, 0)
        preview_grid.addWidget(self.overlay_box, 1, 1, 1, 2)

        layout.addLayout(preview_grid, stretch=1)

        # Quantitative Metrics Grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        self.card_dice = MetricBox("Dice Similarity Coefficient", STATUS_NOT_CALCULATED)
        self.card_iou = MetricBox("Intersection over Union (IoU)", STATUS_NOT_CALCULATED)
        self.card_gt_match = MetricBox("Ground Truth Status", "Unverified")
        self.card_quality = MetricBox("Compression Quality", STATUS_NOT_CALCULATED)

        metrics_grid.addWidget(self.card_dice, 0, 0)
        metrics_grid.addWidget(self.card_iou, 0, 1)
        metrics_grid.addWidget(self.card_gt_match, 0, 2)
        metrics_grid.addWidget(self.card_quality, 0, 3)

        layout.addLayout(metrics_grid)

