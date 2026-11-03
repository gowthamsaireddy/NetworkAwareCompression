"""Sender JPEG Compression & Image Quality Analysis View."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from shared.constants import STATUS_NOT_CALCULATED
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class SenderCompressionView(QWidget):
    """Interactive variable-rate JPEG compression and distortion evaluation."""

    def __init__(self, parent=None):
        super().__init__(parent)
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

