"""Receiver Image Viewer & Inspection View."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from shared.constants import STATUS_NO_DATA
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class ReceiverImageView(QWidget):
    """Detailed visual inspection and navigation of received ultrasound images."""

    def __init__(self, parent=None):
        super().__init__(parent)
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

