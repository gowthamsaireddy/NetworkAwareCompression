"""Sender Dataset Management View."""

from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from shared.constants import STATUS_NO_DATA
from ui.shared.widgets import ImagePreviewBox, SectionHeader


class SenderDatasetView(QWidget):
    """Manages BUSI dataset scanning, file indexing, and preview without modifying originals."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Dataset Management",
            "Select and index the Breast Ultrasound Images (BUSI) dataset without altering source files",
        )
        layout.addWidget(header)

        # Directory selection bar
        dir_frame = QFrame()
        dir_frame.setObjectName("CardFrame")
        dir_layout = QHBoxLayout(dir_frame)
        dir_layout.setContentsMargins(12, 10, 12, 10)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter or browse BUSI dataset directory path...")
        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.setObjectName("SecondaryBtn")
        self.scan_btn = QPushButton("Scan Dataset")

        dir_layout.addWidget(QLabel("Dataset Directory:"))
        dir_layout.addWidget(self.path_input, stretch=1)
        dir_layout.addWidget(self.browse_btn)
        dir_layout.addWidget(self.scan_btn)

        layout.addWidget(dir_frame)

        # Splitter: Table on left, Preview on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table of images
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Image ID", "Category", "Dimensions", "Size (Bytes)", "Masks Found"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        splitter.addWidget(self.table)

        # Preview panel
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_box = ImagePreviewBox("Ultrasound Preview", STATUS_NO_DATA)
        preview_layout.addWidget(self.preview_box)
        splitter.addWidget(preview_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        # Connect signals
        self.browse_btn.clicked.connect(self._browse_directory)

    def _browse_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select BUSI Dataset Directory")
        if folder:
            self.path_input.setText(folder)

