"""Sender Dataset Management View."""

from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from common.config import DATA_DIR
from sender.dataset_manager import DatasetItem, DatasetManager
from shared.constants import STATUS_NO_DATA
from ui.shared.widgets import ImagePreviewBox, SectionHeader


class SenderDatasetView(QWidget):
    """Manages BUSI dataset scanning, file indexing, and preview without modifying originals."""

    image_selected = pyqtSignal(object)  # Emits selected DatasetItem
    dataset_scanned = pyqtSignal(int, dict)  # Emits total count and breakdown dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset_manager = DatasetManager()
        self.current_filtered_items: list[DatasetItem] = []
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

        # Directory selection & filter bar
        dir_frame = QFrame()
        dir_frame.setObjectName("CardFrame")
        dir_layout = QHBoxLayout(dir_frame)
        dir_layout.setContentsMargins(12, 10, 12, 10)

        default_busi = str(DATA_DIR / "BUSI")
        self.path_input = QLineEdit(default_busi)
        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.setObjectName("SecondaryBtn")
        self.scan_btn = QPushButton("Scan Dataset")

        self.category_filter = QComboBox()
        self.category_filter.addItems(["All Categories", "Benign", "Malignant", "Normal"])

        self.export_manifest_btn = QPushButton("Export Manifest")
        self.export_manifest_btn.setObjectName("SecondaryBtn")

        dir_layout.addWidget(QLabel("Dataset Directory:"))
        dir_layout.addWidget(self.path_input, stretch=1)
        dir_layout.addWidget(self.browse_btn)
        dir_layout.addWidget(self.category_filter)
        dir_layout.addWidget(self.scan_btn)
        dir_layout.addWidget(self.export_manifest_btn)

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
        self.scan_btn.clicked.connect(self._scan_dataset)
        self.category_filter.currentIndexChanged.connect(self._apply_filter)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.export_manifest_btn.clicked.connect(self._export_manifest)

    def _browse_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select BUSI Dataset Directory")
        if folder:
            self.path_input.setText(folder)

    def _scan_dataset(self):
        path_str = self.path_input.text().strip()
        path = Path(path_str)
        if not path.is_dir():
            QMessageBox.warning(self, "Invalid Directory", f"The directory does not exist:\n{path_str}")
            return

        self.dataset_manager.set_dataset_directory(path)
        items = self.dataset_manager.scan_dataset()

        # Emit scanned statistics
        stats = {
            "benign": len(self.dataset_manager.filter_by_category("benign")),
            "malignant": len(self.dataset_manager.filter_by_category("malignant")),
            "normal": len(self.dataset_manager.filter_by_category("normal")),
        }
        self.dataset_scanned.emit(len(items), stats)

        self._apply_filter()

        if not items:
            QMessageBox.information(
                self,
                "Scan Complete",
                f"Directory scanned, but no images found in {path_str}.\n"
                "Ensure dataset contains benign, malignant, or normal image files (.png/.jpg).",
            )
        else:
            QMessageBox.information(
                self,
                "Scan Complete",
                f"Successfully identified {len(items)} images:\n"
                f"• Benign: {stats['benign']}\n"
                f"• Malignant: {stats['malignant']}\n"
                f"• Normal: {stats['normal']}",
            )

    def _apply_filter(self):
        category = self.category_filter.currentText()
        if category == "All Categories":
            self.current_filtered_items = list(self.dataset_manager.items)
        else:
            self.current_filtered_items = self.dataset_manager.filter_by_category(category)

        self.table.setRowCount(len(self.current_filtered_items))
        for row_idx, item in enumerate(self.current_filtered_items):
            self.table.setItem(row_idx, 0, QTableWidgetItem(item.image_id))
            self.table.setItem(row_idx, 1, QTableWidgetItem(item.category.capitalize()))
            dim_str = f"{item.width} × {item.height}" if item.width else "Unknown"
            self.table.setItem(row_idx, 2, QTableWidgetItem(dim_str))
            self.table.setItem(row_idx, 3, QTableWidgetItem(str(item.file_size_bytes)))
            mask_str = f"{len(item.mask_paths)} mask file(s)" if item.mask_paths else "No mask"
            self.table.setItem(row_idx, 4, QTableWidgetItem(mask_str))

        if self.current_filtered_items:
            self.table.selectRow(0)
        else:
            self.preview_box.clear_image("No images match category filter")

    def _on_table_selection(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        if 0 <= row < len(self.current_filtered_items):
            item = self.current_filtered_items[row]
            if item.image_path.exists():
                pixmap = QPixmap(str(item.image_path))
                caption = f"{item.width} × {item.height} | {item.file_size_bytes:,} bytes | {len(item.mask_paths)} mask(s)"
                self.preview_box.set_pixmap(pixmap, caption=caption)
                self.image_selected.emit(item)

    def _export_manifest(self):
        if not self.dataset_manager.items:
            QMessageBox.warning(self, "No Scanned Data", "Please scan a dataset directory first.")
            return

        target_file, _ = QFileDialog.getSaveFileName(
            self, "Save Dataset Manifest", str(DATA_DIR / "manifest_busi.json"), "JSON Files (*.json)"
        )
        if target_file:
            path = self.dataset_manager.export_manifest(Path(target_file))
            QMessageBox.information(self, "Manifest Exported", f"Dataset manifest successfully saved to:\n{path}")
