"""Receiver Experiment Results & Evaluation Telemetry View."""

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from common.results_manager import ResultsManager
from ui.shared.widgets import SectionHeader


class ReceiverResultsView(QWidget):
    """Complete traceable log of received frames, transmission metrics, and segmentation scores."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_manager = ResultsManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Receiver Evaluation Records",
            "Traceable records of received images, measured transmission metrics, Dice, and IoU scores",
        )
        layout.addWidget(header)

        # Action row
        act_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Records")
        self.refresh_btn.setObjectName("SecondaryBtn")
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setObjectName("SecondaryBtn")

        act_row.addWidget(self.refresh_btn)
        act_row.addWidget(self.export_btn)
        act_row.addStretch()
        layout.addLayout(act_row)

        # Results table
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels([
            "Experiment ID", "Image ID", "Quality", "Comp Size", "Tx Time (s)",
            "Throughput (Kbps)", "Checksum", "Dice", "IoU", "PSNR", "SSIM"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        self.refresh_btn.clicked.connect(self.load_records)
        self.load_records()

    def load_records(self):
        """Loads records from database (Zero-Dummy: stays empty if no records yet)."""
        records = self.results_manager.fetch_all()
        self.table.setRowCount(len(records))

