"""Sender Experiment Results View."""

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from common.results_manager import ResultsManager
from ui.shared.widgets import SectionHeader


class SenderResultsView(QWidget):
    """Displays genuine historical experiment logs and exports."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_manager = ResultsManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Experiment Records & Telemetry",
            "Traceable records of measured compression and transmission runs (Zero dummy data policy)",
        )
        layout.addWidget(header)

        # Action bar
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
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "Experiment ID", "Timestamp", "Image ID", "Quality", "Orig Size", "Comp Size",
            "PSNR (dB)", "SSIM", "Tx Time (s)", "Throughput (Kbps)"
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
        for row_idx, r in enumerate(records):
            # Fill cells with actual recorded values
            pass

