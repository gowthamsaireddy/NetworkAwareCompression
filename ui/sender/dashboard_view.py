"""Sender Dashboard View."""

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from shared.constants import STATUS_IDLE, STATUS_NO_DATA
from ui.shared.widgets import MetricBox, SectionHeader


class SenderDashboardView(QWidget):
    """High-level operational overview for the Sender node."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Sender Node Dashboard",
            "Operating node: Ubuntu Linux VM | Role: Dataset preparation, compression, tc/netem shaping, TCP transmission",
        )
        layout.addWidget(header)

        # Metrics grid
        grid = QGridLayout()
        grid.setSpacing(12)

        self.card_dataset = MetricBox("Scanned Images", STATUS_NO_DATA)
        self.card_network = MetricBox("Network Shaping", "Unshaped (Default)")
        self.card_target = MetricBox("Receiver Endpoint", "127.0.0.1:5000")
        self.card_transfers = MetricBox("Active Status", STATUS_IDLE)

        grid.addWidget(self.card_dataset, 0, 0)
        grid.addWidget(self.card_network, 0, 1)
        grid.addWidget(self.card_target, 1, 0)
        grid.addWidget(self.card_transfers, 1, 1)

        layout.addLayout(grid)

        # System Information Card
        info_frame = QFrame()
        info_frame.setObjectName("CardFrame")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(8)

        info_title = QLabel("System & Hardware Status")
        info_title.setObjectName("CardTitle")
        info_layout.addWidget(info_title)

        self.info_details = QLabel(
            "• Operating System: Ubuntu Linux (Bridged Ethernet Interface)\n"
            "• Network Shaping Engine: Linux tc / netem\n"
            "• Deep Learning Model: Evaluated on Receiver (Windows 11)\n"
            "• Zero-Dummy-Data Policy: Active (All metrics reflect real physical experiments)"
        )
        self.info_details.setStyleSheet("color: #94a3b8; line-height: 1.4;")
        info_layout.addWidget(self.info_details)

        layout.addWidget(info_frame)
        layout.addStretch()

