"""Receiver TCP Server Management View."""

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from shared.constants import DEFAULT_HOST, DEFAULT_PORT, STATUS_WAITING
from ui.shared.widgets import MetricBox, SectionHeader


class ReceiverTCPView(QWidget):
    """Manages the TCP listening daemon and incoming transfer streams."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "TCP Image Receiver",
            "Listen for inbound sender connections, assemble binary frames, and verify SHA-256 integrity",
        )
        layout.addWidget(header)

        # Server configuration controls
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("CardFrame")
        ctrl_layout = QHBoxLayout(ctrl_frame)

        self.bind_ip_input = QLineEdit(DEFAULT_HOST)
        self.bind_port_input = QSpinBox()
        self.bind_port_input.setRange(1024, 65535)
        self.bind_port_input.setValue(DEFAULT_PORT)

        self.start_btn = QPushButton("Start Listening Server")
        self.stop_btn = QPushButton("Stop Server")
        self.stop_btn.setObjectName("DangerBtn")
        self.stop_btn.setEnabled(False)

        ctrl_layout.addWidget(QLabel("Listen IP:"))
        ctrl_layout.addWidget(self.bind_ip_input)
        ctrl_layout.addWidget(QLabel("Port:"))
        ctrl_layout.addWidget(self.bind_port_input)
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addWidget(self.stop_btn)

        layout.addWidget(ctrl_frame)

        # Status cards
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        self.card_status = MetricBox("Server Status", "Stopped")
        self.card_last_img = MetricBox("Last Received Image", "None")
        self.card_checksum = MetricBox("Checksum Verification", STATUS_WAITING)
        self.card_client = MetricBox("Connected Client", "None")

        metrics_grid.addWidget(self.card_status, 0, 0)
        metrics_grid.addWidget(self.card_client, 0, 1)
        metrics_grid.addWidget(self.card_last_img, 1, 0)
        metrics_grid.addWidget(self.card_checksum, 1, 1)

        layout.addLayout(metrics_grid)

        # Live reception stream table
        layout.addWidget(QLabel("Incoming Stream Log:"))
        self.log_table = QTableWidget(0, 6)
        self.log_table.setHorizontalHeaderLabels([
            "Timestamp", "Image ID", "Quality", "Payload (Bytes)", "SHA-256 Verified", "ACK Status"
        ])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.log_table, stretch=1)

