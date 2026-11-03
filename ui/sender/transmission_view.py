"""Sender TCP Transmission View."""

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from shared.constants import DEFAULT_PORT, STATUS_DISCONNECTED, STATUS_NOT_CALCULATED
from ui.shared.widgets import MetricBox, SectionHeader


class SenderTransmissionView(QWidget):
    """Controls TCP transmission of compressed images to the Receiver."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "TCP Image Transmission",
            "Transmit length-prefixed binary frames and SHA-256 metadata over direct Ethernet link",
        )
        layout.addWidget(header)

        # Connection settings frame
        conn_frame = QFrame()
        conn_frame.setObjectName("CardFrame")
        conn_layout = QHBoxLayout(conn_frame)

        self.ip_input = QLineEdit("127.0.0.1")
        self.port_input = QSpinBox()
        self.port_input.setRange(1024, 65535)
        self.port_input.setValue(DEFAULT_PORT)

        self.connect_btn = QPushButton("Connect to Receiver")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setObjectName("SecondaryBtn")
        self.disconnect_btn.setEnabled(False)

        conn_layout.addWidget(QLabel("Receiver Host/IP:"))
        conn_layout.addWidget(self.ip_input, stretch=1)
        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.port_input)
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.disconnect_btn)

        layout.addWidget(conn_frame)

        # Transfer controls & progress
        trans_frame = QFrame()
        trans_frame.setObjectName("CardFrame")
        trans_layout = QVBoxLayout(trans_frame)

        ctrl_row = QHBoxLayout()
        self.send_single_btn = QPushButton("Send Selected Image")
        self.send_batch_btn = QPushButton("Send Batch (Selected Category)")
        self.send_batch_btn.setObjectName("SecondaryBtn")
        ctrl_row.addWidget(self.send_single_btn)
        ctrl_row.addWidget(self.send_batch_btn)
        ctrl_row.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle (0%)")

        trans_layout.addLayout(ctrl_row)
        trans_layout.addWidget(QLabel("Transfer Progress:"))
        trans_layout.addWidget(self.progress_bar)

        layout.addWidget(trans_frame)

        # Measured transmission metrics
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        self.card_time = MetricBox("Measured Transmission Time", STATUS_NOT_CALCULATED)
        self.card_throughput = MetricBox("Measured Throughput", STATUS_NOT_CALCULATED)
        self.card_ack = MetricBox("Receiver Checksum ACK", "Waiting")
        self.card_status = MetricBox("Link Status", STATUS_DISCONNECTED)

        metrics_grid.addWidget(self.card_status, 0, 0)
        metrics_grid.addWidget(self.card_time, 0, 1)
        metrics_grid.addWidget(self.card_throughput, 1, 0)
        metrics_grid.addWidget(self.card_ack, 1, 1)

        layout.addLayout(metrics_grid)

        # Transfer history table
        layout.addWidget(QLabel("Transfer History Log:"))
        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Image ID", "Payload (Bytes)", "Tx Time (s)", "Throughput (Kbps)", "ACK Status"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.history_table, stretch=1)

