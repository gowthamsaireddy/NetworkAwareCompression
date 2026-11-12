"""Sender Dashboard View with Live Operational Telemetry."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from shared.constants import STATUS_DISCONNECTED, STATUS_IDLE, STATUS_NO_DATA
from ui.shared.widgets import MetricBox, SectionHeader


class SenderDashboardView(QWidget):
    """Live operational monitoring dashboard for the Sender Node."""

    test_connection_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Sender Node Dashboard",
            "Operating node: Ubuntu Linux | Role: Dataset preparation, compression, tc/netem shaping, TCP transmission",
        )
        layout.addWidget(header)

        # Connection status & probe action bar
        conn_frame = QFrame()
        conn_frame.setObjectName("CardFrame")
        conn_layout = QHBoxLayout(conn_frame)
        conn_layout.setContentsMargins(12, 8, 12, 8)

        self.endpoint_lbl = QLabel("Receiver Target: 127.0.0.1:5000")
        self.endpoint_lbl.setStyleSheet("font-weight: 600; color: #38bdf8;")

        self.conn_status_badge = QLabel("Status: Disconnected")
        self.conn_status_badge.setStyleSheet("color: #ef4444; font-weight: 600;")

        self.test_conn_btn = QPushButton("Test Receiver Reachability")
        self.test_conn_btn.setObjectName("SecondaryBtn")
        self.test_conn_btn.clicked.connect(self.test_connection_requested.emit)

        conn_layout.addWidget(self.endpoint_lbl)
        conn_layout.addStretch()
        conn_layout.addWidget(self.conn_status_badge)
        conn_layout.addWidget(self.test_conn_btn)
        layout.addWidget(conn_frame)

        # 12-Card Telemetry Grid (4 rows x 3 columns)
        grid = QGridLayout()
        grid.setSpacing(10)

        # Row 0: Network & Image Metadata
        self.card_conn = MetricBox("Connection State", STATUS_DISCONNECTED)
        self.card_image = MetricBox("Selected Dataset Image", "None Selected")
        self.card_orig_size = MetricBox("Original Image Size", STATUS_NO_DATA)

        # Row 1: Compression Fidelity
        self.card_quality = MetricBox("Selected JPEG Quality", "Q70 (Default)")
        self.card_comp_size = MetricBox("Compressed Size", STATUS_NO_DATA)
        self.card_ratio = MetricBox("Compression Ratio", STATUS_NO_DATA)

        # Row 2: Objective Image Metrics & Channel State
        self.card_psnr_ssim = MetricBox("Fidelity (PSNR / SSIM)", STATUS_NO_DATA)
        self.card_profile = MetricBox("Active Network Profile", "Unshaped (Default)")
        self.card_wire_time = MetricBox("Actual Transmission Time", STATUS_NO_DATA)

        # Row 3: Live Wire Telemetry
        self.card_throughput = MetricBox("Measured Throughput", STATUS_NO_DATA)
        self.card_ack = MetricBox("ACK & Integrity Status", STATUS_IDLE)
        self.card_scanned = MetricBox("Total Scanned Images", "0")

        grid.addWidget(self.card_conn, 0, 0)
        grid.addWidget(self.card_image, 0, 1)
        grid.addWidget(self.card_orig_size, 0, 2)

        grid.addWidget(self.card_quality, 1, 0)
        grid.addWidget(self.card_comp_size, 1, 1)
        grid.addWidget(self.card_ratio, 1, 2)

        grid.addWidget(self.card_psnr_ssim, 2, 0)
        grid.addWidget(self.card_profile, 2, 1)
        grid.addWidget(self.card_wire_time, 2, 2)

        grid.addWidget(self.card_throughput, 3, 0)
        grid.addWidget(self.card_ack, 3, 1)
        grid.addWidget(self.card_scanned, 3, 2)

        layout.addLayout(grid)

        # Live Transmission Progress Bar
        prog_frame = QFrame()
        prog_frame.setObjectName("CardFrame")
        prog_layout = QVBoxLayout(prog_frame)
        prog_layout.setSpacing(4)
        prog_layout.addWidget(QLabel("Live Wire Transmission Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle (Ready)")
        prog_layout.addWidget(self.progress_bar)
        layout.addWidget(prog_frame)

        # System Information Card
        info_frame = QFrame()
        info_frame.setObjectName("CardFrame")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(6)

        info_title = QLabel("System Architecture & Experimental Method")
        info_title.setObjectName("CardTitle")
        info_layout.addWidget(info_title)

        self.info_details = QLabel(
            "• Operating System: Ubuntu Linux (Dedicated Telemedicine Sender)\n"
            "• Network Emulation Engine: Linux kernel traffic control (tc) & netem (qdisc)\n"
            "• Receiver Destination: Windows 11 Medical Node via Length-Prefixed Binary Protocol\n"
            "• Zero-Dummy-Data Policy: Active (All transmission times, throughputs, and fidelity metrics are physical measurements)"
        )
        self.info_details.setStyleSheet("color: #94a3b8; line-height: 1.4; font-size: 12px;")
        info_layout.addWidget(self.info_details)

        layout.addWidget(info_frame)
        layout.addStretch()

    def update_connection_state(self, is_connected: bool, msg: str = ""):
        if is_connected:
            self.conn_status_badge.setText("Status: Online / Connected")
            self.conn_status_badge.setStyleSheet("color: #10b981; font-weight: 600;")
            self.card_conn.set_value("Connected")
        else:
            self.conn_status_badge.setText(f"Status: {msg or 'Disconnected'}")
            self.conn_status_badge.setStyleSheet("color: #ef4444; font-weight: 600;")
            self.card_conn.set_value("Disconnected")
