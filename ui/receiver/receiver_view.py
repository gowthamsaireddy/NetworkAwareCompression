"""Receiver TCP Server Management View."""

from pathlib import Path
from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from receiver.tcp_server import TCPReceiverServer
from shared.constants import DEFAULT_HOST, DEFAULT_PORT, STATUS_WAITING
from shared.metadata import TransmissionMetadata
from ui.shared.widgets import MetricBox, SectionHeader


class ReceiverSignalBridge(QObject):
    """Bridge for routing TCP background thread events to PyQt main thread slots."""

    status_changed = pyqtSignal(bool, str)
    client_connected = pyqtSignal(str)
    client_disconnected = pyqtSignal(str)
    image_received = pyqtSignal(object, object, bool, object, str)  # meta, bytes, verified, path, timestamp


class ReceiverTCPView(QWidget):
    """Manages the TCP listening daemon and incoming transfer streams."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.server = TCPReceiverServer()
        self.bridge = ReceiverSignalBridge()
        self._wire_server_callbacks()
        self.init_ui()

    def _wire_server_callbacks(self):
        # TCP server thread calls these; bridge emits Qt signals safely
        self.server.on_status_changed = lambda running, msg: self.bridge.status_changed.emit(running, msg)
        self.server.on_client_connected = lambda addr: self.bridge.client_connected.emit(addr)
        self.server.on_client_disconnected = lambda addr: self.bridge.client_disconnected.emit(addr)
        self.server.on_image_received = lambda meta, b, v, p, t: self.bridge.image_received.emit(meta, b, v, p, t)

        # Connect Qt signals to GUI slot handlers
        self.bridge.status_changed.connect(self._on_server_status)
        self.bridge.client_connected.connect(self._on_client_connected)
        self.bridge.client_disconnected.connect(self._on_client_disconnected)
        self.bridge.image_received.connect(self._on_image_received)

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
        self.card_client = MetricBox("Connected Client", "None")
        self.card_last_img = MetricBox("Last Received Image", "None")
        self.card_checksum = MetricBox("Checksum Verification", STATUS_WAITING)

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

        # Connect button actions
        self.start_btn.clicked.connect(self._handle_start)
        self.stop_btn.clicked.connect(self._handle_stop)

    def _handle_start(self):
        host = self.bind_ip_input.text().strip()
        port = self.bind_port_input.value()
        self.server.host = host
        self.server.port = port
        success, msg = self.server.start()
        if not success:
            QMessageBox.critical(self, "Server Bind Error", msg)

    def _handle_stop(self):
        self.server.stop()

    def _on_server_status(self, is_running: bool, message: str):
        if is_running:
            self.card_status.set_value(f"Listening on {self.server.port}")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.bind_ip_input.setEnabled(False)
            self.bind_port_input.setEnabled(False)
        else:
            self.card_status.set_value("Stopped")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.bind_ip_input.setEnabled(True)
            self.bind_port_input.setEnabled(True)
            self.card_client.set_value("None")

    def _on_client_connected(self, client_addr: str):
        self.card_client.set_value(client_addr)

    def _on_client_disconnected(self, client_addr: str):
        self.card_client.set_value(f"Disconnected ({client_addr})")

    def _on_image_received(
        self,
        metadata: TransmissionMetadata,
        image_bytes: bytes,
        verified: bool,
        save_path: Path | None,
        timestamp: str,
    ):
        self.card_last_img.set_value(metadata.image_id)
        if verified:
            self.card_checksum.set_value("Verified (SHA-256 Match)")
        else:
            self.card_checksum.set_value("Failed (Checksum Mismatch)")

        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        self.log_table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.log_table.setItem(row, 1, QTableWidgetItem(metadata.image_id))
        self.log_table.setItem(row, 2, QTableWidgetItem(f"Q{metadata.compression_quality}"))
        self.log_table.setItem(row, 3, QTableWidgetItem(str(metadata.image_byte_length)))
        ver_text = "Verified" if verified else "Mismatch"
        self.log_table.setItem(row, 4, QTableWidgetItem(ver_text))
        ack_text = "ACK_OK (0)" if verified else "ACK_MISMATCH (1)"
        self.log_table.setItem(row, 5, QTableWidgetItem(ack_text))
