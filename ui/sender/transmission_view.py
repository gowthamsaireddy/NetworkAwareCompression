"""Sender TCP Transmission View."""

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sender.tcp_client import TCPTransmissionClient, TransmissionReport
from shared.constants import (
    DEFAULT_PORT,
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    STATUS_NOT_CALCULATED,
)
from shared.metadata import TransmissionMetadata
from ui.shared.widgets import MetricBox, SectionHeader


class SingleTransmissionWorker(QThread):
    """Background worker to transmit an image over TCP without freezing the GUI."""

    progress_updated = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(object)  # TransmissionReport

    def __init__(self, client: TCPTransmissionClient, metadata: TransmissionMetadata, image_bytes: bytes):
        super().__init__()
        self.client = client
        self.metadata = metadata
        self.image_bytes = image_bytes

    def run(self):
        def _cb(curr, total):
            self.progress_updated.emit(curr, total)

        report = self.client.send_image(self.metadata, self.image_bytes, progress_callback=_cb)
        self.finished.emit(report)


class SenderTransmissionView(QWidget):
    """Controls TCP transmission of compressed images to the Receiver."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client = TCPTransmissionClient()
        self._current_metadata: TransmissionMetadata | None = None
        self._current_image_bytes: bytes | None = None
        self._worker: SingleTransmissionWorker | None = None
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
        self.send_single_btn = QPushButton("Send Current Frame")
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

        self.card_status = MetricBox("Link Status", STATUS_DISCONNECTED)
        self.card_time = MetricBox("Measured Transmission Time", STATUS_NOT_CALCULATED)
        self.card_throughput = MetricBox("Measured Throughput", STATUS_NOT_CALCULATED)
        self.card_ack = MetricBox("Receiver Checksum ACK", "Waiting")

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

        # Connect actions
        self.connect_btn.clicked.connect(self._handle_connect)
        self.disconnect_btn.clicked.connect(self._handle_disconnect)
        self.send_single_btn.clicked.connect(self._handle_send_single)

    def _handle_connect(self):
        host = self.ip_input.text().strip()
        port = self.port_input.value()
        self.client.host = host
        self.client.port = port
        success, msg = self.client.connect()
        if success:
            self.card_status.set_value(STATUS_CONNECTED)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
        else:
            self.card_status.set_value(STATUS_DISCONNECTED)
            QMessageBox.critical(self, "Connection Error", msg)

    def _handle_disconnect(self):
        self.client.disconnect()
        self.card_status.set_value(STATUS_DISCONNECTED)
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)

    def set_payload(self, metadata: TransmissionMetadata, image_bytes: bytes):
        """Sets the active payload to be transmitted."""
        self._current_metadata = metadata
        self._current_image_bytes = image_bytes

    def _handle_send_single(self):
        if not self._current_metadata or not self._current_image_bytes:
            QMessageBox.warning(
                self,
                "No Image Loaded",
                "Please select and compress an ultrasound image first (or use the test transmitter).",
            )
            return

        self.send_single_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Transmitting: %p%")

        self._worker = SingleTransmissionWorker(
            self.client,
            self._current_metadata,
            self._current_image_bytes,
        )
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished.connect(self._on_transfer_finished)
        self._worker.start()

    def _on_progress(self, curr: int, total: int):
        if total > 0:
            pct = int((curr / total) * 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"Transmitting: {curr}/{total} bytes ({pct}%)")

    def _on_transfer_finished(self, report: TransmissionReport):
        self.send_single_btn.setEnabled(True)
        if report.is_success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Transfer Succeeded (ACK Verified)")
            self.card_time.set_value(f"{report.transmission_time_sec:.4f} s")
            self.card_throughput.set_value(f"{report.throughput_kbps:.2f} Kbps")
            self.card_ack.set_value("Verified (ACK_OK)")
        else:
            self.progress_bar.setFormat("Transfer Failed")
            self.card_ack.set_value(f"Failed ({report.error_message or 'Unverified'})")
            QMessageBox.warning(self, "Transfer Unverified", f"Transmission unverified: {report.error_message}")

        # Add to history table
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        self.history_table.setItem(row, 0, QTableWidgetItem(report.timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(report.image_id))
        self.history_table.setItem(row, 2, QTableWidgetItem(str(report.bytes_sent)))
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{report.transmission_time_sec:.4f}"))
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{report.throughput_kbps:.2f}"))
        ack_text = "Verified" if report.is_success else f"Error ({report.ack_status})"
        self.history_table.setItem(row, 5, QTableWidgetItem(ack_text))
