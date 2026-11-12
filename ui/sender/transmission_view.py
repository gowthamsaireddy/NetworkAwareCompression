"""Sender TCP Transmission View with Single and Automated Batch Modes."""

from typing import List, Optional
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from common.config import get_sender_config
from common.results_manager import ExperimentRecord, ResultsManager
from sender.adaptive_controller import AdaptiveController
from sender.dataset_manager import DatasetItem
from sender.experiment_orchestrator import ExperimentOrchestrator
from sender.network_control import (
    NetworkController,
    NetworkProfile,
    compute_theoretical_transmission_time,
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


class BatchExperimentWorker(QThread):
    """Background worker executing automated multi-image experimental sequences."""

    step_updated = pyqtSignal(int, int, str)  # current, total, desc
    trial_finished = pyqtSignal(object)  # ExperimentRecord
    batch_completed = pyqtSignal(int, int, str)  # completed, failed, summary

    def __init__(
        self,
        orchestrator: ExperimentOrchestrator,
        items: List[DatasetItem],
        quality_strategy: str,
        fixed_qualities: List[int],
        profile: Optional[NetworkProfile],
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.items = items
        self.quality_strategy = quality_strategy
        self.fixed_qualities = fixed_qualities
        self.profile = profile
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        def _cb(curr, total, desc, rec):
            self.step_updated.emit(curr, total, desc)
            if rec:
                self.trial_finished.emit(rec)

        def _is_stopped():
            return self._stop_requested

        completed, failed, recs, msg = self.orchestrator.run_batch_experiment(
            items=self.items,
            quality_strategy=self.quality_strategy,
            fixed_qualities=self.fixed_qualities,
            profiles=[self.profile],
            progress_callback=_cb,
            stop_requested=_is_stopped,
        )
        self.batch_completed.emit(completed, failed, msg)


class SenderTransmissionView(QWidget):
    """Controls TCP transmission of compressed images to the Receiver."""

    trial_recorded = pyqtSignal(object)  # Emits ExperimentRecord
    connection_changed = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_sender_config()
        self.client = TCPTransmissionClient(
            host=self.config.receiver_ip,
            port=self.config.receiver_port,
            timeout=self.config.socket_timeout,
        )
        self.results_manager = ResultsManager()
        self.network_controller = NetworkController()
        self.adaptive_controller = AdaptiveController()
        self.orchestrator = ExperimentOrchestrator(
            tcp_client=self.client,
            network_controller=self.network_controller,
            adaptive_controller=self.adaptive_controller,
            results_manager=self.results_manager,
        )

        self._current_metadata: Optional[TransmissionMetadata] = None
        self._current_image_bytes: Optional[bytes] = None
        self._current_psnr: Optional[float] = None
        self._current_ssim: Optional[float] = None
        self._active_profile: Optional[NetworkProfile] = None
        self._all_dataset_items: List[DatasetItem] = []

        self._worker: Optional[SingleTransmissionWorker] = None
        self._batch_worker: Optional[BatchExperimentWorker] = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "TCP Image Transmission & Automated Experiments",
            "Transmit length-prefixed binary frames and SHA-256 metadata over direct link or execute automated batches",
        )
        layout.addWidget(header)

        # Connection settings frame
        conn_frame = QFrame()
        conn_frame.setObjectName("CardFrame")
        conn_layout = QHBoxLayout(conn_frame)

        self.ip_input = QLineEdit(self.config.receiver_ip)
        self.port_input = QSpinBox()
        self.port_input.setRange(1024, 65535)
        self.port_input.setValue(self.config.receiver_port)

        self.probe_btn = QPushButton("Probe Receiver")
        self.probe_btn.setObjectName("SecondaryBtn")
        self.connect_btn = QPushButton("Connect to Receiver")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setObjectName("SecondaryBtn")
        self.disconnect_btn.setEnabled(False)

        conn_layout.addWidget(QLabel("Receiver Host/IP:"))
        conn_layout.addWidget(self.ip_input, stretch=1)
        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.port_input)
        conn_layout.addWidget(self.probe_btn)
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.disconnect_btn)

        layout.addWidget(conn_frame)

        # Transfer controls & progress
        trans_frame = QFrame()
        trans_frame.setObjectName("CardFrame")
        trans_layout = QVBoxLayout(trans_frame)

        ctrl_row = QHBoxLayout()
        self.send_single_btn = QPushButton("Send Active Frame")
        self.start_batch_btn = QPushButton("Start Automated Experiment Run")
        self.start_batch_btn.setObjectName("SecondaryBtn")
        self.abort_batch_btn = QPushButton("Abort Run")
        self.abort_batch_btn.setObjectName("DangerBtn")
        self.abort_batch_btn.setEnabled(False)

        ctrl_row.addWidget(self.send_single_btn)
        ctrl_row.addWidget(self.start_batch_btn)
        ctrl_row.addWidget(self.abort_batch_btn)
        ctrl_row.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle (0%)")

        self.batch_status_lbl = QLabel("Mode: Single image transmission ready")
        self.batch_status_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        trans_layout.addLayout(ctrl_row)
        trans_layout.addWidget(QLabel("Transfer / Experiment Progress:"))
        trans_layout.addWidget(self.progress_bar)
        trans_layout.addWidget(self.batch_status_lbl)

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
        self.probe_btn.clicked.connect(self._handle_probe)
        self.connect_btn.clicked.connect(self._handle_connect)
        self.disconnect_btn.clicked.connect(self._handle_disconnect)
        self.send_single_btn.clicked.connect(self._handle_send_single)
        self.start_batch_btn.clicked.connect(self._handle_start_batch)
        self.abort_batch_btn.clicked.connect(self._handle_abort_batch)

    def set_dataset_items(self, items: List[DatasetItem]):
        """Caches scanned dataset items for batch processing."""
        self._all_dataset_items = items

    def _sync_endpoint_from_inputs(self):
        host = self.ip_input.text().strip()
        port = self.port_input.value()
        self.client.set_endpoint(host, port)

    def _handle_probe(self):
        self._sync_endpoint_from_inputs()
        avail, rtt, msg = self.client.check_receiver_availability(timeout=2.0)
        if avail:
            QMessageBox.information(self, "Receiver Available", msg)
            self.connection_changed.emit(True, msg)
        else:
            QMessageBox.warning(self, "Receiver Unreachable", msg)
            self.connection_changed.emit(False, msg)

    def _handle_connect(self):
        self._sync_endpoint_from_inputs()
        success, msg = self.client.connect()
        if success:
            self.card_status.set_value(STATUS_CONNECTED)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.connection_changed.emit(True, f"Connected to {self.client.host}:{self.client.port}")
        else:
            self.card_status.set_value(STATUS_DISCONNECTED)
            self.connection_changed.emit(False, msg)
            QMessageBox.critical(self, "Connection Error", msg)

    def _handle_disconnect(self):
        self.client.disconnect()
        self.card_status.set_value(STATUS_DISCONNECTED)
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.connection_changed.emit(False, "Disconnected")

    def set_payload(
        self,
        metadata: TransmissionMetadata,
        image_bytes: bytes,
        psnr: float | None = None,
        ssim: float | None = None,
    ):
        """Sets the active payload to be transmitted along with distortion metrics."""
        self._current_metadata = metadata
        self._current_image_bytes = image_bytes
        self._current_psnr = psnr
        self._current_ssim = ssim

    def set_active_network_profile(self, profile: NetworkProfile | None):
        """Sets active traffic shaping profile for telemetry and theoretical time calculation."""
        self._active_profile = profile

    def _handle_send_single(self):
        if not self._current_metadata or not self._current_image_bytes:
            QMessageBox.warning(
                self,
                "No Image Loaded",
                "Please select and compress an ultrasound image first from Dataset Management.",
            )
            return

        self._sync_endpoint_from_inputs()
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
            self.card_throughput.set_value(f"{report.throughput_kbps:,.2f} Kbps")
            self.card_ack.set_value("Verified (ACK_OK)")

            # Record genuine experiment run in SQLite & CSV
            if self._current_metadata is not None:
                meta = self._current_metadata
                ratio = (meta.original_file_size / meta.compressed_file_size) if meta.compressed_file_size > 0 else 0.0

                bw_str = self._active_profile.bandwidth if self._active_profile else "unshaped"
                loss_val = self._active_profile.loss_percent if self._active_profile else 0.0
                delay_val = float(self._active_profile.delay_ms) if self._active_profile else 0.0
                bw_bps = self._active_profile.bandwidth_bps if self._active_profile else 0.0
                theo_time = compute_theoretical_transmission_time(meta.image_byte_length, bw_bps)

                record = ExperimentRecord(
                    experiment_id=meta.experiment_id,
                    timestamp=report.timestamp,
                    image_id=meta.image_id,
                    compression_quality=meta.compression_quality,
                    original_file_size=meta.original_file_size,
                    compressed_file_size=meta.compressed_file_size,
                    compression_ratio=ratio,
                    psnr=self._current_psnr,
                    ssim=self._current_ssim,
                    bandwidth_limit=bw_str,
                    packet_loss_rate=loss_val,
                    network_delay_ms=delay_val,
                    theoretical_transmission_time_sec=theo_time,
                    transmission_time_sec=report.transmission_time_sec,
                    measured_throughput_kbps=report.throughput_kbps,
                    checksum_verified=report.checksum_verified,
                    dice_coefficient=None,
                    iou=None,
                    notes=f"ACK Code {report.ack_status}",
                )
                self.results_manager.record_experiment(record)
                self.trial_recorded.emit(record)
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
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{report.throughput_kbps:,.2f}"))
        ack_text = "Verified" if report.is_success else f"Error ({report.ack_status})"
        self.history_table.setItem(row, 5, QTableWidgetItem(ack_text))

    def _handle_start_batch(self):
        """Launches automated multi-image experimental batch dialog."""
        if not self._all_dataset_items:
            QMessageBox.warning(
                self,
                "No Dataset Items",
                "Please scan or load dataset in Dataset Management before launching batch experiments.",
            )
            return

        self._sync_endpoint_from_inputs()

        # Simple modal configuration for batch run
        dlg = QDialog(self)
        dlg.setWindowTitle("Configure Automated Experiment Run")
        dlg.resize(420, 260)
        d_layout = QVBoxLayout(dlg)

        d_layout.addWidget(QLabel("<b>Execution Scope:</b>"))
        scope_combo = QComboBox()
        scope_combo.addItem(f"First 10 Images ({min(10, len(self._all_dataset_items))} items)", 10)
        scope_combo.addItem(f"First 25 Images ({min(25, len(self._all_dataset_items))} items)", 25)
        scope_combo.addItem(f"All Scanned Images ({len(self._all_dataset_items)} items)", len(self._all_dataset_items))
        d_layout.addWidget(scope_combo)

        d_layout.addWidget(QLabel("<b>Compression Strategy:</b>"))
        strat_combo = QComboBox()
        strat_combo.addItem("Fixed Quality Sweep (Q30, Q50, Q70, Q90)", "sweep")
        strat_combo.addItem("Fixed Quality Q70 (Standard Balanced)", "q70")
        strat_combo.addItem("Adaptive Rate Controller (Dynamic Mapping)", "adaptive")
        d_layout.addWidget(strat_combo)

        prof_name = self._active_profile.name if self._active_profile else "Unshaped (Direct)"
        d_layout.addWidget(QLabel(f"<b>Active Network Profile:</b> {prof_name}"))

        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        d_layout.addWidget(bbox)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        limit = scope_combo.currentData()
        target_items = self._all_dataset_items[:limit]
        mode = strat_combo.currentData()

        if mode == "sweep":
            strategy = "fixed"
            qualities = [30, 50, 70, 90]
        elif mode == "q70":
            strategy = "fixed"
            qualities = [70]
        else:
            strategy = "adaptive"
            qualities = []

        self.send_single_btn.setEnabled(False)
        self.start_batch_btn.setEnabled(False)
        self.abort_batch_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        self._batch_worker = BatchExperimentWorker(
            orchestrator=self.orchestrator,
            items=target_items,
            quality_strategy=strategy,
            fixed_qualities=qualities,
            profile=self._active_profile,
        )
        self._batch_worker.step_updated.connect(self._on_batch_step)
        self._batch_worker.trial_finished.connect(self._on_batch_trial_finished)
        self._batch_worker.batch_completed.connect(self._on_batch_completed)
        self._batch_worker.start()

    def _handle_abort_batch(self):
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.stop()
            self.batch_status_lbl.setText("Aborting batch experiment run...")
            self.abort_batch_btn.setEnabled(False)

    def _on_batch_step(self, curr: int, total: int, desc: str):
        pct = int((curr / max(total, 1)) * 100)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"Batch: {curr}/{total} ({pct}%)")
        self.batch_status_lbl.setText(desc)

    def _on_batch_trial_finished(self, record: ExperimentRecord):
        self.trial_recorded.emit(record)

        # Insert into history table
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        self.history_table.setItem(row, 0, QTableWidgetItem(record.timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(record.image_id))
        self.history_table.setItem(row, 2, QTableWidgetItem(str(record.compressed_file_size)))
        self.history_table.setItem(row, 3, QTableWidgetItem(f"{record.transmission_time_sec:.4f}"))
        thru = record.measured_throughput_kbps or 0.0
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{thru:,.2f}"))
        self.history_table.setItem(row, 5, QTableWidgetItem("Verified" if record.checksum_verified else "Failed"))

    def _on_batch_completed(self, completed: int, failed: int, msg: str):
        self.send_single_btn.setEnabled(True)
        self.start_batch_btn.setEnabled(True)
        self.abort_batch_btn.setEnabled(False)
        self.batch_status_lbl.setText(msg)
        self.progress_bar.setFormat("Batch Complete (100%)")
        QMessageBox.information(self, "Experiment Batch Complete", msg)
