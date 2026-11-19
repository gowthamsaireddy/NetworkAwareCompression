"""Sender TCP Transmission View with Controlled Batch Execution and Live Telemetry."""

import json
from pathlib import Path
import time
from typing import List, Optional

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from common.config import get_sender_config
from common.logging_config import setup_logger
from common.results_manager import ExperimentRecord, ResultsManager
from sender.adaptive_controller import AdaptiveController
from sender.dataset_manager import DatasetItem
from sender.experiment_orchestrator import ExperimentOrchestrator
from sender.network_control import (
    PRESET_PROFILES,
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

logger = setup_logger("TransmissionView")


class SingleTransmissionWorker(QThread):
    """Background worker to transmit an image frame over TCP without blocking GUI."""

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
    """Background worker executing controlled, sequential multi-image experimental batches."""

    step_updated = pyqtSignal(int, int, str)  # current, total, desc
    trial_finished = pyqtSignal(object)  # ExperimentRecord
    batch_completed = pyqtSignal(int, int, int, str)  # completed, skipped, failed, summary

    def __init__(
        self,
        orchestrator: ExperimentOrchestrator,
        items: List[DatasetItem],
        compression_method: str = "NORMAL",
        fixed_qualities: Optional[List[int]] = None,
        adaptive_margins: Optional[List[int]] = None,
        adaptive_roi_quality: int = 90,
        adaptive_bg_quality: int = 20,
        roi_mode: str = "ML_MODEL",
        profiles: Optional[List[Optional[NetworkProfile]]] = None,
        skip_completed: bool = True,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.items = items
        self.compression_method = compression_method
        self.fixed_qualities = fixed_qualities
        self.adaptive_margins = adaptive_margins
        self.adaptive_roi_quality = adaptive_roi_quality
        self.adaptive_bg_quality = adaptive_bg_quality
        self.roi_mode = roi_mode
        self.profiles = profiles
        self.skip_completed = skip_completed
        self.max_batch_size = max_batch_size

        self._stop_requested = False
        self._is_paused = False

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def is_paused(self) -> bool:
        return self._is_paused

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            def _cb(curr, total, desc, rec):
                self.step_updated.emit(curr, total, desc)
                if rec:
                    self.trial_finished.emit(rec)

            def _is_stopped():
                return self._stop_requested

            def _is_paused():
                return self._is_paused

            completed, skipped, failed, recs, msg = self.orchestrator.run_batch_experiment(
                items=self.items,
                compression_method=self.compression_method,
                fixed_qualities=self.fixed_qualities,
                adaptive_margins=self.adaptive_margins,
                adaptive_roi_quality=self.adaptive_roi_quality,
                adaptive_bg_quality=self.adaptive_bg_quality,
                roi_mode=self.roi_mode,
                profiles=self.profiles,
                skip_completed=self.skip_completed,
                max_batch_size=self.max_batch_size,
                progress_callback=_cb,
                stop_requested=_is_stopped,
                pause_requested=_is_paused,
            )
            self.batch_completed.emit(completed, skipped, failed, msg)
        except Exception as e:
            logger.error(f"Batch execution exception: {e}")
            self.batch_completed.emit(0, 0, len(self.items), f"Batch error: {e}")


class SenderTransmissionView(QWidget):
    """Controls TCP transmission of compressed images with strict controlled execution."""

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
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "TCP Transmission & Controlled Experiment Execution",
            "Transmit binary image payloads over direct TCP link and conduct controlled, confirmed research batches",
        )
        layout.addWidget(header)

        # -------------------------------------------------------------
        # Connection Settings Frame
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Active Payload Info Banner
        # -------------------------------------------------------------
        self.active_payload_banner = QLabel("Active Transmission Payload: No image compressed yet")
        self.active_payload_banner.setStyleSheet(
            "background: #1e293b; color: #38bdf8; font-weight: 600; padding: 10px 14px; border-radius: 6px; font-size: 12px;"
        )
        layout.addWidget(self.active_payload_banner)

        # -------------------------------------------------------------
        # Transfer & Batch Controls Frame
        # -------------------------------------------------------------
        trans_frame = QFrame()
        trans_frame.setObjectName("CardFrame")
        trans_layout = QVBoxLayout(trans_frame)

        ctrl_row = QHBoxLayout()
        self.send_single_btn = QPushButton("Send Active Image Frame")
        self.send_single_btn.setStyleSheet(
            "background: #0284c7; color: white; font-weight: 700; padding: 8px 16px; border-radius: 6px;"
        )

        self.start_batch_btn = QPushButton("Configure Controlled Batch...")
        self.start_batch_btn.setObjectName("SecondaryBtn")

        self.pause_batch_btn = QPushButton("Pause Batch")
        self.pause_batch_btn.setEnabled(False)

        self.abort_batch_btn = QPushButton("Cancel / Abort Batch")
        self.abort_batch_btn.setObjectName("DangerBtn")
        self.abort_batch_btn.setEnabled(False)

        ctrl_row.addWidget(self.send_single_btn)
        ctrl_row.addWidget(self.start_batch_btn)
        ctrl_row.addWidget(self.pause_batch_btn)
        ctrl_row.addWidget(self.abort_batch_btn)
        ctrl_row.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Idle (0%)")

        self.batch_status_lbl = QLabel("Execution Status: Idle (Zero uncontrolled automatic background loops)")
        self.batch_status_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        trans_layout.addLayout(ctrl_row)
        trans_layout.addWidget(QLabel("Transfer / Controlled Batch Progress:"))
        trans_layout.addWidget(self.progress_bar)
        trans_layout.addWidget(self.batch_status_lbl)

        layout.addWidget(trans_frame)

        # -------------------------------------------------------------
        # Measured Transmission Telemetry Grid
        # -------------------------------------------------------------
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        self.card_status = MetricBox("Link Status", STATUS_DISCONNECTED)
        self.card_wire_time = MetricBox("Measured Wire Time", STATUS_NOT_CALCULATED)
        self.card_wire_throughput = MetricBox("Measured Wire Throughput", STATUS_NOT_CALCULATED)
        self.card_rtt = MetricBox("Round-Trip Latency (RTT)", STATUS_NOT_CALCULATED)
        self.card_ack = MetricBox("Receiver Checksum ACK", "Waiting")
        self.card_receiver_timers = MetricBox("Receiver Verification & Decode", STATUS_NOT_CALCULATED)

        metrics_grid.addWidget(self.card_status, 0, 0)
        metrics_grid.addWidget(self.card_wire_time, 0, 1)
        metrics_grid.addWidget(self.card_wire_throughput, 0, 2)
        metrics_grid.addWidget(self.card_rtt, 1, 0)
        metrics_grid.addWidget(self.card_ack, 1, 1)
        metrics_grid.addWidget(self.card_receiver_timers, 1, 2)

        layout.addLayout(metrics_grid)

        # -------------------------------------------------------------
        # Transfer History Log Table
        # -------------------------------------------------------------
        layout.addWidget(QLabel("Real-Time Measured Transfer History:"))
        self.history_table = QTableWidget(0, 7)
        self.history_table.setHorizontalHeaderLabels([
            "Timestamp", "Image ID", "Mode", "Wire Bytes", "Wire Tx (s)", "Throughput (Kbps)", "ACK Status"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setMinimumHeight(240)
        layout.addWidget(self.history_table)

        # Finalize scroll area setup
        scroll_area.setWidget(content_widget)
        outer_layout.addWidget(scroll_area)

        # Connect actions
        self.probe_btn.clicked.connect(self._handle_probe)
        self.connect_btn.clicked.connect(self._handle_connect)
        self.disconnect_btn.clicked.connect(self._handle_disconnect)
        self.send_single_btn.clicked.connect(self._handle_send_single)
        self.start_batch_btn.clicked.connect(self._handle_start_batch)
        self.pause_batch_btn.clicked.connect(self._handle_toggle_pause)
        self.abort_batch_btn.clicked.connect(self._handle_abort_batch)

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

        mode_str = metadata.compression_mode
        if mode_str == "REGION_ADAPTIVE":
            desc = (
                f"Active Payload: ML-Based Region-Adaptive | Image: {metadata.image_id} | "
                f"Margin: {metadata.region_adaptive_margin}px | ROI Q: {metadata.compression_quality} | "
                f"Payload: {len(image_bytes):,} B"
            )
        else:
            desc = (
                f"Active Payload: Normal JPEG Q{metadata.compression_quality} | "
                f"Image: {metadata.image_id} | Payload: {len(image_bytes):,} B"
            )
        self.active_payload_banner.setText(desc)

    def set_active_network_profile(self, profile: NetworkProfile | None):
        """Sets active traffic shaping profile for telemetry and theoretical time calculation."""
        self._active_profile = profile

    def set_dataset_items(self, items: List[DatasetItem]):
        """Caches scanned dataset items for controlled batch execution."""
        self._all_dataset_items = items

    def _handle_send_single(self):
        if not self._current_metadata or not self._current_image_bytes:
            QMessageBox.warning(
                self,
                "No Image Loaded",
                "Please select and compress an ultrasound image first from Dataset Management or Compression View.",
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
        meta = self._current_metadata

        if report.is_success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Transfer Succeeded (ACK Verified)")
            self.card_wire_time.set_value(f"{report.wire_time_sec:.4f} s")
            self.card_wire_throughput.set_value(f"{report.throughput_kbps:,.2f} Kbps")
            self.card_rtt.set_value(f"{report.round_trip_time_sec:.4f} s")
            self.card_ack.set_value("Verified (ACK_OK)")

            v_ms = report.verification_time_ms or 0.0
            d_ms = report.decoding_time_ms or 0.0
            self.card_receiver_timers.set_value(f"Verif: {v_ms:.2f}ms | Dec: {d_ms:.2f}ms")

            # Record genuine experiment run in SQLite & CSV with all 38+ metrics
            if meta is not None:
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
                    image_category=meta.image_category,
                    compression_method=meta.compression_mode,
                    compression_quality=meta.compression_quality,
                    region_adaptive_margin=meta.region_adaptive_margin,
                    compression_model_version=meta.compression_model_version,
                    original_file_size=meta.original_file_size,
                    compressed_file_size=meta.compressed_file_size,
                    transmitted_bytes=report.bytes_sent,
                    received_bytes=report.received_bytes,
                    compression_ratio=ratio,
                    psnr=self._current_psnr,
                    ssim=self._current_ssim,
                    bandwidth_limit=bw_str,
                    packet_loss_rate=loss_val,
                    network_delay_ms=delay_val,
                    theoretical_transmission_time_sec=theo_time,
                    transmission_time_sec=report.wire_time_sec,
                    measured_throughput_kbps=report.throughput_kbps,
                    connection_time_ms=report.connection_time_ms,
                    round_trip_time_sec=report.round_trip_time_sec,
                    effective_bandwidth_kbps=report.effective_bandwidth_kbps,
                    decoding_time_ms=report.decoding_time_ms,
                    verification_time_ms=report.verification_time_ms,
                    inference_time_ms=report.inference_time_ms,
                    checksum_verified=report.checksum_verified,
                    dice_coefficient=report.dice,
                    iou=report.iou,
                    experiment_status="SUCCESS",
                    transmission_mode="MANUAL",
                    notes=f"ACK OK | Mode: {meta.compression_mode}",
                )
                self.results_manager.record_experiment(record)
                self.trial_recorded.emit(record)
        else:
            self.progress_bar.setFormat("Transfer Failed")
            self.card_ack.set_value(f"Failed ({report.error_message or 'Unverified'})")
            QMessageBox.warning(self, "Transfer Unverified", f"Transmission unverified: {report.error_message}")

        # Add to history table
        mode_tag = meta.compression_mode if meta else "NORMAL"
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        self.history_table.setItem(row, 0, QTableWidgetItem(report.timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(report.image_id))
        self.history_table.setItem(row, 2, QTableWidgetItem(mode_tag))
        self.history_table.setItem(row, 3, QTableWidgetItem(str(report.bytes_sent)))
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{report.wire_time_sec:.4f}"))
        self.history_table.setItem(row, 5, QTableWidgetItem(f"{report.throughput_kbps:,.2f}"))
        ack_text = "Verified" if report.is_success else f"Error ({report.ack_status})"
        self.history_table.setItem(row, 6, QTableWidgetItem(ack_text))

    def _handle_start_batch(self):
        """Displays controlled batch execution dialog requiring explicit confirmation before running."""
        if not self._all_dataset_items:
            QMessageBox.warning(
                self,
                "No Dataset Items",
                "Please scan or load dataset in Dataset Management before configuring controlled experiments.",
            )
            return

        self._sync_endpoint_from_inputs()

        # Controlled Experiment Configuration & Confirmation Modal Dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Controlled Experimental Execution Configuration")
        dlg.resize(540, 480)
        d_layout = QVBoxLayout(dlg)
        d_layout.setSpacing(12)

        # 1. Image Selection Controls
        d_layout.addWidget(QLabel("<b>1. Manual Image Selection & Limits:</b>"))
        img_scope_frame = QFrame()
        img_scope_frame.setStyleSheet("background: #0f172a; border-radius: 6px; padding: 10px;")
        img_scope_layout = QVBoxLayout(img_scope_frame)

        category_row = QHBoxLayout()
        category_row.addWidget(QLabel("Category Filter:"))
        cat_combo = QComboBox()
        cat_combo.addItems(["All Categories", "Benign Only", "Malignant Only", "Normal Only"])
        category_row.addWidget(cat_combo)

        category_row.addWidget(QLabel("Max Image Batch Size:"))
        limit_spin = QSpinBox()
        limit_spin.setRange(1, 50)  # Safe bounded batch size strictly preventing 780-image runaway
        limit_spin.setValue(5)
        limit_spin.setSuffix(" images")
        category_row.addWidget(limit_spin)
        img_scope_layout.addLayout(category_row)
        d_layout.addWidget(img_scope_frame)

        # 2. Compression Method Selection
        d_layout.addWidget(QLabel("<b>2. Compression Method:</b>"))
        comp_frame = QFrame()
        comp_frame.setStyleSheet("background: #0f172a; border-radius: 6px; padding: 10px;")
        comp_layout = QVBoxLayout(comp_frame)

        method_group = QButtonGroup(dlg)
        radio_normal = QRadioButton("Option A: Normal Compression (Standard JPEG)")
        radio_adaptive = QRadioButton("Option B: ML-Based Region-Adaptive Compression")
        radio_normal.setChecked(True)
        method_group.addButton(radio_normal, 1)
        method_group.addButton(radio_adaptive, 2)

        comp_layout.addWidget(radio_normal)
        comp_layout.addWidget(radio_adaptive)
        d_layout.addWidget(comp_frame)

        # 3. Parameters / Sweeps
        d_layout.addWidget(QLabel("<b>3. Quality / Margin Parameters:</b>"))
        param_frame = QFrame()
        param_frame.setStyleSheet("background: #0f172a; border-radius: 6px; padding: 10px;")
        param_layout = QVBoxLayout(param_frame)

        # Normal sweep checkboxes
        normal_opts_widget = QWidget()
        n_opt_layout = QHBoxLayout(normal_opts_widget)
        chk_q30 = QCheckBox("Q30")
        chk_q50 = QCheckBox("Q50")
        chk_q70 = QCheckBox("Q70")
        chk_q70.setChecked(True)
        chk_q90 = QCheckBox("Q90")
        n_opt_layout.addWidget(QLabel("Qualities:"))
        n_opt_layout.addWidget(chk_q30)
        n_opt_layout.addWidget(chk_q50)
        n_opt_layout.addWidget(chk_q70)
        n_opt_layout.addWidget(chk_q90)
        param_layout.addWidget(normal_opts_widget)

        # Adaptive options
        adaptive_opts_widget = QWidget()
        a_opt_layout = QHBoxLayout(adaptive_opts_widget)
        a_opt_layout.addWidget(QLabel("Margin:"))
        adapt_margin_spin = QSpinBox()
        adapt_margin_spin.setRange(5, 40)
        adapt_margin_spin.setValue(15)
        adapt_margin_spin.setSuffix(" px")
        a_opt_layout.addWidget(adapt_margin_spin)

        a_opt_layout.addWidget(QLabel("ROI Q:"))
        adapt_roi_q_combo = QComboBox()
        adapt_roi_q_combo.addItems(["85", "90", "95"])
        adapt_roi_q_combo.setCurrentText("90")
        a_opt_layout.addWidget(adapt_roi_q_combo)

        a_opt_layout.addWidget(QLabel("BG Q:"))
        adapt_bg_q_combo = QComboBox()
        adapt_bg_q_combo.addItems(["15", "20", "25"])
        adapt_bg_q_combo.setCurrentText("20")
        a_opt_layout.addWidget(adapt_bg_q_combo)
        param_layout.addWidget(adaptive_opts_widget)
        adaptive_opts_widget.setVisible(False)

        def _update_method_ui():
            is_n = radio_normal.isChecked()
            normal_opts_widget.setVisible(is_n)
            adaptive_opts_widget.setVisible(not is_n)
            _update_estimates()

        radio_normal.toggled.connect(_update_method_ui)
        radio_adaptive.toggled.connect(_update_method_ui)
        d_layout.addWidget(param_frame)

        # 4. Network Profiles
        d_layout.addWidget(QLabel("<b>4. Network Emulation Profile:</b>"))
        net_frame = QFrame()
        net_frame.setStyleSheet("background: #0f172a; border-radius: 6px; padding: 10px;")
        net_layout = QHBoxLayout(net_frame)
        net_combo = QComboBox()
        net_combo.addItem("Unshaped Baseline (Direct link, zero delay/loss)", None)
        for p in PRESET_PROFILES:
            net_combo.addItem(f"{p.name} ({p.bandwidth}, {p.delay_ms}ms delay, {p.loss_percent}% loss)", p)
        net_layout.addWidget(net_combo)
        d_layout.addWidget(net_frame)

        # Deduplication Checkbox
        chk_skip_completed = QCheckBox("Exclude / Skip experiments already successfully completed in database")
        chk_skip_completed.setChecked(True)
        d_layout.addWidget(chk_skip_completed)

        # Dynamic Pre-Execution Estimation Box
        est_banner = QLabel("Estimated: Calculating...")
        est_banner.setStyleSheet(
            "background: #1e3a8a; color: #bfdbfe; font-weight: 700; padding: 10px; border-radius: 6px;"
        )
        d_layout.addWidget(est_banner)

        def _update_estimates():
            is_n = radio_normal.isChecked()
            meth = "NORMAL" if is_n else "REGION_ADAPTIVE"
            count = limit_spin.value()
            if is_n:
                q_count = sum([chk_q30.isChecked(), chk_q50.isChecked(), chk_q70.isChecked(), chk_q90.isChecked()])
                q_count = max(q_count, 1)
            else:
                q_count = 1

            total_trials, est_sec = ExperimentOrchestrator.estimate_batch_run(
                items_count=count,
                compression_method=meth,
                qualities_count=q_count,
                profiles=[net_combo.currentData()],
            )
            est_banner.setText(
                f"Planned Trials: {total_trials} | Estimated Total Duration: ~{est_sec:.1f} seconds (Sequential)"
            )

        limit_spin.valueChanged.connect(lambda _: _update_estimates())
        chk_q30.toggled.connect(lambda _: _update_estimates())
        chk_q50.toggled.connect(lambda _: _update_estimates())
        chk_q70.toggled.connect(lambda _: _update_estimates())
        chk_q90.toggled.connect(lambda _: _update_estimates())
        net_combo.currentIndexChanged.connect(lambda _: _update_estimates())
        _update_estimates()

        # Confirmation Buttons
        btn_box = QDialogButtonBox()
        confirm_btn = QPushButton("Confirm & Start Controlled Batch")
        confirm_btn.setStyleSheet(
            "background: #16a34a; color: white; font-weight: 700; padding: 8px 16px; border-radius: 6px;"
        )
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("SecondaryBtn")
        btn_box.addButton(confirm_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        btn_box.addButton(cancel_btn, QDialogButtonBox.ButtonRole.RejectRole)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        d_layout.addWidget(btn_box)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Prepare filtered items based on user selection
        cat_sel = cat_combo.currentText().lower()
        if "benign" in cat_sel:
            candidates = [it for it in self._all_dataset_items if it.category == "benign"]
        elif "malignant" in cat_sel:
            candidates = [it for it in self._all_dataset_items if it.category == "malignant"]
        elif "normal" in cat_sel:
            candidates = [it for it in self._all_dataset_items if it.category == "normal"]
        else:
            candidates = list(self._all_dataset_items)

        batch_items = candidates[:limit_spin.value()]

        is_normal_mode = radio_normal.isChecked()
        if is_normal_mode:
            method_str = "NORMAL"
            selected_qualities = []
            if chk_q30.isChecked(): selected_qualities.append(30)
            if chk_q50.isChecked(): selected_qualities.append(50)
            if chk_q70.isChecked(): selected_qualities.append(70)
            if chk_q90.isChecked(): selected_qualities.append(90)
            if not selected_qualities:
                selected_qualities = [70]
            margins_list = [0]
            roi_q = 90
            bg_q = 20
        else:
            method_str = "REGION_ADAPTIVE"
            selected_qualities = [int(adapt_roi_q_combo.currentText())]
            margins_list = [adapt_margin_spin.value()]
            roi_q = int(adapt_roi_q_combo.currentText())
            bg_q = int(adapt_bg_q_combo.currentText())

        prof = net_combo.currentData()

        # Update UI Controls for running batch
        self.start_batch_btn.setEnabled(False)
        self.send_single_btn.setEnabled(False)
        self.pause_batch_btn.setEnabled(True)
        self.pause_batch_btn.setText("Pause Batch")
        self.abort_batch_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Batch Running: %p%")

        self._batch_worker = BatchExperimentWorker(
            orchestrator=self.orchestrator,
            items=batch_items,
            compression_method=method_str,
            fixed_qualities=selected_qualities,
            adaptive_margins=margins_list,
            adaptive_roi_quality=roi_q,
            adaptive_bg_quality=bg_q,
            roi_mode="ML_MODEL",
            profiles=[prof],
            skip_completed=chk_skip_completed.isChecked(),
            max_batch_size=limit_spin.value(),
        )

        self._batch_worker.step_updated.connect(self._on_batch_step)
        self._batch_worker.trial_finished.connect(self._on_batch_trial_finished)
        self._batch_worker.batch_completed.connect(self._on_batch_completed)
        self._batch_worker.start()

    def _handle_toggle_pause(self):
        """Pauses or resumes the active batch worker."""
        if not self._batch_worker:
            return
        if self._batch_worker.is_paused():
            self._batch_worker.resume()
            self.pause_batch_btn.setText("Pause Batch")
            self.batch_status_lbl.setText("Execution Status: Resumed batch running...")
        else:
            self._batch_worker.pause()
            self.pause_batch_btn.setText("Resume Batch")
            self.batch_status_lbl.setText("Execution Status: PAUSED (Waiting for user resume)")

    def _handle_abort_batch(self):
        """Cooperatively aborts batch worker and restores unshaped network."""
        if self._batch_worker:
            self._batch_worker.stop()
            self.abort_batch_btn.setEnabled(False)
            self.batch_status_lbl.setText("Execution Status: Aborting batch run safely...")

    def _on_batch_step(self, curr: int, total: int, desc: str):
        if total > 0:
            pct = int((curr / total) * 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"Batch: {curr}/{total} trials ({pct}%)")
        self.batch_status_lbl.setText(f"Status: {desc}")

    def _on_batch_trial_finished(self, record: ExperimentRecord):
        self.trial_recorded.emit(record)

        # Append to live history table
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        self.history_table.setItem(row, 0, QTableWidgetItem(record.timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(record.image_id))
        self.history_table.setItem(row, 2, QTableWidgetItem(record.compression_method))
        self.history_table.setItem(row, 3, QTableWidgetItem(str(record.transmitted_bytes or record.compressed_file_size)))
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{record.transmission_time_sec or 0.0:.4f}"))
        self.history_table.setItem(row, 5, QTableWidgetItem(f"{record.measured_throughput_kbps or 0.0:,.2f}"))
        ack_text = "Verified" if record.checksum_verified else "Failed"
        self.history_table.setItem(row, 6, QTableWidgetItem(ack_text))

    def _on_batch_completed(self, completed: int, skipped: int, failed: int, summary: str):
        self.start_batch_btn.setEnabled(True)
        self.send_single_btn.setEnabled(True)
        self.pause_batch_btn.setEnabled(False)
        self.pause_batch_btn.setText("Pause Batch")
        self.abort_batch_btn.setEnabled(False)

        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Batch Complete")
        self.batch_status_lbl.setText(f"Batch Result: {summary}")

        QMessageBox.information(
            self,
            "Controlled Batch Finished",
            f"Experimental Execution Finished:\n\n"
            f"• Succeeded & Verified: {completed}\n"
            f"• Skipped (Deduplicated): {skipped}\n"
            f"• Failed: {failed}\n\n"
            f"All metrics persistently recorded in SQLite database.\nCheck Results tab for analysis.",
        )
