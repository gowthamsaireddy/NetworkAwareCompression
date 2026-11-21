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
from sender.automated_experiment_runner import (
    AutomatedExperimentRunner,
    CompressionConfig,
    get_supported_compression_configs,
    load_test_dataset_items,
)
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


class AutomatedExperimentWorker(QThread):
    """Background worker executing the complete automated compression x network experiment matrix."""

    progress_updated = pyqtSignal(int, int, str, str, str, int, int, object)  # curr, total, img_id, cfg_name, prof_name, succ, fail, rec
    finished = pyqtSignal(bool, str, dict)  # is_completed, summary_msg, manifest

    def __init__(
        self,
        runner: AutomatedExperimentRunner,
        items: List[DatasetItem],
        configs: Optional[List[CompressionConfig]] = None,
        profiles: Optional[List[NetworkProfile]] = None,
    ):
        super().__init__()
        self.runner = runner
        self.items = items
        self.configs = configs
        self.profiles = profiles
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        def _cb(curr, total, img_id, cfg_name, prof_name, succ, fail, rec):
            self.progress_updated.emit(curr, total, img_id, cfg_name, prof_name, succ, fail, rec)

        def _is_stopped():
            return self._stop_requested

        completed, msg, manifest = self.runner.run_all_experiments(
            items=self.items,
            compression_configs=self.configs,
            network_profiles=self.profiles,
            progress_callback=_cb,
            stop_requested=_is_stopped,
        )
        self.finished.emit(completed, msg, manifest)


class TestImageSelectionDialog(QDialog):
    """Modal dialog allowing the user to select exactly 10 images from the held-out test split."""

    def __init__(self, current_selection: List[DatasetItem], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select 10 Held-Out Test Images")
        self.resize(720, 520)
        self.selected_items: List[DatasetItem] = list(current_selection)
        self.all_test_items: List[DatasetItem] = []
        self._load_available_test_items()
        self.init_ui()

    def _load_available_test_items(self):
        try:
            # Load all test split images from splits.json (up to 117)
            self.all_test_items = load_test_dataset_items(count=150)
        except Exception as e:
            logger.error(f"Error loading test split items: {e}")
            self.all_test_items = []

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header_lbl = QLabel("<b>Select exactly 10 images from the held-out test split:</b>")
        header_lbl.setStyleSheet("color: #f8fafc; font-size: 13px;")
        layout.addWidget(header_lbl)

        # Filter and helper buttons row
        btn_bar = QHBoxLayout()
        self.counter_lbl = QLabel(f"Selected: {len(self.selected_items)} / 10 images")
        self.counter_lbl.setStyleSheet(
            "background: #0f172a; color: #38bdf8; font-weight: 700; padding: 6px 12px; border-radius: 4px;"
        )
        btn_bar.addWidget(self.counter_lbl)
        btn_bar.addStretch()

        self.btn_default_10 = QPushButton("Use First 10 Images")
        self.btn_default_10.setObjectName("SecondaryBtn")
        self.btn_stratified_10 = QPushButton("Select 10 Stratified (5 Benign, 3 Malignant, 2 Normal)")
        self.btn_stratified_10.setObjectName("SecondaryBtn")
        self.btn_clear_all = QPushButton("Clear Selection")
        self.btn_clear_all.setObjectName("SecondaryBtn")

        btn_bar.addWidget(self.btn_default_10)
        btn_bar.addWidget(self.btn_stratified_10)
        btn_bar.addWidget(self.btn_clear_all)
        layout.addLayout(btn_bar)

        # Table of test items
        self.table = QTableWidget(len(self.all_test_items), 5)
        self.table.setHorizontalHeaderLabels(["Select", "Image Identifier", "Category", "Dimensions", "Size (Bytes)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        sel_ids = {it.image_id for it in self.selected_items}

        self.checkboxes: List[Tuple[QCheckBox, DatasetItem]] = []
        for row, item in enumerate(self.all_test_items):
            chk = QCheckBox()
            if item.image_id in sel_ids:
                chk.setChecked(True)
            chk.toggled.connect(self._on_checkbox_toggled)

            chk_cell = QWidget()
            chk_lay = QHBoxLayout(chk_cell)
            chk_lay.addWidget(chk)
            chk_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk_lay.setContentsMargins(0, 0, 0, 0)

            self.table.setCellWidget(row, 0, chk_cell)
            self.table.setItem(row, 1, QTableWidgetItem(item.image_id))
            self.table.setItem(row, 2, QTableWidgetItem(item.category.capitalize()))
            self.table.setItem(row, 3, QTableWidgetItem(f"{item.width} × {item.height}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{item.file_size_bytes:,}"))
            self.checkboxes.append((chk, item))

        layout.addWidget(self.table)

        # Dialog Buttons (OK enabled only when count == 10)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_btn = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_btn.setText("Confirm 10 Images")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._update_counter_state()

        self.btn_default_10.clicked.connect(self._select_first_10)
        self.btn_stratified_10.clicked.connect(self._select_stratified_10)
        self.btn_clear_all.clicked.connect(self._clear_all)

    def _on_checkbox_toggled(self):
        curr_selected = []
        for chk, item in self.checkboxes:
            if chk.isChecked():
                curr_selected.append(item)
        self.selected_items = curr_selected
        self._update_counter_state()

    def _update_counter_state(self):
        count = len(self.selected_items)
        if count == 10:
            self.counter_lbl.setText("Selected: Exactly 10 / 10 images (Ready)")
            self.counter_lbl.setStyleSheet(
                "background: #064e3b; color: #34d399; font-weight: 700; padding: 6px 12px; border-radius: 4px;"
            )
            self.ok_btn.setEnabled(True)
        else:
            self.counter_lbl.setText(f"Selected: {count} / 10 images (Must be exactly 10)")
            self.counter_lbl.setStyleSheet(
                "background: #450a0a; color: #f87171; font-weight: 700; padding: 6px 12px; border-radius: 4px;"
            )
            self.ok_btn.setEnabled(False)

    def _select_first_10(self):
        for i, (chk, _) in enumerate(self.checkboxes):
            chk.blockSignals(True)
            chk.setChecked(i < 10)
            chk.blockSignals(False)
        self.selected_items = [item for _, item in self.checkboxes[:10]]
        self._update_counter_state()

    def _select_stratified_10(self):
        b_count, m_count, n_count = 0, 0, 0
        new_sel = []
        for chk, item in self.checkboxes:
            chk.blockSignals(True)
            cat = item.category.lower()
            if cat == "benign" and b_count < 5:
                chk.setChecked(True)
                new_sel.append(item)
                b_count += 1
            elif cat == "malignant" and m_count < 3:
                chk.setChecked(True)
                new_sel.append(item)
                m_count += 1
            elif cat == "normal" and n_count < 2:
                chk.setChecked(True)
                new_sel.append(item)
                n_count += 1
            else:
                chk.setChecked(False)
            chk.blockSignals(False)
        self.selected_items = new_sel
        self._update_counter_state()

    def _clear_all(self):
        for chk, _ in self.checkboxes:
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
        self.selected_items = []
        self._update_counter_state()

    def get_selected_items(self) -> List[DatasetItem]:
        return list(self.selected_items)


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

    def __init__(self, parent=None, network_controller: Optional[NetworkController] = None):
        super().__init__(parent)
        self.config = get_sender_config()
        self.client = TCPTransmissionClient(
            host=self.config.receiver_ip,
            port=self.config.receiver_port,
            timeout=self.config.socket_timeout,
        )
        self.results_manager = ResultsManager()
        self.network_controller = network_controller or NetworkController()
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
        self._auto_worker: Optional[AutomatedExperimentWorker] = None

        # Automated Full-Matrix Experiment Runner and initial 10 test items
        self.auto_runner = AutomatedExperimentRunner(
            tcp_client=self.client,
            network_controller=self.network_controller,
            results_manager=self.results_manager,
        )
        self.selected_test_items: List[DatasetItem] = []
        try:
            self.selected_test_items = load_test_dataset_items(count=10)
        except Exception as e:
            logger.warning(f"Could not load initial 10 test items: {e}")

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
        # Network Conditions Frame
        # -------------------------------------------------------------
        net_cond_frame = QFrame()
        net_cond_frame.setObjectName("CardFrame")
        net_cond_layout = QVBoxLayout(net_cond_frame)
        net_cond_layout.setSpacing(10)

        net_header_row = QHBoxLayout()
        net_sec_title = QLabel("Network Conditions")
        net_sec_title.setObjectName("CardTitle")
        self.active_net_status_badge = QLabel("Active Profile: None (Unshaped Baseline)")
        self.active_net_status_badge.setStyleSheet(
            "background: #0f172a; color: #38bdf8; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
        )
        net_header_row.addWidget(net_sec_title)
        net_header_row.addStretch()
        net_header_row.addWidget(self.active_net_status_badge)
        net_cond_layout.addLayout(net_header_row)

        # Dropdown row
        prof_row = QHBoxLayout()
        prof_row.addWidget(QLabel("<b>Network Profile:</b>"))
        self.trans_profile_combo = QComboBox()
        self.trans_profile_combo.addItem("Unshaped Baseline (Direct link, zero added delay/loss)", None)
        for p in PRESET_PROFILES:
            self.trans_profile_combo.addItem(f"{p.name} ({p.bandwidth_display}, {p.delay_ms}ms delay)", p)
        self.trans_profile_combo.addItem("Custom Configuration (Manual)", None)
        prof_row.addWidget(self.trans_profile_combo, stretch=1)
        net_cond_layout.addLayout(prof_row)

        # Description label
        self.trans_profile_desc_lbl = QLabel(
            "Direct unshaped local link without artificial bandwidth throttling, delay, or packet loss."
        )
        self.trans_profile_desc_lbl.setStyleSheet("color: #94a3b8; font-style: italic; font-size: 11px;")
        self.trans_profile_desc_lbl.setWordWrap(True)
        net_cond_layout.addWidget(self.trans_profile_desc_lbl)

        # Displays Grid
        params_grid = QGridLayout()
        params_grid.setSpacing(8)

        params_grid.addWidget(QLabel("Bandwidth:"), 0, 0)
        self.lbl_bw_val = QLabel("Unlimited (Unshaped)")
        self.lbl_bw_val.setStyleSheet("font-weight: 700; color: #f8fafc;")
        params_grid.addWidget(self.lbl_bw_val, 0, 1)

        params_grid.addWidget(QLabel("One-way Delay:"), 0, 2)
        self.lbl_delay_val = QLabel("0.0 ms")
        self.lbl_delay_val.setStyleSheet("font-weight: 700; color: #f8fafc;")
        params_grid.addWidget(self.lbl_delay_val, 0, 3)

        params_grid.addWidget(QLabel("Jitter:"), 1, 0)
        self.lbl_jitter_val = QLabel("±0.0 ms")
        self.lbl_jitter_val.setStyleSheet("font-weight: 700; color: #f8fafc;")
        params_grid.addWidget(self.lbl_jitter_val, 1, 1)

        params_grid.addWidget(QLabel("Packet Loss:"), 1, 2)
        self.lbl_loss_val = QLabel("0.00 %")
        self.lbl_loss_val.setStyleSheet("font-weight: 700; color: #f8fafc;")
        params_grid.addWidget(self.lbl_loss_val, 1, 3)

        net_cond_layout.addLayout(params_grid)

        # Buttons row
        net_btn_row = QHBoxLayout()
        self.trans_apply_profile_btn = QPushButton("Apply Network Profile")
        self.trans_apply_profile_btn.setStyleSheet(
            "background: #0284c7; color: white; font-weight: 700; padding: 6px 14px; border-radius: 5px;"
        )
        self.trans_reset_profile_btn = QPushButton("Reset Network")
        self.trans_reset_profile_btn.setObjectName("SecondaryBtn")
        net_btn_row.addWidget(self.trans_apply_profile_btn)
        net_btn_row.addWidget(self.trans_reset_profile_btn)
        net_btn_row.addStretch()

        net_cond_layout.addLayout(net_btn_row)
        layout.addWidget(net_cond_frame)

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
        # One-Click Automated Image Transmission Frame
        # -------------------------------------------------------------
        auto_frame = QFrame()
        auto_frame.setObjectName("CardFrame")
        auto_layout = QVBoxLayout(auto_frame)
        auto_layout.setSpacing(10)

        auto_header_row = QHBoxLayout()
        auto_title = QLabel("One-Click Automated Image Transmission")
        auto_title.setObjectName("CardTitle")
        self.lbl_selected_img_count = QLabel(f"Selected Images: {len(self.selected_test_items)} / 10 from held-out test split")
        self.lbl_selected_img_count.setStyleSheet(
            "background: #0f172a; color: #10b981; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
        )
        auto_header_row.addWidget(auto_title)
        auto_header_row.addStretch()
        auto_header_row.addWidget(self.lbl_selected_img_count)
        auto_layout.addLayout(auto_header_row)

        auto_desc = QLabel(
            "Automated Transmission: Runs the complete cross-product matrix across 10 held-out test images × "
            "8 compression configurations (4 Uniform JPEG + 4 Region-Adaptive ROI combinations) × "
            "6 network emulation profiles = 480 total transmissions over TCP with end-to-end receiver ACK verification."
        )
        auto_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        auto_desc.setWordWrap(True)
        auto_layout.addWidget(auto_desc)

        # Buttons row
        auto_btn_row = QHBoxLayout()
        self.run_all_experiments_btn = QPushButton("Run All Compression × Network Experiments")
        self.run_all_experiments_btn.setStyleSheet(
            "background: #059669; color: white; font-weight: 800; font-size: 12px; padding: 10px 18px; border-radius: 6px;"
        )
        self.select_test_images_btn = QPushButton("Select 10 Test Images...")
        self.select_test_images_btn.setObjectName("SecondaryBtn")

        self.stop_auto_btn = QPushButton("Stop Experiments")
        self.stop_auto_btn.setObjectName("DangerBtn")
        self.stop_auto_btn.setEnabled(False)

        auto_btn_row.addWidget(self.run_all_experiments_btn)
        auto_btn_row.addWidget(self.select_test_images_btn)
        auto_btn_row.addWidget(self.stop_auto_btn)
        auto_btn_row.addStretch()
        auto_layout.addLayout(auto_btn_row)

        # Real-time Telemetry Grid
        telemetry_grid = QGridLayout()
        telemetry_grid.setSpacing(6)

        self.auto_lbl_current_img = QLabel("Current Image: Idle (Click button to begin)")
        self.auto_lbl_current_img.setStyleSheet("font-weight: 600; color: #f8fafc; font-size: 11px;")

        self.auto_lbl_current_comp = QLabel("Current Compression: Idle")
        self.auto_lbl_current_comp.setStyleSheet("font-weight: 600; color: #38bdf8; font-size: 11px;")

        self.auto_lbl_current_net = QLabel("Current Network Profile: Idle")
        self.auto_lbl_current_net.setStyleSheet("font-weight: 600; color: #fbbf24; font-size: 11px;")

        self.auto_lbl_counts = QLabel("Results: 0 Succeeded | 0 Failed")
        self.auto_lbl_counts.setStyleSheet("font-weight: 700; color: #10b981; font-size: 11px;")

        telemetry_grid.addWidget(self.auto_lbl_current_img, 0, 0)
        telemetry_grid.addWidget(self.auto_lbl_current_comp, 0, 1)
        telemetry_grid.addWidget(self.auto_lbl_current_net, 1, 0)
        telemetry_grid.addWidget(self.auto_lbl_counts, 1, 1)
        auto_layout.addLayout(telemetry_grid)

        # Progress bar
        self.auto_progress_bar = QProgressBar()
        self.auto_progress_bar.setValue(0)
        self.auto_progress_bar.setTextVisible(True)
        self.auto_progress_bar.setFormat("Idle (0 / 480 transmissions)")
        auto_layout.addWidget(self.auto_progress_bar)

        layout.addWidget(auto_frame)

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
        self.trans_profile_combo.currentIndexChanged.connect(self._on_trans_profile_changed)
        self.trans_apply_profile_btn.clicked.connect(self._handle_apply_trans_profile)
        self.trans_reset_profile_btn.clicked.connect(self._handle_reset_trans_network)
        self.run_all_experiments_btn.clicked.connect(self._handle_run_all_experiments)
        self.select_test_images_btn.clicked.connect(self._handle_select_test_images)
        self.stop_auto_btn.clicked.connect(self._handle_stop_auto_experiments)

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

    def _on_trans_profile_changed(self, index: int):
        profile: NetworkProfile | None = self.trans_profile_combo.currentData()
        if profile is not None:
            self.trans_profile_desc_lbl.setText(profile.description)
            self.lbl_bw_val.setText(profile.bandwidth_display)
            self.lbl_delay_val.setText(f"{profile.delay_ms} ms")
            self.lbl_jitter_val.setText(f"±{profile.jitter_ms} ms")
            self.lbl_loss_val.setText(f"{profile.loss_percent:.2f} %")
        elif index == 0:
            self.trans_profile_desc_lbl.setText(
                "Direct unshaped local link without artificial bandwidth throttling, delay, or packet loss."
            )
            self.lbl_bw_val.setText("Unlimited (Unshaped)")
            self.lbl_delay_val.setText("0.0 ms")
            self.lbl_jitter_val.setText("±0.0 ms")
            self.lbl_loss_val.setText("0.00 %")
        else:
            self.trans_profile_desc_lbl.setText("Custom user-specified network conditions.")

    def _handle_apply_trans_profile(self):
        profile: NetworkProfile | None = self.trans_profile_combo.currentData()
        if profile is None:
            self._handle_reset_trans_network()
            return

        success, msg = self.network_controller.apply_profile(profile)
        self._active_profile = profile
        if success:
            self.active_net_status_badge.setText(f"Active Profile: {profile.name} (Applied)")
            self.active_net_status_badge.setStyleSheet(
                "background: #064e3b; color: #6ee7b7; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
            )
            QMessageBox.information(self, "Network Profile Applied", msg)
        else:
            self.active_net_status_badge.setText(f"Selected Profile: {profile.name} (Emulation Unavailable on Windows)")
            self.active_net_status_badge.setStyleSheet(
                "background: #78350f; color: #fde68a; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
            )
            QMessageBox.warning(self, "Network Emulation Notice", msg)

    def _handle_reset_trans_network(self):
        success, msg = self.network_controller.reset_configuration()
        self._active_profile = None
        self.trans_profile_combo.setCurrentIndex(0)
        self.active_net_status_badge.setText("Active Profile: None (Unshaped Baseline)")
        self.active_net_status_badge.setStyleSheet(
            "background: #0f172a; color: #38bdf8; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
        )
        QMessageBox.information(self, "Network Reset", msg)

    def set_active_network_profile(self, profile: NetworkProfile | None):
        """Sets active traffic shaping profile for telemetry and theoretical time calculation."""
        self._active_profile = profile
        if profile is not None:
            # Sync dropdown selection
            idx = -1
            for i in range(self.trans_profile_combo.count()):
                p = self.trans_profile_combo.itemData(i)
                if p and p.name == profile.name:
                    idx = i
                    break
            if idx >= 0:
                self.trans_profile_combo.blockSignals(True)
                self.trans_profile_combo.setCurrentIndex(idx)
                self.trans_profile_combo.blockSignals(False)
            self.trans_profile_desc_lbl.setText(profile.description)
            self.lbl_bw_val.setText(profile.bandwidth_display)
            self.lbl_delay_val.setText(f"{profile.delay_ms} ms")
            self.lbl_jitter_val.setText(f"±{profile.jitter_ms} ms")
            self.lbl_loss_val.setText(f"{profile.loss_percent:.2f} %")
            self.active_net_status_badge.setText(f"Active Profile: {profile.name}")
            self.active_net_status_badge.setStyleSheet(
                "background: #064e3b; color: #6ee7b7; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
            )
        else:
            self.trans_profile_combo.blockSignals(True)
            self.trans_profile_combo.setCurrentIndex(0)
            self.trans_profile_combo.blockSignals(False)
            self.trans_profile_desc_lbl.setText(
                "Direct unshaped local link without artificial bandwidth throttling, delay, or packet loss."
            )
            self.lbl_bw_val.setText("Unlimited (Unshaped)")
            self.lbl_delay_val.setText("0.0 ms")
            self.lbl_jitter_val.setText("±0.0 ms")
            self.lbl_loss_val.setText("0.00 %")
            self.active_net_status_badge.setText("Active Profile: None (Unshaped Baseline)")
            self.active_net_status_badge.setStyleSheet(
                "background: #0f172a; color: #38bdf8; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
            )

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
                loss_val = float(self._active_profile.loss_percent) if self._active_profile else 0.0
                delay_val = float(self._active_profile.delay_ms) if self._active_profile else 0.0
                jitter_val = float(self._active_profile.jitter_ms) if self._active_profile else 0.0
                prof_name = self._active_profile.name if self._active_profile else "Unshaped Baseline"
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
                    network_jitter_ms=jitter_val,
                    network_profile=prof_name,
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
            net_combo.addItem(f"{p.name} ({p.bandwidth_display}, {p.delay_ms}ms delay, ±{p.jitter_ms}ms jitter, {p.loss_percent}% loss)", p)
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

    # -------------------------------------------------------------
    # One-Click Automated Experiment Handlers
    # -------------------------------------------------------------
    def _handle_select_test_images(self):
        """Opens modal dialog allowing user to choose exactly 10 images from the held-out test split."""
        dlg = TestImageSelectionDialog(self.selected_test_items, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.selected_test_items = dlg.get_selected_items()
            self._update_selected_images_display()

    def _update_selected_images_display(self):
        count = len(self.selected_test_items)
        self.lbl_selected_img_count.setText(f"Selected Images: {count} / 10 from held-out test split")
        if count == 10:
            self.lbl_selected_img_count.setStyleSheet(
                "background: #064e3b; color: #34d399; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
            )
        else:
            self.lbl_selected_img_count.setStyleSheet(
                "background: #450a0a; color: #f87171; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 11px;"
            )

    def _handle_run_all_experiments(self):
        """Validates prerequisites and executes the full automated experiment matrix."""
        if len(self.selected_test_items) != 10:
            QMessageBox.warning(
                self,
                "Selection Incomplete",
                f"Exactly 10 test images are required before execution.\nCurrently selected: {len(self.selected_test_items)}.\n\n"
                "Please click 'Select 10 Test Images...' to choose 10 images.",
            )
            return

        self._sync_endpoint_from_inputs()

        # Check receiver readiness
        avail, rtt, msg = self.client.check_receiver_availability(timeout=2.0)
        if not avail:
            QMessageBox.warning(
                self,
                "Receiver Not Ready",
                f"Cannot begin automated transmission:\n\n"
                f"The Receiver is unreachable at {self.client.host}:{self.client.port}.\n\n"
                f"Details: {msg}\n\n"
                f"Please ensure the Receiver application (run_receiver.bat) is running and listening on port {self.client.port}.",
            )
            return

        configs = get_supported_compression_configs()
        profiles = list(PRESET_PROFILES)
        total_trials = len(self.selected_test_items) * len(configs) * len(profiles)

        # Confirm execution with user
        reply = QMessageBox.question(
            self,
            "Confirm Automated Research Execution",
            f"Ready to execute complete automated research matrix?\n\n"
            f"• Images: {len(self.selected_test_items)} held-out test images\n"
            f"• Compression Configurations ({len(configs)} total):\n"
            f"   - 4 Uniform JPEG (Q30, Q50, Q70, Q90)\n"
            f"   - 1 ML ROI Baseline (ROI Q90 + BG Q20)\n"
            f"   - 3 ML ROI + JPEG Combinations (ROI Q90 + BG Q30, Q50, Q70)\n"
            f"• Network Emulation Profiles ({len(profiles)} total):\n"
            f"   - Hospital LAN, Broadband, 4G LTE, 3G Cellular, Rural Link, Satellite\n"
            f"• Total Transmissions: {total_trials} over direct TCP link\n\n"
            f"The receiver will receive, reconstruct, verify, and store every transmitted image under unique identifiers.\n"
            f"Transmitted compressed images and per-run logs will be saved to a unique directory under experiments/runs/.\n\n"
            f"Proceed with execution?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Prepare UI for running state
        self.run_all_experiments_btn.setEnabled(False)
        self.select_test_images_btn.setEnabled(False)
        self.stop_auto_btn.setEnabled(True)
        self.send_single_btn.setEnabled(False)
        self.start_batch_btn.setEnabled(False)

        self.auto_progress_bar.setValue(0)
        self.auto_progress_bar.setFormat(f"Starting... (0 / {total_trials})")
        self.auto_lbl_current_img.setText("Current Image: Initializing...")
        self.auto_lbl_current_comp.setText("Current Compression: Initializing...")
        self.auto_lbl_current_net.setText("Current Network Profile: Initializing...")
        self.auto_lbl_counts.setText("Results: 0 Succeeded | 0 Failed")

        # Launch background worker
        self._auto_worker = AutomatedExperimentWorker(
            runner=self.auto_runner,
            items=self.selected_test_items,
            configs=configs,
            profiles=profiles,
        )
        self._auto_worker.progress_updated.connect(self._on_auto_progress_updated)
        self._auto_worker.finished.connect(self._on_auto_finished)
        self._auto_worker.start()

    def _handle_stop_auto_experiments(self):
        """Requests cooperative stop of automated experiment sequence."""
        if self._auto_worker and self._auto_worker.isRunning():
            self.stop_auto_btn.setEnabled(False)
            self.auto_progress_bar.setFormat("Stopping safely... restoring network...")
            self._auto_worker.stop()

    def _on_auto_progress_updated(
        self,
        current: int,
        total: int,
        image_id: str,
        config_name: str,
        profile_name: str,
        success_count: int,
        failed_count: int,
        record: Optional[ExperimentRecord],
    ):
        pct = (current * 100) // max(total, 1)
        self.auto_progress_bar.setValue(pct)
        self.auto_progress_bar.setFormat(f"{current} / {total} transmissions ({pct}%)")

        self.auto_lbl_current_img.setText(f"Current Image: {image_id}")
        self.auto_lbl_current_comp.setText(f"Current Compression: {config_name}")
        self.auto_lbl_current_net.setText(f"Current Network Profile: {profile_name}")
        self.auto_lbl_counts.setText(f"Results: {success_count} Succeeded | {failed_count} Failed")

        if record:
            self.trial_recorded.emit(record)

            # Update live history table
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            self.history_table.setItem(row, 0, QTableWidgetItem(record.timestamp))
            self.history_table.setItem(row, 1, QTableWidgetItem(record.image_id))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"{record.compression_method} ({record.compression_quality})"))
            self.history_table.setItem(row, 3, QTableWidgetItem(str(record.transmitted_bytes or record.compressed_file_size)))
            self.history_table.setItem(row, 4, QTableWidgetItem(f"{record.transmission_time_sec or 0.0:.4f}"))
            self.history_table.setItem(row, 5, QTableWidgetItem(f"{record.measured_throughput_kbps or 0.0:,.2f}"))
            ack_text = "Verified" if record.checksum_verified else "Failed"
            self.history_table.setItem(row, 6, QTableWidgetItem(ack_text))

    def _on_auto_finished(self, completed: bool, summary_msg: str, manifest: dict):
        self.run_all_experiments_btn.setEnabled(True)
        self.select_test_images_btn.setEnabled(True)
        self.stop_auto_btn.setEnabled(False)
        self.send_single_btn.setEnabled(True)
        self.start_batch_btn.setEnabled(True)

        succ = manifest.get("successful_transmissions", 0)
        fail = manifest.get("failed_transmissions", 0)
        total = manifest.get("total_planned_transmissions", 0)
        run_dir = manifest.get("output_directory", "")

        status_text = "Complete" if completed else "Stopped Early"
        self.auto_progress_bar.setFormat(f"{status_text} ({succ} Succeeded, {fail} Failed)")

        QMessageBox.information(
            self,
            "Automated Transmission Run Finished",
            f"Automated Sequence {status_text}!\n\n"
            f"• Successful & Verified: {succ} / {total}\n"
            f"• Failed Transmissions: {fail}\n"
            f"• Total Duration: {manifest.get('duration_seconds', 0.0)}s\n\n"
            f"Output Artifacts Saved to:\n{run_dir}\n\n"
            f"• Transmitted Compressed Images: {manifest.get('transmitted_images_directory')}\n"
            f"• Per-Transmission CSV Log: {manifest.get('records_csv')}\n"
            f"• Global SQLite database has been updated for all trials.",
        )
