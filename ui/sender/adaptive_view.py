"""Sender Adaptive Compression View with Provisional Controller and Trial Execution."""

from typing import Optional

from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from common.results_manager import ExperimentRecord, ResultsManager
from sender.adaptive_controller import (
    AdaptiveController,
    AdaptiveDecision,
    AdaptiveThresholds,
    BandwidthEstimationMode,
)
from sender.dataset_manager import DatasetItem
from sender.experiment_orchestrator import ExperimentOrchestrator
from sender.network_control import NetworkController, NetworkProfile
from sender.tcp_client import TCPTransmissionClient
from ui.shared.widgets import MetricBox, SectionHeader


class AdaptiveTrialWorker(QThread):
    """Background worker for executing an adaptive compression and transmission trial."""

    progress_updated = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(bool, object, str)  # is_success, record, msg

    def __init__(
        self,
        orchestrator: ExperimentOrchestrator,
        item: DatasetItem,
        profile: Optional[NetworkProfile],
        manual_kbps: Optional[float],
        use_ewma: bool,
    ):
        super().__init__()
        self.orchestrator = orchestrator
        self.item = item
        self.profile = profile
        self.manual_kbps = manual_kbps
        self.use_ewma = use_ewma

    def run(self):
        def _cb(curr, total):
            self.progress_updated.emit(curr, total)

        success, record, msg = self.orchestrator.run_adaptive_trial(
            item=self.item,
            profile=self.profile,
            manual_kbps=self.manual_kbps,
            use_ewma=self.use_ewma,
            progress_callback=_cb,
        )
        self.finished.emit(success, record, msg)


class SenderAdaptiveView(QWidget):
    """Provisional rule-based adaptive compression controller configuration and execution."""

    trial_recorded = pyqtSignal(object)  # Emits ExperimentRecord

    def __init__(
        self,
        tcp_client: Optional[TCPTransmissionClient] = None,
        network_controller: Optional[NetworkController] = None,
        results_manager: Optional[ResultsManager] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.controller = AdaptiveController()
        self.tcp_client = tcp_client or TCPTransmissionClient()
        self.network_controller = network_controller or NetworkController()
        self.results_manager = results_manager or ResultsManager()
        self.orchestrator = ExperimentOrchestrator(
            tcp_client=self.tcp_client,
            network_controller=self.network_controller,
            adaptive_controller=self.controller,
            results_manager=self.results_manager,
        )

        self._selected_item: Optional[DatasetItem] = None
        self._active_profile: Optional[NetworkProfile] = None
        self._worker: Optional[AdaptiveTrialWorker] = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Adaptive Compression Controller",
            "Configurable provisional bandwidth-to-quality mapping rules (Separated from fixed-rate trials)",
        )
        layout.addWidget(header)

        # Upper grid: Thresholds (Left) & Channel Input Mode (Right)
        top_grid = QGridLayout()
        top_grid.setSpacing(16)

        # 1. Left Card: Controller Thresholds Configuration
        thresh_frame = QFrame()
        thresh_frame.setObjectName("CardFrame")
        thresh_layout = QGridLayout(thresh_frame)
        thresh_layout.setSpacing(10)

        thresh_title = QLabel("Provisional Controller Thresholds")
        thresh_title.setObjectName("CardTitle")
        thresh_layout.addWidget(thresh_title, 0, 0, 1, 2)

        thresh_layout.addWidget(QLabel("Tier 1 - High Quality (Q90) Min (Kbps):"), 1, 0)
        self.q90_spin = QDoubleSpinBox()
        self.q90_spin.setRange(100.0, 100000.0)
        self.q90_spin.setValue(self.controller.thresholds.q90_min_kbps)
        self.q90_spin.valueChanged.connect(self._on_thresholds_changed)
        thresh_layout.addWidget(self.q90_spin, 1, 1)

        thresh_layout.addWidget(QLabel("Tier 2 - Balanced (Q70) Min (Kbps):"), 2, 0)
        self.q70_spin = QDoubleSpinBox()
        self.q70_spin.setRange(100.0, 100000.0)
        self.q70_spin.setValue(self.controller.thresholds.q70_min_kbps)
        self.q70_spin.valueChanged.connect(self._on_thresholds_changed)
        thresh_layout.addWidget(self.q70_spin, 2, 1)

        thresh_layout.addWidget(QLabel("Tier 3 - Constrained (Q50) Min (Kbps):"), 3, 0)
        self.q50_spin = QDoubleSpinBox()
        self.q50_spin.setRange(100.0, 100000.0)
        self.q50_spin.setValue(self.controller.thresholds.q50_min_kbps)
        self.q50_spin.valueChanged.connect(self._on_thresholds_changed)
        thresh_layout.addWidget(self.q50_spin, 3, 1)

        thresh_layout.addWidget(QLabel("Tier 4 - Severely Constrained:"), 4, 0)
        q30_lbl = QLabel("< Tier 3 threshold -> Q30")
        q30_lbl.setStyleSheet("color: #94a3b8; font-style: italic;")
        thresh_layout.addWidget(q30_lbl, 4, 1)

        thresh_layout.addWidget(QLabel("EWMA Smoothing Factor (α):"), 5, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.05, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.controller.ewma_alpha)
        self.alpha_spin.valueChanged.connect(self._on_alpha_changed)
        thresh_layout.addWidget(self.alpha_spin, 5, 1)

        top_grid.addWidget(thresh_frame, 0, 0)

        # 2. Right Card: Channel Estimation & Mode Selection
        mode_frame = QFrame()
        mode_frame.setObjectName("CardFrame")
        mode_layout = QVBoxLayout(mode_frame)
        mode_layout.setSpacing(10)

        mode_title = QLabel("Channel State & Bandwidth Input Source")
        mode_title.setObjectName("CardTitle")
        mode_layout.addWidget(mode_title)

        self.mode_group = QButtonGroup(self)

        # Radio 1: Manual Input
        manual_row = QHBoxLayout()
        self.radio_manual = QRadioButton("Manual Channel Bandwidth:")
        self.manual_spin = QDoubleSpinBox()
        self.manual_spin.setRange(10.0, 100000.0)
        self.manual_spin.setValue(1500.0)
        self.manual_spin.setSuffix(" Kbps")
        self.manual_spin.valueChanged.connect(self._evaluate_decision)
        manual_row.addWidget(self.radio_manual)
        manual_row.addWidget(self.manual_spin)
        manual_row.addStretch()
        mode_layout.addLayout(manual_row)
        self.mode_group.addButton(self.radio_manual, 1)

        # Radio 2: Active Network Profile
        prof_row = QHBoxLayout()
        self.radio_profile = QRadioButton("Active Linux tc Profile:")
        self.profile_status_lbl = QLabel("None (Unshaped)")
        self.profile_status_lbl.setStyleSheet("color: #38bdf8; font-weight: 600;")
        prof_row.addWidget(self.radio_profile)
        prof_row.addWidget(self.profile_status_lbl)
        prof_row.addStretch()
        mode_layout.addLayout(prof_row)
        self.mode_group.addButton(self.radio_profile, 2)

        # Radio 3: Measured EWMA Throughput
        ewma_row = QHBoxLayout()
        self.radio_ewma = QRadioButton("Measured EWMA Throughput:")
        self.ewma_status_lbl = QLabel("No throughput measured yet")
        self.ewma_status_lbl.setStyleSheet("color: #38bdf8; font-weight: 600;")
        ewma_row.addWidget(self.radio_ewma)
        ewma_row.addWidget(self.ewma_status_lbl)
        ewma_row.addStretch()
        mode_layout.addLayout(ewma_row)
        self.mode_group.addButton(self.radio_ewma, 3)

        self.radio_manual.setChecked(True)
        self.mode_group.buttonToggled.connect(self._on_mode_toggled)

        # Selected Image Indicator
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Target Ultrasound Image:"))
        self.selected_img_lbl = QLabel("No image selected (select in Dataset Management)")
        self.selected_img_lbl.setStyleSheet("color: #f59e0b; font-weight: bold;")
        img_row.addWidget(self.selected_img_lbl)
        img_row.addStretch()
        mode_layout.addLayout(img_row)

        top_grid.addWidget(mode_frame, 0, 1)
        layout.addLayout(top_grid)

        # 3. Controller Decision Banner
        decision_frame = QFrame()
        decision_frame.setObjectName("CardFrame")
        decision_layout = QVBoxLayout(decision_frame)
        decision_layout.setSpacing(8)

        decision_header_row = QHBoxLayout()
        d_title = QLabel("Provisional Quality Recommendation")
        d_title.setObjectName("CardTitle")
        decision_header_row.addWidget(d_title)
        decision_header_row.addStretch()

        self.evaluate_btn = QPushButton("Evaluate Decision")
        self.evaluate_btn.setObjectName("SecondaryBtn")
        self.evaluate_btn.clicked.connect(self._evaluate_decision)
        decision_header_row.addWidget(self.evaluate_btn)
        decision_layout.addLayout(decision_header_row)

        self.decision_badge = QLabel("Decision: Evaluating...")
        self.decision_badge.setStyleSheet(
            "font-size: 13px; font-weight: 600; background-color: #0b1120; padding: 10px 14px; "
            "border-radius: 6px; color: #38bdf8; border: 1px solid #1e293b;"
        )
        decision_layout.addWidget(self.decision_badge)

        # Action row
        action_row = QHBoxLayout()
        self.run_trial_btn = QPushButton("Execute Adaptive Transmission Trial")
        self.run_trial_btn.clicked.connect(self._handle_run_trial)
        action_row.addWidget(self.run_trial_btn)
        action_row.addStretch()
        decision_layout.addLayout(action_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle (Ready)")
        decision_layout.addWidget(self.progress_bar)

        layout.addWidget(decision_frame)

        # 4. Adaptive Trial History Table
        layout.addWidget(QLabel("Adaptive Decision & Trial History:"))
        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels([
            "Timestamp",
            "Image ID",
            "Selected Quality",
            "Estimation Mode",
            "Bandwidth (Kbps)",
            "Tx Time (s)",
            "Throughput (Kbps)",
            "Status / ACK",
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.history_table, stretch=1)

        # Initial evaluation
        self._evaluate_decision()

    def set_dataset_item(self, item: Optional[DatasetItem]):
        """Sets the active dataset item from dataset view."""
        self._selected_item = item
        if item is not None:
            self.selected_img_lbl.setText(f"{item.image_id} [{item.category.capitalize()}]")
            self.selected_img_lbl.setStyleSheet("color: #10b981; font-weight: bold;")
        else:
            self.selected_img_lbl.setText("No image selected (select in Dataset Management)")
            self.selected_img_lbl.setStyleSheet("color: #f59e0b; font-weight: bold;")

    def set_active_network_profile(self, profile: Optional[NetworkProfile]):
        """Sets the active network shaping profile."""
        self._active_profile = profile
        if profile is not None:
            self.profile_status_lbl.setText(f"{profile.name} ({profile.bandwidth}, {profile.delay_ms}ms, {profile.loss_percent}%)")
        else:
            self.profile_status_lbl.setText("None (Unshaped)")
        if self.radio_profile.isChecked():
            self._evaluate_decision()

    def update_measured_throughput(self, kbps: float):
        """Updates measured EWMA throughput and refreshes UI label."""
        smoothed = self.controller.update_measured_throughput(kbps)
        self.ewma_status_lbl.setText(f"{smoothed:,.2f} Kbps (Smoothed)")
        if self.radio_ewma.isChecked():
            self._evaluate_decision()

    def _on_thresholds_changed(self):
        self.controller.thresholds = AdaptiveThresholds(
            q90_min_kbps=self.q90_spin.value(),
            q70_min_kbps=self.q70_spin.value(),
            q50_min_kbps=self.q50_spin.value(),
        )
        self._evaluate_decision()

    def _on_alpha_changed(self):
        self.controller.ewma_alpha = self.alpha_spin.value()

    def _on_mode_toggled(self):
        self._evaluate_decision()

    def _evaluate_decision(self) -> AdaptiveDecision:
        manual_kbps = self.manual_spin.value() if self.radio_manual.isChecked() else None
        profile_bps = self._active_profile.bandwidth_bps if (self.radio_profile.isChecked() and self._active_profile) else None
        use_ewma = self.radio_ewma.isChecked()

        decision = self.controller.evaluate_with_fallback(
            manual_kbps=manual_kbps,
            profile_bw_bps=profile_bps,
            use_ewma=use_ewma,
        )

        quality_color = {
            90: "#10b981",  # Emerald / High Quality
            70: "#38bdf8",  # Sky Blue / Balanced
            50: "#f59e0b",  # Amber / Constrained
            30: "#ef4444",  # Red / Severely Constrained
        }.get(decision.quality, "#38bdf8")

        self.decision_badge.setText(
            f"Recommended Quality: Q{decision.quality}  |  Source: {decision.mode}  |  {decision.explanation}"
        )
        self.decision_badge.setStyleSheet(
            f"font-size: 13px; font-weight: 600; background-color: #0b1120; padding: 10px 14px; "
            f"border-radius: 6px; color: {quality_color}; border: 1px solid {quality_color}40;"
        )
        return decision

    def _handle_run_trial(self):
        if self._selected_item is None:
            QMessageBox.warning(
                self,
                "No Image Selected",
                "Please select an ultrasound image from Dataset Management before executing an adaptive trial.",
            )
            return

        self.run_trial_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Transmitting Adaptive Frame: %p%")

        manual_kbps = self.manual_spin.value() if self.radio_manual.isChecked() else None
        use_ewma = self.radio_ewma.isChecked()

        self._worker = AdaptiveTrialWorker(
            orchestrator=self.orchestrator,
            item=self._selected_item,
            profile=self._active_profile,
            manual_kbps=manual_kbps,
            use_ewma=use_ewma,
        )
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished.connect(self._on_trial_finished)
        self._worker.start()

    def _on_progress(self, curr: int, total: int):
        if total > 0:
            pct = int((curr / total) * 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"Transmitting: {curr}/{total} bytes ({pct}%)")

    def _on_trial_finished(self, success: bool, record: Optional[ExperimentRecord], msg: str):
        self.run_trial_btn.setEnabled(True)

        if success and record is not None:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Adaptive Trial Completed & Logged (ACK_OK)")

            # Update EWMA label
            if record.measured_throughput_kbps:
                self.update_measured_throughput(record.measured_throughput_kbps)

            # Insert into history table
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            self.history_table.setItem(row, 0, QTableWidgetItem(record.timestamp))
            self.history_table.setItem(row, 1, QTableWidgetItem(record.image_id))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"Q{record.compression_quality}"))
            self.history_table.setItem(row, 3, QTableWidgetItem(record.notes.replace("Mode: ", "")))
            bw_val = f"{record.bandwidth_limit}" if record.bandwidth_limit != "unshaped" else "Unshaped"
            self.history_table.setItem(row, 4, QTableWidgetItem(bw_val))
            self.history_table.setItem(row, 5, QTableWidgetItem(f"{record.transmission_time_sec:.4f}"))
            self.history_table.setItem(row, 6, QTableWidgetItem(f"{record.measured_throughput_kbps:.2f}"))
            self.history_table.setItem(row, 7, QTableWidgetItem("Verified (ACK_OK)"))

            # Emit signal to update results view and dashboard
            self.trial_recorded.emit(record)
        else:
            self.progress_bar.setFormat("Adaptive Trial Failed")
            QMessageBox.warning(self, "Adaptive Trial Failed", f"Transmission unverified: {msg}")
