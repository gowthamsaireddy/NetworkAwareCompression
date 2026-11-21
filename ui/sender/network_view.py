"""Sender Network Configuration View (Linux tc & netem).

Implements predefined experimental network profiles and custom emulation controls.
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sender.network_control import (
    NetworkController,
    NetworkProfile,
    PRESET_PROFILES,
    parse_bandwidth_to_bps,
)
from ui.shared.widgets import SectionHeader


class SenderNetworkView(QWidget):
    """Controls network emulation profiles and verifies kernel qdisc status."""

    profile_changed = pyqtSignal(object)  # Emits active NetworkProfile or None

    def __init__(self, parent=None, controller: NetworkController | None = None):
        super().__init__(parent)
        self.controller = controller or NetworkController()
        self._updating_ui = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Network Conditions",
            "Predefined experimental network emulation profiles and custom channel shaping via Linux tc/netem",
        )
        layout.addWidget(header)

        # Representative emulation notice banner
        notice_banner = QLabel(
            "ℹ Representative experimental emulation profiles. "
            "Displayed parameters reflect configured channel conditions, not real-time physical link measurements."
        )
        notice_banner.setStyleSheet(
            "background: #1e293b; color: #94a3b8; font-size: 11px; padding: 8px 12px; "
            "border-radius: 6px; border-left: 3px solid #38bdf8;"
        )
        layout.addWidget(notice_banner)

        # Active kernel status frame
        status_frame = QFrame()
        status_frame.setObjectName("CardFrame")
        status_layout = QVBoxLayout(status_frame)

        status_title = QLabel("Verified Kernel Traffic Shaping Status")
        status_title.setObjectName("CardTitle")
        self.kernel_status_lbl = QLabel(self.controller.verify_status())
        self.kernel_status_lbl.setStyleSheet(
            "font-family: monospace; background-color: #0b1120; padding: 10px; border-radius: 4px; color: #38bdf8;"
        )
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.kernel_status_lbl)

        layout.addWidget(status_frame)

        # -------------------------------------------------------------
        # Network Conditions Form
        # -------------------------------------------------------------
        config_frame = QFrame()
        config_frame.setObjectName("CardFrame")
        form_grid = QGridLayout(config_frame)
        form_grid.setSpacing(12)

        # Network interface
        form_grid.addWidget(QLabel("<b>Network Interface:</b>"), 0, 0)
        self.iface_combo = QComboBox()
        self.iface_combo.setEditable(True)
        interfaces = self.controller.detect_interfaces()
        self.iface_combo.addItems(interfaces)
        if self.controller.interface in interfaces:
            self.iface_combo.setCurrentText(self.controller.interface)
        form_grid.addWidget(self.iface_combo, 0, 1)

        # Preset profiles dropdown
        form_grid.addWidget(QLabel("<b>Network Profile:</b>"), 1, 0)
        self.preset_combo = QComboBox()
        for p in PRESET_PROFILES:
            self.preset_combo.addItem(f"{p.name} ({p.bandwidth_display}, {p.delay_ms}ms delay)", p)
        self.preset_combo.addItem("Custom Configuration (Manual)", None)
        form_grid.addWidget(self.preset_combo, 1, 1)

        # Profile description label
        form_grid.addWidget(QLabel("<b>Profile Description:</b>"), 2, 0)
        self.desc_lbl = QLabel(PRESET_PROFILES[0].description)
        self.desc_lbl.setStyleSheet("color: #38bdf8; font-style: italic; font-size: 12px;")
        self.desc_lbl.setWordWrap(True)
        form_grid.addWidget(self.desc_lbl, 2, 1)

        # Bandwidth Limit display / edit
        form_grid.addWidget(QLabel("<b>Bandwidth Limit:</b>"), 3, 0)
        self.bw_input = QComboBox()
        self.bw_input.setEditable(True)
        self.bw_input.addItems(["100mbit", "50mbit", "10mbit", "2mbit", "512kbit", "128kbit"])
        form_grid.addWidget(self.bw_input, 3, 1)

        # One-way Delay display / edit
        form_grid.addWidget(QLabel("<b>One-way Delay:</b>"), 4, 0)
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 2000.0)
        self.delay_spin.setDecimals(1)
        self.delay_spin.setSingleStep(1.0)
        self.delay_spin.setSuffix(" ms")
        self.delay_spin.setValue(PRESET_PROFILES[0].delay_ms)
        form_grid.addWidget(self.delay_spin, 4, 1)

        # Delay Jitter display / edit
        form_grid.addWidget(QLabel("<b>Delay Jitter:</b>"), 5, 0)
        self.jitter_spin = QDoubleSpinBox()
        self.jitter_spin.setRange(0.0, 500.0)
        self.jitter_spin.setDecimals(2)
        self.jitter_spin.setSingleStep(0.5)
        self.jitter_spin.setSuffix(" ms")
        self.jitter_spin.setValue(PRESET_PROFILES[0].jitter_ms)
        form_grid.addWidget(self.jitter_spin, 5, 1)

        # Packet Loss Rate display / edit
        form_grid.addWidget(QLabel("<b>Packet Loss Rate:</b>"), 6, 0)
        self.loss_spin = QDoubleSpinBox()
        self.loss_spin.setRange(0.0, 100.0)
        self.loss_spin.setDecimals(2)
        self.loss_spin.setSingleStep(0.05)
        self.loss_spin.setSuffix(" %")
        self.loss_spin.setValue(PRESET_PROFILES[0].loss_percent)
        form_grid.addWidget(self.loss_spin, 6, 1)

        layout.addWidget(config_frame)

        # -------------------------------------------------------------
        # Action Buttons
        # -------------------------------------------------------------
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Network Profile")
        self.apply_btn.setStyleSheet(
            "background: #0284c7; color: white; font-weight: 700; padding: 9px 18px; border-radius: 6px;"
        )
        self.reset_btn = QPushButton("Reset Network")
        self.reset_btn.setObjectName("SecondaryBtn")
        self.verify_btn = QPushButton("Re-verify Kernel Status")
        self.verify_btn.setObjectName("SecondaryBtn")

        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.verify_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)
        layout.addStretch()

        # Connect actions
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.iface_combo.currentTextChanged.connect(self._on_iface_changed)
        self.bw_input.currentTextChanged.connect(self._on_param_modified)
        self.delay_spin.valueChanged.connect(self._on_param_modified)
        self.jitter_spin.valueChanged.connect(self._on_param_modified)
        self.loss_spin.valueChanged.connect(self._on_param_modified)

        self.apply_btn.clicked.connect(self._confirm_and_apply)
        self.reset_btn.clicked.connect(self._reset_rules)
        self.verify_btn.clicked.connect(self._refresh_status)

        # Initialize to first predefined profile (Hospital LAN)
        self._on_preset_changed(0)

    def _on_iface_changed(self, iface: str):
        if iface:
            self.controller.interface = iface.strip()
            self._refresh_status()

    def _on_preset_changed(self, index: int):
        profile: NetworkProfile | None = self.preset_combo.currentData()
        if profile is not None:
            self._updating_ui = True
            self.desc_lbl.setText(profile.description)
            self.bw_input.setCurrentText(profile.tc_bandwidth_str)
            self.delay_spin.setValue(profile.delay_ms)
            self.jitter_spin.setValue(profile.jitter_ms)
            self.loss_spin.setValue(profile.loss_percent)
            self._updating_ui = False
        else:
            self.desc_lbl.setText("Custom user-specified network conditions.")

    def _on_param_modified(self, *args):
        if self._updating_ui:
            return
        # If user manually changes parameters away from active preset, mark as Custom
        current_prof = self.preset_combo.currentData()
        if current_prof is not None:
            if (
                self.bw_input.currentText().strip() != current_prof.tc_bandwidth_str
                or abs(self.delay_spin.value() - current_prof.delay_ms) > 0.01
                or abs(self.jitter_spin.value() - current_prof.jitter_ms) > 0.01
                or abs(self.loss_spin.value() - current_prof.loss_percent) > 0.001
            ):
                # Switch to Custom Configuration (last item)
                self._updating_ui = True
                self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)
                self.desc_lbl.setText("Custom user-specified network conditions.")
                self._updating_ui = False

    def _confirm_and_apply(self):
        iface = self.iface_combo.currentText().strip()
        if not iface:
            QMessageBox.warning(self, "Interface Required", "Please specify or select a network interface.")
            return

        bw = self.bw_input.currentText().strip()
        delay = self.delay_spin.value()
        jitter = self.jitter_spin.value()
        loss = self.loss_spin.value()

        # Check if selected from preset
        selected_prof: NetworkProfile | None = self.preset_combo.currentData()
        if selected_prof is not None:
            prof_name = selected_prof.name
            desc = selected_prof.description
        else:
            prof_name = f"Custom ({bw}, {delay}ms, ±{jitter}ms, {loss}%)"
            desc = "Custom user-configured network profile"

        profile = NetworkProfile(
            name=prof_name,
            bandwidth=bw,
            delay_ms=delay,
            jitter_ms=jitter,
            loss_percent=loss,
            description=desc,
        )

        confirm_msg = (
            f"Apply the following network emulation profile on interface '{iface}'?\n\n"
            f"• Profile: {prof_name}\n"
            f"• Bandwidth: {profile.bandwidth_display} ({bw})\n"
            f"• One-way Delay: {delay} ms\n"
            f"• Jitter: ±{jitter} ms\n"
            f"• Packet Loss: {loss} %\n\n"
            "This will apply Linux kernel traffic control (tc netem). Proceed?"
        )

        reply = QMessageBox.question(
            self,
            "Apply Network Profile",
            confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.controller.interface = iface
            success, msg = self.controller.apply_profile(profile)
            self._refresh_status()
            if success:
                self.profile_changed.emit(profile)
                QMessageBox.information(self, "Network Profile Active", msg)
            else:
                self.profile_changed.emit(None)
                QMessageBox.warning(self, "Network Emulation Notice", msg)

    def _reset_rules(self):
        success, msg = self.controller.reset_configuration()
        self._refresh_status()
        self.profile_changed.emit(None)
        QMessageBox.information(self, "Network Emulation Reset", msg)

    def _refresh_status(self):
        status_text = self.controller.verify_status()
        self.kernel_status_lbl.setText(status_text)
