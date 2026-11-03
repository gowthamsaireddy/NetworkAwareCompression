"""Sender Network Configuration View (Linux tc & netem)."""

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from sender.network_control import NetworkController, PRESET_PROFILES
from ui.shared.widgets import MetricBox, SectionHeader


class SenderNetworkView(QWidget):
    """Controls network emulation profiles and verifies kernel qdisc status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = NetworkController()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Network Emulation (tc / netem)",
            "Emulate constrained Ethernet channel conditions (Bandwidth, Delay, Loss) via Linux kernel qdisc",
        )
        layout.addWidget(header)

        # Active status frame
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

        # Configuration Form
        config_frame = QFrame()
        config_frame.setObjectName("CardFrame")
        form_grid = QGridLayout(config_frame)
        form_grid.setSpacing(12)

        # Network interface
        form_grid.addWidget(QLabel("Network Interface:"), 0, 0)
        self.iface_combo = QComboBox()
        interfaces = self.controller.detect_interfaces()
        self.iface_combo.addItems(interfaces)
        form_grid.addWidget(self.iface_combo, 0, 1)

        # Preset profiles
        form_grid.addWidget(QLabel("Preset Profile:"), 1, 0)
        self.preset_combo = QComboBox()
        for p in PRESET_PROFILES:
            self.preset_combo.addItem(p.name, p)
        self.preset_combo.addItem("Custom Configuration", None)
        form_grid.addWidget(self.preset_combo, 1, 1)

        # Bandwidth
        form_grid.addWidget(QLabel("Bandwidth Limit:"), 2, 0)
        self.bw_input = QComboBox()
        self.bw_input.addItems(["10mbit", "2mbit", "512kbit", "128kbit"])
        form_grid.addWidget(self.bw_input, 2, 1)

        # Delay
        form_grid.addWidget(QLabel("Network Delay (ms):"), 3, 0)
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 1000)
        self.delay_spin.setValue(0)
        form_grid.addWidget(self.delay_spin, 3, 1)

        # Packet Loss
        form_grid.addWidget(QLabel("Packet Loss Rate (%):"), 4, 0)
        self.loss_spin = QDoubleSpinBox()
        self.loss_spin.setRange(0.0, 50.0)
        self.loss_spin.setSingleStep(0.1)
        self.loss_spin.setValue(0.0)
        form_grid.addWidget(self.loss_spin, 4, 1)

        layout.addWidget(config_frame)

        # Buttons
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Traffic Shaping")
        self.reset_btn = QPushButton("Reset to Unshaped (Default)")
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
        self.apply_btn.clicked.connect(self._confirm_and_apply)
        self.reset_btn.clicked.connect(self._reset_rules)
        self.verify_btn.clicked.connect(self._refresh_status)

    def _confirm_and_apply(self):
        reply = QMessageBox.question(
            self,
            "Confirm Network Shaping",
            "Applying traffic control requires root/sudo privileges and will alter network queueing on the selected interface. Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Full application in Phase 4
            QMessageBox.information(
                self,
                "Traffic Shaping",
                "tc/netem commands will be executed in Phase 4 once the VM testbed link is connected.",
            )

    def _reset_rules(self):
        success, msg = self.controller.reset_configuration()
        self._refresh_status()
        QMessageBox.information(self, "Traffic Shaping Reset", msg)

    def _refresh_status(self):
        self.kernel_status_lbl.setText(self.controller.verify_status())

