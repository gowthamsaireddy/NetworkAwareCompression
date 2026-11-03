"""Sender Adaptive Compression View."""

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sender.adaptive_controller import AdaptiveController, AdaptiveThresholds
from ui.shared.widgets import MetricBox, SectionHeader


class SenderAdaptiveView(QWidget):
    """Provisional rule-based adaptive compression controller configuration and execution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = AdaptiveController()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Adaptive Compression Controller",
            "Configure provisional bandwidth-to-quality mapping rules (Independent from fixed-rate experiments)",
        )
        layout.addWidget(header)

        # Thresholds configuration frame
        thresh_frame = QFrame()
        thresh_frame.setObjectName("CardFrame")
        thresh_layout = QGridLayout(thresh_frame)
        thresh_layout.setSpacing(12)

        thresh_title = QLabel("Provisional Controller Thresholds")
        thresh_title.setObjectName("CardTitle")
        thresh_layout.addWidget(thresh_title, 0, 0, 1, 2)

        thresh_layout.addWidget(QLabel("Tier 1 - High Quality (Q90) Min Bandwidth (Kbps):"), 1, 0)
        self.q90_spin = QDoubleSpinBox()
        self.q90_spin.setRange(100.0, 100000.0)
        self.q90_spin.setValue(self.controller.thresholds.q90_min_kbps)
        thresh_layout.addWidget(self.q90_spin, 1, 1)

        thresh_layout.addWidget(QLabel("Tier 2 - Balanced (Q70) Min Bandwidth (Kbps):"), 2, 0)
        self.q70_spin = QDoubleSpinBox()
        self.q70_spin.setRange(100.0, 100000.0)
        self.q70_spin.setValue(self.controller.thresholds.q70_min_kbps)
        thresh_layout.addWidget(self.q70_spin, 2, 1)

        thresh_layout.addWidget(QLabel("Tier 3 - Constrained (Q50) Min Bandwidth (Kbps):"), 3, 0)
        self.q50_spin = QDoubleSpinBox()
        self.q50_spin.setRange(100.0, 100000.0)
        self.q50_spin.setValue(self.controller.thresholds.q50_min_kbps)
        thresh_layout.addWidget(self.q50_spin, 3, 1)

        thresh_layout.addWidget(QLabel("Tier 4 - Severely Constrained (< Q50 threshold):"), 4, 0)
        thresh_layout.addWidget(QLabel("Q30 (High Compression)"), 4, 1)

        layout.addWidget(thresh_frame)

        # Simulation / Evaluation frame
        sim_frame = QFrame()
        sim_frame.setObjectName("CardFrame")
        sim_layout = QVBoxLayout(sim_frame)

        sim_title = QLabel("Test Controller Decision Logic")
        sim_title.setObjectName("CardTitle")
        sim_layout.addWidget(sim_title)

        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("Test Bandwidth (Kbps):"))
        self.test_bw_input = QDoubleSpinBox()
        self.test_bw_input.setRange(10.0, 100000.0)
        self.test_bw_input.setValue(1500.0)
        self.evaluate_btn = QPushButton("Evaluate Decision")

        test_row.addWidget(self.test_bw_input)
        test_row.addWidget(self.evaluate_btn)
        test_row.addStretch()

        sim_layout.addLayout(test_row)

        self.decision_lbl = QLabel("Enter bandwidth and evaluate to test decision mapping.")
        self.decision_lbl.setStyleSheet(
            "font-family: monospace; background-color: #0b1120; padding: 10px; border-radius: 4px; color: #38bdf8;"
        )
        sim_layout.addWidget(self.decision_lbl)

        layout.addWidget(sim_frame)
        layout.addStretch()

        self.evaluate_btn.clicked.connect(self._evaluate_decision)

    def _evaluate_decision(self):
        # Update thresholds from spins
        self.controller.thresholds = AdaptiveThresholds(
            q90_min_kbps=self.q90_spin.value(),
            q70_min_kbps=self.q70_spin.value(),
            q50_min_kbps=self.q50_spin.value(),
        )
        bw = self.test_bw_input.value()
        q, explanation = self.controller.determine_quality(bw)
        self.decision_lbl.setText(f"Controller Decision: {explanation}")

