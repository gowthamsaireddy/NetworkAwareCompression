"""Receiver Dashboard View."""

from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from shared.constants import (
    STATUS_DISCONNECTED,
    STATUS_NO_DATA,
    STATUS_NOT_CALCULATED,
)
from ui.shared.widgets import MetricBox, SectionHeader


class ReceiverDashboardView(QWidget):
    """High-level operational overview for the Receiver node."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Receiver Node Dashboard",
            "Operating node: Windows 11 | Role: TCP server, SHA-256 verification, frozen U-Net inference, Dice / IoU evaluation",
        )
        layout.addWidget(header)

        # Metrics grid
        grid = QGridLayout()
        grid.setSpacing(12)

        self.card_server = MetricBox("TCP Server Status", "Stopped")
        self.card_received = MetricBox("Received Images", "0")
        self.card_model = MetricBox("Active Model Checkpoint", "No model loaded")
        self.card_eval = MetricBox("Evaluation Metrics", STATUS_NOT_CALCULATED)

        grid.addWidget(self.card_server, 0, 0)
        grid.addWidget(self.card_received, 0, 1)
        grid.addWidget(self.card_model, 1, 0)
        grid.addWidget(self.card_eval, 1, 1)

        layout.addLayout(grid)

        # Operational Policy & Environment Card
        info_frame = QFrame()
        info_frame.setObjectName("CardFrame")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(8)

        info_title = QLabel("Receiver Pipeline Configuration")
        info_title.setObjectName("CardTitle")
        info_layout.addWidget(info_title)

        self.info_details = QLabel(
            "• Operating System: Windows 11 (Physical Ethernet Interface)\n"
            "• Inference Engine: PyTorch (Frozen Evaluation Mode, torch.no_grad)\n"
            "• Input Preprocessing: Grayscale -> 256x256 -> Normalization -> Binary Lesion Mask\n"
            "• Ground-Truth Verification: Matched by Image ID; genuine Dice/IoU calculation only"
        )
        self.info_details.setStyleSheet("color: #94a3b8; line-height: 1.4;")
        info_layout.addWidget(self.info_details)

        layout.addWidget(info_frame)
        layout.addStretch()

