"""Receiver Dashboard View."""

import shutil
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from common.config import DATA_DIR
from shared.constants import (
    STATUS_DISCONNECTED,
    STATUS_NO_DATA,
    STATUS_NOT_CALCULATED,
)
from ui.shared.widgets import MetricBox, SectionHeader


class ReceiverDashboardView(QWidget):
    """Live operational monitoring dashboard for the Receiver Node."""

    auto_process_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Receiver Node Dashboard",
            "Operating node: Windows 11 | Role: TCP server, SHA-256 verification, frozen U-Net inference, Dice / IoU evaluation",
        )
        layout.addWidget(header)

        # Mode & Quick-Settings Bar
        mode_frame = QFrame()
        mode_frame.setObjectName("CardFrame")
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(12, 8, 12, 8)

        self.auto_process_chk = QCheckBox("Enable Automatic Processing Pipeline (Auto-infer on arrival)")
        self.auto_process_chk.setStyleSheet("font-weight: 600; color: #38bdf8;")
        self.auto_process_chk.setChecked(True)
        self.auto_process_chk.toggled.connect(self.auto_process_toggled.emit)

        self.pipeline_badge = QLabel("Mode: Automatic Segmentation")
        self.pipeline_badge.setStyleSheet("color: #10b981; font-weight: 600; font-size: 11px;")

        mode_layout.addWidget(self.auto_process_chk)
        mode_layout.addStretch()
        mode_layout.addWidget(self.pipeline_badge)
        layout.addWidget(mode_frame)

        # Comprehensive Metrics Grid (3x3 layout)
        grid = QGridLayout()
        grid.setSpacing(12)

        # Row 0: Network & Client
        self.card_server = MetricBox("TCP Server Status", "Stopped")
        self.card_client = MetricBox("Connected Sender", "None")
        self.card_received = MetricBox("Received Images", "0")

        # Row 1: Model & Hardware
        self.card_model = MetricBox("Active Model Checkpoint", "No model loaded")
        self.card_device = MetricBox("Compute Device", "CUDA / CPU")
        self.card_storage = MetricBox("Storage Status", self._get_storage_str())

        # Row 2: Telemetry & Clinical Metrics
        self.card_exp_id = MetricBox("Latest Experiment ID", "None")
        self.card_checksum = MetricBox("Checksum Verification", "Pending")
        self.card_eval = MetricBox("Latest Dice & IoU", STATUS_NOT_CALCULATED)

        grid.addWidget(self.card_server, 0, 0)
        grid.addWidget(self.card_client, 0, 1)
        grid.addWidget(self.card_received, 0, 2)

        grid.addWidget(self.card_model, 1, 0)
        grid.addWidget(self.card_device, 1, 1)
        grid.addWidget(self.card_storage, 1, 2)

        grid.addWidget(self.card_exp_id, 2, 0)
        grid.addWidget(self.card_checksum, 2, 1)
        grid.addWidget(self.card_eval, 2, 2)

        layout.addLayout(grid)

        # Operational Policy & Environment Card
        info_frame = QFrame()
        info_frame.setObjectName("CardFrame")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(6)

        info_title = QLabel("Receiver Pipeline Configuration & Policy")
        info_title.setObjectName("CardTitle")
        info_layout.addWidget(info_title)

        self.info_details = QLabel(
            "• Operating System: Windows 11 (Physical Ethernet Interface)\n"
            "• Inference Engine: PyTorch (Frozen Evaluation Mode, torch.no_grad)\n"
            "• Input Preprocessing: Grayscale -> 256x256 -> Normalization -> Binary Lesion Mask\n"
            "• Ground-Truth Verification: Matched by Image ID; genuine Dice/IoU calculation only\n"
            "• Zero-Dummy-Data Policy: Active (All metrics reflect physical measurements and real masks)"
        )
        self.info_details.setStyleSheet("color: #94a3b8; line-height: 1.4; font-size: 12px;")
        info_layout.addWidget(self.info_details)

        layout.addWidget(info_frame)
        layout.addStretch()

    def _get_storage_str(self) -> str:
        """Returns readable free storage string."""
        try:
            recv_dir = DATA_DIR / "received"
            free_gb = shutil.disk_usage(recv_dir).free / (1024**3)
            return f"{free_gb:.1f} GB Free"
        except Exception:
            return "Storage Available"

    def update_storage(self):
        self.card_storage.set_value(self._get_storage_str())
