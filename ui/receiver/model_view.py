"""Receiver U-Net Model Management View."""

from pathlib import Path
from typing import Optional
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from receiver.model_manager import ModelManager
from ui.shared.widgets import MetricBox, SectionHeader


class ReceiverModelView(QWidget):
    """Manages loading and freezing pre-trained deep learning checkpoint weights."""

    model_loaded = pyqtSignal(str)  # Emits status description

    def __init__(self, model_manager: Optional[ModelManager] = None, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager or ModelManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "U-Net Model Management",
            "Load external pre-trained checkpoint weights; model is strictly frozen during inference (Zero training policy)",
        )
        layout.addWidget(header)

        # Checkpoint selection frame
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("CardFrame")
        ctrl_layout = QHBoxLayout(ctrl_frame)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select pre-trained PyTorch checkpoint (.pth or .pt)...")
        self.browse_btn = QPushButton("Browse Checkpoint")
        self.browse_btn.setObjectName("SecondaryBtn")
        self.load_btn = QPushButton("Load Checkpoint")

        ctrl_layout.addWidget(QLabel("Checkpoint Path:"))
        ctrl_layout.addWidget(self.path_input, stretch=1)
        ctrl_layout.addWidget(self.browse_btn)
        ctrl_layout.addWidget(self.load_btn)

        layout.addWidget(ctrl_frame)

        # Status & hardware cards
        status_grid = QGridLayout()
        status_grid.setSpacing(12)

        self.card_status = MetricBox("Checkpoint Status", self.model_manager.get_status())
        self.card_device = MetricBox("Compute Device", str(self.model_manager.device).upper())
        self.card_frozen = MetricBox("Weights State", "Frozen (Inference Only)")
        self.card_target_res = MetricBox("Expected Resolution", "256 × 256")

        status_grid.addWidget(self.card_status, 0, 0)
        status_grid.addWidget(self.card_device, 0, 1)
        status_grid.addWidget(self.card_frozen, 1, 0)
        status_grid.addWidget(self.card_target_res, 1, 1)

        layout.addLayout(status_grid)

        # Architecture and protocol specifications card
        spec_frame = QFrame()
        spec_frame.setObjectName("CardFrame")
        spec_layout = QVBoxLayout(spec_frame)
        spec_layout.setSpacing(8)

        spec_title = QLabel("Model Evaluation Constraints")
        spec_title.setObjectName("CardTitle")
        spec_layout.addWidget(spec_title)

        spec_text = QLabel(
            "• Architecture: U-Net (Provided externally in Phase 5)\n"
            "• Execution Mode: Frozen evaluation mode (model.eval(), torch.no_grad())\n"
            "• Ground-Truth Verification: Strict lookup by image ID — no mock or fabricated Dice/IoU scores\n"
            "• Target Dataset: Breast Ultrasound Images (BUSI)"
        )
        spec_text.setStyleSheet("color: #94a3b8; line-height: 1.4;")
        spec_layout.addWidget(spec_text)

        layout.addWidget(spec_frame)
        layout.addStretch()

        # Connect actions
        self.browse_btn.clicked.connect(self._browse_checkpoint)
        self.load_btn.clicked.connect(self._load_checkpoint)

    def _browse_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "", "PyTorch Checkpoints (*.pth *.pt)"
        )
        if file_path:
            self.path_input.setText(file_path)

    def _load_checkpoint(self):
        path_str = self.path_input.text().strip()
        if not path_str:
            QMessageBox.warning(self, "No File", "Please select a checkpoint file.")
            return

        success, msg = self.model_manager.load_checkpoint(Path(path_str))
        self.card_status.set_value(self.model_manager.get_status())
        if success:
            self.model_loaded.emit(self.model_manager.get_status())
            QMessageBox.information(self, "Model Loaded", msg)
        else:
            QMessageBox.warning(self, "Checkpoint Error", msg)

