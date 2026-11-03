"""Receiver Application Main Window with Sidebar Navigation."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from receiver.model_manager import ModelManager
from ui.receiver.dashboard_view import ReceiverDashboardView
from ui.receiver.evaluation_view import ReceiverEvaluationView
from ui.receiver.image_view import ReceiverImageView
from ui.receiver.model_view import ReceiverModelView
from ui.receiver.receiver_view import ReceiverTCPView
from ui.receiver.results_view import ReceiverResultsView
from ui.receiver.segmentation_view import ReceiverSegmentationView
from ui.shared.styles import APP_STYLESHEET


class ReceiverMainWindow(QMainWindow):
    """Main window for the Receiver Node running on Windows 11."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network-Aware Compression — Receiver Node (Windows 11)")
        self.resize(1200, 800)
        self.setMinimumSize(980, 640)
        self.setStyleSheet(APP_STYLESHEET)
        self.model_manager = ModelManager()
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar container
        sidebar_container = QWidget()
        sidebar_container.setObjectName("SidebarList")
        sidebar_container.setFixedWidth(240)
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(12, 16, 12, 16)
        sidebar_layout.setSpacing(12)

        # Brand header
        brand_label = QLabel("RECEIVER NODE")
        brand_label.setStyleSheet("font-size: 14px; font-weight: 800; color: #38bdf8; letter-spacing: 1px;")
        sub_brand = QLabel("Network-Aware Compression")
        sub_brand.setStyleSheet("font-size: 11px; color: #64748b; font-weight: 500;")
        sidebar_layout.addWidget(brand_label)
        sidebar_layout.addWidget(sub_brand)

        # Navigation list
        self.nav_list = QListWidget()
        self.nav_list.setObjectName("SidebarList")

        nav_items = [
            "Dashboard",
            "TCP Receiver",
            "Image Viewer",
            "Model Management",
            "Segmentation",
            "Evaluation (Dice/IoU)",
            "Experiment Results",
        ]

        for item_text in nav_items:
            item = QListWidgetItem(item_text)
            self.nav_list.addItem(item)

        sidebar_layout.addWidget(self.nav_list)

        # Environment status tag at bottom of sidebar
        env_lbl = QLabel("Environment: Windows 11")
        env_lbl.setStyleSheet("font-size: 11px; color: #475569; padding: 6px;")
        sidebar_layout.addWidget(env_lbl)

        main_layout.addWidget(sidebar_container)

        # Stacked view container
        self.stack = QStackedWidget()

        # Initialize view panels with shared model manager
        self.dashboard_view = ReceiverDashboardView()
        self.tcp_view = ReceiverTCPView()
        self.image_view = ReceiverImageView()
        self.model_view = ReceiverModelView(self.model_manager)
        self.segmentation_view = ReceiverSegmentationView(self.model_manager)
        self.evaluation_view = ReceiverEvaluationView()
        self.results_view = ReceiverResultsView()

        self.stack.addWidget(self.dashboard_view)
        self.stack.addWidget(self.tcp_view)
        self.stack.addWidget(self.image_view)
        self.stack.addWidget(self.model_view)
        self.stack.addWidget(self.segmentation_view)
        self.stack.addWidget(self.evaluation_view)
        self.stack.addWidget(self.results_view)

        main_layout.addWidget(self.stack, stretch=1)

        # Connect navigation
        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

        # Wire cross-view dataflow
        self._connect_receiver_signals()

    def _connect_receiver_signals(self):
        # 1. Forward received verified images to ImageView and update Dashboard
        self.tcp_view.bridge.image_received.connect(self._on_image_received)
        self.tcp_view.bridge.status_changed.connect(self._on_server_status_changed)

        # 2. Model loaded updates dashboard card
        self.model_view.model_loaded.connect(self._on_model_loaded)

        # 3. Selected image in catalog prepares segmentation and evaluation
        self.image_view.image_selected.connect(self._on_image_selected_for_pipeline)

        # 4. Completed segmentation forwards directly to evaluation view
        self.segmentation_view.segmentation_completed.connect(self._on_segmentation_completed)

        # 5. Evaluation completion updates dashboard metrics
        self.evaluation_view.evaluation_completed.connect(self._on_evaluation_completed)

    def _on_image_received(self, metadata, image_bytes, verified, save_path, timestamp):
        if verified and save_path is not None:
            self.image_view.add_received_image(metadata, save_path)
            total_count = len(self.image_view.received_items)
            self.dashboard_view.card_received.set_value(str(total_count))

    def _on_server_status_changed(self, is_running, message):
        status_text = f"Listening ({self.tcp_view.server.port})" if is_running else "Stopped"
        self.dashboard_view.card_server.set_value(status_text)

    def _on_model_loaded(self, status_desc: str):
        self.dashboard_view.card_model.set_value(status_desc)

    def _on_image_selected_for_pipeline(self, metadata, save_path):
        self.segmentation_view.set_active_image(save_path, metadata)

    def _on_segmentation_completed(self, result, metadata):
        self.evaluation_view.load_evaluation_data(
            result,
            metadata,
            comp_image_path=self.segmentation_view.current_image_path,
        )

    def _on_evaluation_completed(self, metadata, dice, iou):
        if dice is not None and iou is not None:
            self.dashboard_view.card_eval.set_value(f"Dice: {dice:.4f} | IoU: {iou:.4f}")
        else:
            self.dashboard_view.card_eval.set_value("No GT Mask Found")

