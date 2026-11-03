"""Sender Application Main Window with Sidebar Navigation."""

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
from ui.sender.adaptive_view import SenderAdaptiveView
from ui.sender.compression_view import SenderCompressionView
from ui.sender.dashboard_view import SenderDashboardView
from ui.sender.dataset_view import SenderDatasetView
from ui.sender.network_view import SenderNetworkView
from ui.sender.results_view import SenderResultsView
from ui.sender.transmission_view import SenderTransmissionView
from ui.shared.styles import APP_STYLESHEET


class SenderMainWindow(QMainWindow):
    """Main window for the Sender Node running on Ubuntu Linux."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network-Aware Compression — Sender Node (Ubuntu)")
        self.resize(1200, 800)
        self.setMinimumSize(980, 640)
        self.setStyleSheet(APP_STYLESHEET)
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
        brand_label = QLabel("SENDER NODE")
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
            "Dataset Management",
            "Compression & Quality",
            "Network Emulation",
            "TCP Transmission",
            "Adaptive Compression",
            "Experiment Results",
        ]

        for item_text in nav_items:
            item = QListWidgetItem(item_text)
            self.nav_list.addItem(item)

        sidebar_layout.addWidget(self.nav_list)

        # Environment status tag at bottom of sidebar
        env_lbl = QLabel("Environment: Ubuntu (Bridged)")
        env_lbl.setStyleSheet("font-size: 11px; color: #475569; padding: 6px;")
        sidebar_layout.addWidget(env_lbl)

        main_layout.addWidget(sidebar_container)

        # Stacked view container
        self.stack = QStackedWidget()

        # Initialize view panels
        self.dashboard_view = SenderDashboardView()
        self.dataset_view = SenderDatasetView()
        self.compression_view = SenderCompressionView()
        self.network_view = SenderNetworkView()
        self.transmission_view = SenderTransmissionView()
        self.adaptive_view = SenderAdaptiveView()
        self.results_view = SenderResultsView()

        self.stack.addWidget(self.dashboard_view)
        self.stack.addWidget(self.dataset_view)
        self.stack.addWidget(self.compression_view)
        self.stack.addWidget(self.network_view)
        self.stack.addWidget(self.transmission_view)
        self.stack.addWidget(self.adaptive_view)
        self.stack.addWidget(self.results_view)

        main_layout.addWidget(self.stack, stretch=1)

        # Connect navigation
        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(0)

