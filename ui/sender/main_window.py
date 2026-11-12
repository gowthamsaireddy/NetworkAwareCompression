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
        self.adaptive_view = SenderAdaptiveView(
            tcp_client=self.transmission_view.client,
            network_controller=self.network_view.controller,
        )
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

        # Wire cross-view dataflow
        self._connect_view_signals()

    def _connect_view_signals(self):
        # 1. Dataset selection forwards to Compression view, Adaptive view, and Dashboard
        self.dataset_view.image_selected.connect(self.compression_view.load_dataset_item)
        self.dataset_view.image_selected.connect(self.adaptive_view.set_dataset_item)
        self.dataset_view.image_selected.connect(self._on_image_selected)

        # 2. Dataset scan updates dashboard metrics & transmission batch cache
        self.dataset_view.dataset_scanned.connect(self._on_dataset_scanned)

        # 3. Compression completion forwards payload to Transmission view & Dashboard
        self.compression_view.compression_completed.connect(self.transmission_view.set_payload)
        self.compression_view.compression_completed.connect(self._on_compression_completed)

        # 4. Network profile change updates Transmission view, Adaptive view, and Dashboard
        self.network_view.profile_changed.connect(self.transmission_view.set_active_network_profile)
        self.network_view.profile_changed.connect(self.adaptive_view.set_active_network_profile)
        self.network_view.profile_changed.connect(self._on_network_profile_changed)

        # 5. Connection state sync
        self.transmission_view.connection_changed.connect(self.dashboard_view.update_connection_state)
        self.dashboard_view.test_connection_requested.connect(self.transmission_view._handle_probe)

        # 6. Recorded trials update Results view, Dashboard, and Adaptive EWMA
        self.transmission_view.trial_recorded.connect(self.results_view.load_records)
        self.transmission_view.trial_recorded.connect(self._on_trial_recorded)
        self.transmission_view.trial_recorded.connect(self._on_transmission_trial_recorded)

        self.adaptive_view.trial_recorded.connect(self.results_view.load_records)
        self.adaptive_view.trial_recorded.connect(self._on_trial_recorded)

    def _on_image_selected(self, item):
        if item is not None:
            self.dashboard_view.card_image.set_value(item.image_id)
            self.dashboard_view.card_orig_size.set_value(f"{item.file_size_bytes:,} B")

    def _on_compression_completed(self, metadata, comp_bytes, psnr, ssim):
        self.dashboard_view.card_quality.set_value(f"Q{metadata.compression_quality}")
        self.dashboard_view.card_comp_size.set_value(f"{metadata.compressed_file_size:,} B")
        ratio = (metadata.original_file_size / metadata.compressed_file_size) if metadata.compressed_file_size > 0 else 1.0
        self.dashboard_view.card_ratio.set_value(f"{ratio:.2f}:1")
        psnr_str = f"{psnr:.2f} dB" if psnr is not None else "-"
        ssim_str = f"{ssim:.4f}" if ssim is not None else "-"
        self.dashboard_view.card_psnr_ssim.set_value(f"{psnr_str} | {ssim_str}")

    def _on_dataset_scanned(self, total_count: int, stats: dict):
        self.transmission_view.set_dataset_items(self.dataset_view.dataset_manager.items)
        self.dashboard_view.card_scanned.set_value(str(total_count))

    def _on_network_profile_changed(self, profile):
        if profile is not None:
            self.dashboard_view.card_profile.set_value(f"{profile.name} ({profile.bandwidth})")
        else:
            self.dashboard_view.card_profile.set_value("Unshaped (Direct)")

    def _on_trial_recorded(self, record):
        if record.transmission_time_sec is not None:
            self.dashboard_view.card_wire_time.set_value(f"{record.transmission_time_sec:.4f} s")
        if record.measured_throughput_kbps is not None:
            self.dashboard_view.card_throughput.set_value(f"{record.measured_throughput_kbps:,.2f} Kbps")
        ack_val = "Verified (ACK_OK)" if record.checksum_verified else "Failed"
        self.dashboard_view.card_ack.set_value(ack_val)
        self.dashboard_view.progress_bar.setValue(100)
        self.dashboard_view.progress_bar.setFormat(f"Transfer Done ({record.image_id})")

    def _on_transmission_trial_recorded(self, record):
        if record.measured_throughput_kbps and record.measured_throughput_kbps > 0:
            self.adaptive_view.update_measured_throughput(record.measured_throughput_kbps)

