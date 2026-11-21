"""Receiver Application Main Window with Sidebar Navigation."""

from datetime import datetime
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from common.config import get_receiver_config
from common.results_manager import ExperimentRecord, ResultsManager
from receiver.batch_segmentation_worker import BatchSegmentationWorker
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

        self.config = get_receiver_config()
        self.auto_process_enabled = self.config.auto_process_enabled
        self.model_manager = ModelManager(device=self.config.device if self.config.device != "auto" else None)
        self.results_manager = ResultsManager()
        self._rx_telemetry = {}
        self._batch_worker: Optional[BatchSegmentationWorker] = None

        self.init_ui()
        self._auto_load_default_checkpoint()

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
        self.dashboard_view.auto_process_chk.setChecked(self.auto_process_enabled)
        self.dashboard_view.card_device.set_value(str(self.model_manager.device).upper())

        self.tcp_view = ReceiverTCPView()
        self.tcp_view.bind_ip_input.setText(self.config.listen_ip)
        self.tcp_view.bind_port_input.setValue(self.config.listen_port)

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
        self.nav_list.currentRowChanged.connect(self._on_navigation_changed)
        self.nav_list.setCurrentRow(0)

        # Wire cross-view dataflow
        self._connect_receiver_signals()

        # Index existing received images from disk storage
        existing_count = self.image_view.load_existing_received_images()
        if existing_count > 0:
            self.dashboard_view.card_received.set_value(str(existing_count))
        self.dashboard_view.update_storage()

    def _auto_load_default_checkpoint(self):
        """Attempts to auto-load default model checkpoint on launch if present."""
        chk_path = Path(self.config.model_checkpoint_path)
        if chk_path.exists():
            success, msg = self.model_manager.load_checkpoint(chk_path)
            if success:
                self.model_view.path_input.setText(str(chk_path))
                self.model_view.card_status.set_value(self.model_manager.get_status())
                self.dashboard_view.card_model.set_value(chk_path.name)
                self.dashboard_view.card_device.set_value(str(self.model_manager.device).upper())

    def _connect_receiver_signals(self):
        # 1. Forward received verified images to ImageView and update Dashboard
        self.tcp_view.bridge.image_received.connect(self._on_image_received)
        self.tcp_view.bridge.status_changed.connect(self._on_server_status_changed)
        self.tcp_view.bridge.client_connected.connect(self._on_client_connected)
        self.tcp_view.bridge.client_disconnected.connect(self._on_client_disconnected)

        # 2. Model loaded updates dashboard card
        self.model_view.model_loaded.connect(self._on_model_loaded)

        # 3. Selected image in catalog prepares segmentation and evaluation
        self.image_view.image_selected.connect(self._on_image_selected_for_pipeline)

        # 4. Completed segmentation forwards directly to evaluation view
        self.segmentation_view.segmentation_completed.connect(self._on_segmentation_completed)

        # 5. Batch segmentation requested & cancelled
        self.segmentation_view.run_batch_btn.clicked.connect(self._on_batch_segmentation_requested)
        self.segmentation_view.cancel_batch_btn.clicked.connect(self._on_cancel_batch_segmentation)

        # 6. Evaluation completion updates dashboard metrics and results table
        self.evaluation_view.evaluation_completed.connect(self._on_evaluation_completed)

        # 7. Auto process toggle
        self.dashboard_view.auto_process_toggled.connect(self._on_auto_process_toggled)

    def _on_navigation_changed(self, index: int):
        self.stack.setCurrentIndex(index)
        if index == 6:  # Experiment Results tab
            self.results_view.load_records()
        elif index == 0:  # Dashboard tab
            self.dashboard_view.card_received.set_value(str(len(self.image_view.received_items)))
            self.dashboard_view.update_storage()

    def _on_auto_process_toggled(self, enabled: bool):
        self.auto_process_enabled = enabled
        mode_text = "Mode: Automatic Segmentation" if enabled else "Mode: Manual Segmentation Only"
        self.dashboard_view.pipeline_badge.setText(mode_text)
        color = "#10b981" if enabled else "#f59e0b"
        self.dashboard_view.pipeline_badge.setStyleSheet(f"color: {color}; font-weight: 600; font-size: 11px;")

    def _on_client_connected(self, client_addr: str):
        self.dashboard_view.card_client.set_value(client_addr)

    def _on_client_disconnected(self, client_addr: str):
        self.dashboard_view.card_client.set_value("Disconnected")

    def _on_image_received(
        self,
        metadata,
        image_bytes,
        verified,
        save_path,
        timestamp,
        rx_time_sec: float = 0.0,
        rx_throughput_kbps: float = 0.0,
    ):
        # Update dashboard telemetry
        self.dashboard_view.card_exp_id.set_value(metadata.experiment_id)
        chk_status = "Verified (SHA-256 Match)" if verified else "Failed (Checksum Mismatch)"
        self.dashboard_view.card_checksum.set_value(chk_status)

        # Cache measured transmission telemetry in memory
        if not hasattr(self, "_rx_telemetry"):
            self._rx_telemetry = {}
        if metadata.experiment_id:
            self._rx_telemetry[metadata.experiment_id] = (rx_time_sec, rx_throughput_kbps)

        if verified and save_path is not None:
            self.image_view.add_received_image(metadata, save_path)
            total_count = len(self.image_view.received_items)
            self.dashboard_view.card_received.set_value(str(total_count))
            self.dashboard_view.update_storage()

            # Record reception in ResultsManager with measured wire telemetry
            ratio = (
                (metadata.original_file_size / metadata.compressed_file_size)
                if metadata.compressed_file_size > 0
                else 1.0
            )
            tx_time_val = rx_time_sec if rx_time_sec > 0.0 else None
            thru_val = rx_throughput_kbps if rx_throughput_kbps > 0.0 else None

            rec = ExperimentRecord(
                experiment_id=metadata.experiment_id,
                timestamp=timestamp,
                image_id=metadata.image_id,
                compression_quality=metadata.compression_quality,
                original_file_size=metadata.original_file_size,
                compressed_file_size=metadata.compressed_file_size,
                compression_ratio=ratio,
                psnr=None,
                ssim=None,
                bandwidth_limit="unshaped",
                packet_loss_rate=0.0,
                network_delay_ms=0.0,
                theoretical_transmission_time_sec=None,
                transmission_time_sec=tx_time_val,
                measured_throughput_kbps=thru_val,
                checksum_verified=verified,
                dice_coefficient=None,
                iou=None,
                notes="Received via TCP",
            )
            self.results_manager.record_or_update_experiment(rec)
            self.results_view.load_records()

            # Automatic Processing Pipeline (if enabled and model is mounted)
            if self.auto_process_enabled and self.model_manager.is_loaded and self.model_manager.model is not None:
                success, result, msg = self.segmentation_view.engine.run_inference(save_path, metadata.image_id)
                if success and result is not None:
                    self.segmentation_view.current_result = result
                    self.evaluation_view.load_evaluation_data(
                        result,
                        metadata,
                        comp_image_path=save_path,
                    )

    def _on_server_status_changed(self, is_running, message):
        status_text = f"Listening ({self.tcp_view.server.port})" if is_running else "Stopped"
        self.dashboard_view.card_server.set_value(status_text)

    def _on_model_loaded(self, status_desc: str):
        self.dashboard_view.card_model.set_value(status_desc)
        self.dashboard_view.card_device.set_value(str(self.model_manager.device).upper())
        self.model_view.card_device.set_value(str(self.model_manager.device).upper())

    def _on_image_selected_for_pipeline(self, metadata, save_path):
        self.segmentation_view.set_active_image(save_path, metadata)

    def _on_segmentation_completed(self, result, metadata):
        self.evaluation_view.load_evaluation_data(
            result,
            metadata,
            comp_image_path=self.segmentation_view.current_image_path,
        )

    def _on_batch_segmentation_requested(self):
        if self._batch_worker and self._batch_worker.isRunning():
            QMessageBox.warning(self, "Batch in Progress", "Batch segmentation is already currently running.")
            return

        if not self.image_view.received_items:
            QMessageBox.information(self, "No Images", "No received images available to segment.")
            return

        if not self.model_manager.is_loaded or self.model_manager.model is None:
            QMessageBox.warning(
                self,
                "No Model Loaded",
                "Please load a pre-trained frozen U-Net model from the Model Management tab first.",
            )
            return

        total_items = len(self.image_view.received_items)
        self.segmentation_view.set_batch_running(True, total=total_items)

        self._batch_worker = BatchSegmentationWorker(
            engine=self.segmentation_view.engine,
            items=self.image_view.received_items,
            results_manager=self.results_manager,
            rx_telemetry=getattr(self, "_rx_telemetry", None),
        )
        self._batch_worker.progress_updated.connect(self._on_batch_progress_updated)
        self._batch_worker.batch_finished.connect(self._on_batch_segmentation_finished)
        self._batch_worker.start()

    def _on_cancel_batch_segmentation(self):
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.stop()
            self.segmentation_view.batch_status_lbl.setText("Cancelling batch segmentation...")

    def _on_batch_progress_updated(self, curr, total, image_id, dice, iou, latency_ms):
        self.segmentation_view.update_batch_progress(curr, total, image_id, dice, iou, latency_ms)
        self.dashboard_view.card_eval.set_value(f"Segmenting: {curr}/{total}")

    def _on_batch_segmentation_finished(self, processed, total, avg_dice, avg_iou, summary):
        self.segmentation_view.set_batch_running(False)
        self.dashboard_view.card_eval.set_value(f"Avg Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
        # Mark results dirty and reload once at batch completion
        self.results_view.load_records(force=True)
        QMessageBox.information(
            self,
            "Batch Segmentation Complete",
            summary + "\n\nAll results are stored in SQLite and CSV, and ready in Experiment Results.",
        )
        self._batch_worker = None

    def _on_evaluation_completed(self, metadata, dice, iou, psnr=None, ssim=None):
        if dice is not None and iou is not None:
            self.dashboard_view.card_eval.set_value(f"Dice: {dice:.4f} | IoU: {iou:.4f}")
        else:
            self.dashboard_view.card_eval.set_value("No GT Mask Found")

        if metadata is not None:
            ratio = (
                (metadata.original_file_size / metadata.compressed_file_size)
                if metadata.compressed_file_size > 0
                else 1.0
            )
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Look up measured transmission telemetry so we don't overwrite with None
            tx_time_val, thru_val = None, None
            if hasattr(self, "_rx_telemetry") and metadata.experiment_id in self._rx_telemetry:
                cached_time, cached_thru = self._rx_telemetry[metadata.experiment_id]
                if cached_time > 0.0:
                    tx_time_val = cached_time
                if cached_thru > 0.0:
                    thru_val = cached_thru

            rec = ExperimentRecord(
                experiment_id=metadata.experiment_id,
                timestamp=now_str,
                image_id=metadata.image_id,
                image_category=metadata.image_category,
                compression_method=metadata.compression_mode,
                compression_quality=metadata.compression_quality,
                region_adaptive_margin=metadata.region_adaptive_margin,
                compression_model_version=metadata.compression_model_version,
                original_file_size=metadata.original_file_size,
                compressed_file_size=metadata.compressed_file_size,
                compression_ratio=ratio,
                psnr=psnr,
                ssim=ssim,
                bandwidth_limit=None,
                packet_loss_rate=None,
                network_delay_ms=None,
                theoretical_transmission_time_sec=None,
                transmission_time_sec=tx_time_val,
                measured_throughput_kbps=thru_val,
                checksum_verified=True,
                dice_coefficient=dice,
                iou=iou,
                notes="Receiver evaluated",
            )
            self.results_manager.record_or_update_experiment(rec)
            # Mark results dirty so opening results tab will reload, without freezing GUI during evaluation
            self.results_view.mark_dirty()
