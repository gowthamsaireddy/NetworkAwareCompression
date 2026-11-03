"""Sender Experiment Results View."""

import csv
from pathlib import Path
import shutil
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from common.config import EXPERIMENTS_DIR
from common.results_manager import ResultsManager
from ui.shared.widgets import SectionHeader


class SenderResultsView(QWidget):
    """Displays genuine historical experiment logs and exports."""

    HEADERS = [
        "Experiment ID",
        "Timestamp",
        "Image ID",
        "Quality",
        "Orig Size (B)",
        "Comp Size (B)",
        "Ratio",
        "PSNR (dB)",
        "SSIM",
        "Bandwidth",
        "Loss %",
        "Delay (ms)",
        "Theoretical Tx (s)",
        "Measured Tx (s)",
        "Throughput (Kbps)",
        "Checksum",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_manager = ResultsManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Experiment Records & Telemetry",
            "Traceable records of measured compression and transmission runs (Zero dummy data policy)",
        )
        layout.addWidget(header)

        # Action bar
        act_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Records")
        self.refresh_btn.setObjectName("SecondaryBtn")
        self.export_btn = QPushButton("Export CSV File")
        self.export_btn.setObjectName("SecondaryBtn")

        self.records_count_lbl = QLabel("Total Records: 0")
        self.records_count_lbl.setStyleSheet("color: #94a3b8; font-weight: 600;")

        act_row.addWidget(self.refresh_btn)
        act_row.addWidget(self.export_btn)
        act_row.addStretch()
        act_row.addWidget(self.records_count_lbl)
        layout.addLayout(act_row)

        # Results table
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        self.refresh_btn.clicked.connect(self.load_records)
        self.export_btn.clicked.connect(self._export_csv)
        self.load_records()

    def load_records(self):
        """Loads records from database (Zero-Dummy: table remains empty until real trials run)."""
        records = self.results_manager.fetch_all()
        self.records_count_lbl.setText(f"Total Records: {len(records)}")
        self.table.setRowCount(len(records))

        for row_idx, r in enumerate(records):
            self.table.setItem(row_idx, 0, QTableWidgetItem(str(r.get("experiment_id", ""))))
            self.table.setItem(row_idx, 1, QTableWidgetItem(str(r.get("timestamp", ""))))
            self.table.setItem(row_idx, 2, QTableWidgetItem(str(r.get("image_id", ""))))
            self.table.setItem(row_idx, 3, QTableWidgetItem(f"Q{r.get('compression_quality', '')}"))
            self.table.setItem(row_idx, 4, QTableWidgetItem(f"{r.get('original_file_size', 0):,}"))
            self.table.setItem(row_idx, 5, QTableWidgetItem(f"{r.get('compressed_file_size', 0):,}"))
            self.table.setItem(row_idx, 6, QTableWidgetItem(f"{r.get('compression_ratio', 0.0):.2f}:1"))

            psnr = r.get("psnr")
            psnr_str = f"{psnr:.2f}" if psnr is not None else "-"
            self.table.setItem(row_idx, 7, QTableWidgetItem(psnr_str))

            ssim = r.get("ssim")
            ssim_str = f"{ssim:.4f}" if ssim is not None else "-"
            self.table.setItem(row_idx, 8, QTableWidgetItem(ssim_str))

            self.table.setItem(row_idx, 9, QTableWidgetItem(str(r.get("bandwidth_limit") or "unshaped")))
            self.table.setItem(row_idx, 10, QTableWidgetItem(f"{r.get('packet_loss_rate', 0.0)}%"))
            self.table.setItem(row_idx, 11, QTableWidgetItem(f"{r.get('network_delay_ms', 0.0)} ms"))

            theo_tx = r.get("theoretical_transmission_time_sec")
            theo_str = f"{theo_tx:.4f}" if theo_tx is not None else "-"
            self.table.setItem(row_idx, 12, QTableWidgetItem(theo_str))

            tx_time = r.get("transmission_time_sec")
            tx_str = f"{tx_time:.4f}" if tx_time is not None else "-"
            self.table.setItem(row_idx, 13, QTableWidgetItem(tx_str))

            thru = r.get("measured_throughput_kbps")
            thru_str = f"{thru:.2f}" if thru is not None else "-"
            self.table.setItem(row_idx, 14, QTableWidgetItem(thru_str))

            chk = r.get("checksum_verified")
            chk_str = "Verified" if chk else "Failed"
            self.table.setItem(row_idx, 15, QTableWidgetItem(chk_str))

    def _export_csv(self):
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Experiment Records",
            str(EXPERIMENTS_DIR / "exported_experiments.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        src_csv = self.results_manager.csv_path
        if src_csv.exists():
            shutil.copyfile(src_csv, target_path)
            QMessageBox.information(self, "Export Complete", f"Experiment records exported to:\n{target_path}")
        else:
            QMessageBox.warning(self, "No Records", "No CSV experiment records found to export.")
