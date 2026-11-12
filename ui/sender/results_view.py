"""Sender Experiment Results View with Filtering and Aggregated Analytics."""

from collections import defaultdict
import csv
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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
    """Displays genuine historical experiment logs and exports with search and summary analytics."""

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
        self._all_records: List[Dict[str, Any]] = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
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
        self.export_summary_btn = QPushButton("Export Summary Report")
        self.export_summary_btn.setObjectName("SecondaryBtn")

        self.records_count_lbl = QLabel("Total Records: 0 (Showing: 0)")
        self.records_count_lbl.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 12px;")

        act_row.addWidget(self.refresh_btn)
        act_row.addWidget(self.export_btn)
        act_row.addWidget(self.export_summary_btn)
        act_row.addStretch()
        act_row.addWidget(self.records_count_lbl)
        layout.addLayout(act_row)

        # Filter bar
        filter_frame = QFrame()
        filter_frame.setObjectName("CardFrame")
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(12, 6, 12, 6)

        filter_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter by Image ID or Experiment ID...")
        self.search_input.textChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.search_input, stretch=2)

        filter_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["All Qualities", "Q90", "Q70", "Q50", "Q30"])
        self.quality_combo.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.quality_combo)

        filter_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["All Modes", "Fixed Rate", "Adaptive"])
        self.mode_combo.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.mode_combo)

        layout.addWidget(filter_frame)

        # Results table
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        self.refresh_btn.clicked.connect(self.load_records)
        self.export_btn.clicked.connect(self._export_csv)
        self.export_summary_btn.clicked.connect(self._export_summary)
        self.load_records()

    def load_records(self):
        """Loads records from database and applies active filters."""
        self._all_records = self.results_manager.fetch_all()
        self._apply_filters()

    def _apply_filters(self):
        """Filters in-memory records and populates table cells."""
        search_txt = self.search_input.text().strip().lower()
        q_sel = self.quality_combo.currentText()
        m_sel = self.mode_combo.currentText()

        filtered = []
        for r in self._all_records:
            # 1. Search text filter
            exp_id = str(r.get("experiment_id") or "").lower()
            img_id = str(r.get("image_id") or "").lower()
            if search_txt and (search_txt not in exp_id and search_txt not in img_id):
                continue

            # 2. Quality filter
            q_val = r.get("compression_quality")
            if q_sel != "All Qualities":
                target_q = int(q_sel.replace("Q", ""))
                if q_val != target_q:
                    continue

            # 3. Mode filter
            notes = str(r.get("notes") or "").lower()
            is_adaptive = "adaptive" in notes or "adaptive" in exp_id
            if m_sel == "Fixed Rate" and is_adaptive:
                continue
            if m_sel == "Adaptive" and not is_adaptive:
                continue

            filtered.append(r)

        self.records_count_lbl.setText(f"Total Records: {len(self._all_records)} (Showing: {len(filtered)})")
        self.table.setRowCount(len(filtered))

        for row_idx, r in enumerate(filtered):
            # 0: Experiment ID
            item_exp = QTableWidgetItem(str(r.get("experiment_id") or ""))
            item_exp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 0, item_exp)

            # 1: Timestamp
            item_ts = QTableWidgetItem(str(r.get("timestamp") or ""))
            item_ts.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 1, item_ts)

            # 2: Image ID
            item_img = QTableWidgetItem(str(r.get("image_id") or ""))
            self.table.setItem(row_idx, 2, item_img)

            # 3: Quality
            q_val = r.get("compression_quality")
            q_str = f"Q{q_val}" if q_val is not None else "-"
            item_q = QTableWidgetItem(q_str)
            item_q.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 3, item_q)

            # 4: Orig Size
            orig_sz = r.get("original_file_size")
            orig_sz_str = f"{orig_sz:,}" if orig_sz is not None else "-"
            item_osz = QTableWidgetItem(orig_sz_str)
            item_osz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row_idx, 4, item_osz)

            # 5: Comp Size
            comp_sz = r.get("compressed_file_size")
            comp_sz_str = f"{comp_sz:,}" if comp_sz is not None else "-"
            item_csz = QTableWidgetItem(comp_sz_str)
            item_csz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row_idx, 5, item_csz)

            # 6: Ratio
            ratio = r.get("compression_ratio")
            ratio_str = f"{ratio:.2f}:1" if ratio is not None else "-"
            item_ratio = QTableWidgetItem(ratio_str)
            item_ratio.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 6, item_ratio)

            # 7: PSNR
            psnr = r.get("psnr")
            psnr_str = f"{psnr:.2f}" if psnr is not None else "-"
            item_psnr = QTableWidgetItem(psnr_str)
            item_psnr.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row_idx, 7, item_psnr)

            # 8: SSIM
            ssim = r.get("ssim")
            ssim_str = f"{ssim:.4f}" if ssim is not None else "-"
            item_ssim = QTableWidgetItem(ssim_str)
            item_ssim.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row_idx, 8, item_ssim)

            # 9: Bandwidth
            bw = str(r.get("bandwidth_limit") or "unshaped")
            item_bw = QTableWidgetItem(bw)
            item_bw.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 9, item_bw)

            # 10: Loss %
            loss = r.get("packet_loss_rate", 0.0)
            item_loss = QTableWidgetItem(f"{loss}%")
            item_loss.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 10, item_loss)

            # 11: Delay (ms)
            delay = r.get("network_delay_ms", 0.0)
            item_delay = QTableWidgetItem(f"{delay} ms")
            item_delay.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row_idx, 11, item_delay)

            # 12: Theoretical Tx (s)
            theo = r.get("theoretical_transmission_time_sec")
            theo_str = f"{theo:.4f}" if theo is not None else "-"
            item_theo = QTableWidgetItem(theo_str)
            item_theo.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row_idx, 12, item_theo)

            # 13: Measured Tx (s)
            tx = r.get("transmission_time_sec")
            tx_str = f"{tx:.4f}" if tx is not None else "-"
            item_tx = QTableWidgetItem(tx_str)
            item_tx.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row_idx, 13, item_tx)

            # 14: Throughput (Kbps)
            thru = r.get("measured_throughput_kbps")
            thru_str = f"{thru:,.2f}" if thru is not None else "-"
            item_thru = QTableWidgetItem(thru_str)
            item_thru.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row_idx, 14, item_thru)

            # 15: Checksum
            chk = r.get("checksum_verified")
            chk_str = "Verified" if chk else "Failed"
            item_chk = QTableWidgetItem(chk_str)
            item_chk.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if chk:
                item_chk.setForeground(QColor("#10b981"))
            else:
                item_chk.setForeground(QColor("#ef4444"))
            self.table.setItem(row_idx, 15, item_chk)

    def _export_csv(self):
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Experiment Records",
            str(EXPERIMENTS_DIR / "sender_experiment_records.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        src_csv = self.results_manager.csv_path
        if src_csv.exists() and src_csv.stat().st_size > 0:
            shutil.copyfile(src_csv, target_path)
            QMessageBox.information(self, "Export Complete", f"Experiment records exported to:\n{target_path}")
        else:
            QMessageBox.warning(self, "No Records", "No experiment records available to export yet.")

    def _export_summary(self):
        """Generates and exports an academic aggregated summary table grouped by quality factor."""
        if not self._all_records:
            QMessageBox.warning(self, "No Data", "No experiment records to summarize.")
            return

        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Experiment Summary Table",
            str(EXPERIMENTS_DIR / "sender_experiment_summary_report.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        groups = defaultdict(list)
        for r in self._all_records:
            q = r.get("compression_quality", 0)
            groups[q].append(r)

        summary_rows = []
        for q in sorted(groups.keys()):
            recs = groups[q]
            n = len(recs)
            psnrs = [r["psnr"] for r in recs if r.get("psnr") is not None]
            ssims = [r["ssim"] for r in recs if r.get("ssim") is not None]
            ratios = [r["compression_ratio"] for r in recs if r.get("compression_ratio") is not None]
            tx_times = [r["transmission_time_sec"] for r in recs if r.get("transmission_time_sec") is not None]
            thru = [r["measured_throughput_kbps"] for r in recs if r.get("measured_throughput_kbps") is not None]

            def _fmt(vals):
                if not vals:
                    return "-"
                return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

            summary_rows.append({
                "Quality_Tier": f"Q{q}",
                "Trial_Count": n,
                "Compression_Ratio": f"{np.mean(ratios):.2f}:1" if ratios else "-",
                "PSNR_dB": _fmt(psnrs),
                "SSIM": _fmt(ssims),
                "Transmission_Time_Sec": _fmt(tx_times),
                "Measured_Throughput_Kbps": _fmt(thru),
            })

        fieldnames = [
            "Quality_Tier",
            "Trial_Count",
            "Compression_Ratio",
            "PSNR_dB",
            "SSIM",
            "Transmission_Time_Sec",
            "Measured_Throughput_Kbps",
        ]

        with open(target_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        QMessageBox.information(
            self,
            "Summary Export Complete",
            f"Academic experiment summary table exported successfully to:\n{target_path}",
        )
