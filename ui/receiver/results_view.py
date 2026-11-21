"""Receiver Experiment Results & Evaluation Telemetry View."""

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
from common.logging_config import setup_logger
from common.results_manager import ResultsManager
from ui.shared.widgets import SectionHeader

logger = setup_logger("ReceiverResultsView")


class ReceiverResultsView(QWidget):
    """Complete traceable log of received frames, transmission metrics, and segmentation scores."""

    HEADERS = [
        "Experiment ID",
        "Image ID",
        "Quality",
        "Comp Size",
        "Tx Time (s)",
        "Throughput (Kbps)",
        "Checksum",
        "Dice",
        "IoU",
        "PSNR",
        "SSIM",
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
            "Receiver Evaluation Records & Research Telemetry",
            "Traceable records of received images, measured transmission metrics, Dice, and IoU scores (Zero-Dummy Policy)",
        )
        layout.addWidget(header)

        # Action row
        act_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Records")
        self.refresh_btn.setObjectName("SecondaryBtn")
        self.export_btn = QPushButton("Export CSV")
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

        # Results table (Interactive mode with pre-configured column widths for 60 FPS performance)
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        col_widths = [220, 140, 70, 95, 90, 110, 85, 75, 75, 85, 80]
        for col_idx, width in enumerate(col_widths):
            self.table.setColumnWidth(col_idx, width)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, stretch=1)

        self._is_dirty = True
        self.refresh_btn.clicked.connect(lambda: self.load_records(force=True))
        self.export_btn.clicked.connect(self._export_csv)
        self.export_summary_btn.clicked.connect(self._export_summary)
        self.load_records(force=True)

    def mark_dirty(self):
        """Flags records as modified so the table refreshes smoothly when viewed."""
        self._is_dirty = True

    def load_records(self, force: bool = False, *args, **kwargs):
        """Loads records from database with smart caching to eliminate UI freezes."""
        if force or self._is_dirty or not self._all_records:
            try:
                self._all_records = self.results_manager.fetch_all()
                self._is_dirty = False
            except Exception as e:
                logger.error(f"Error fetching records: {e}")
        self._apply_filters()

    def _apply_filters(self):
        """Filters in-memory records and populates table cells without triggering redraw stalls."""
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

        # Freeze table updates during bulk insertion to prevent thousands of layout recalculations
        self.table.setUpdatesEnabled(False)
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(filtered))

            for row_idx, r in enumerate(filtered):
                # 0: Experiment ID
                item_exp = QTableWidgetItem(str(r.get("experiment_id") or ""))
                item_exp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_idx, 0, item_exp)

                # 1: Image ID
                item_img = QTableWidgetItem(str(r.get("image_id") or ""))
                self.table.setItem(row_idx, 1, item_img)

                # 2: Quality
                q_val = r.get("compression_quality")
                q_str = f"Q{q_val}" if q_val is not None else "-"
                item_q = QTableWidgetItem(q_str)
                item_q.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_idx, 2, item_q)

                # 3: Comp Size
                comp_sz = r.get("compressed_file_size")
                comp_sz_str = f"{comp_sz:,} B" if comp_sz is not None else "-"
                item_sz = QTableWidgetItem(comp_sz_str)
                item_sz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_idx, 3, item_sz)

                # 4: Tx Time (s)
                tx_time = r.get("transmission_time_sec")
                tx_str = f"{tx_time:.4f}" if tx_time is not None else "-"
                item_tx = QTableWidgetItem(tx_str)
                item_tx.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_idx, 4, item_tx)

                # 5: Throughput (Kbps)
                thru = r.get("measured_throughput_kbps")
                thru_str = f"{thru:,.2f}" if thru is not None else "-"
                item_thru = QTableWidgetItem(thru_str)
                item_thru.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_idx, 5, item_thru)

                # 6: Checksum
                chk = r.get("checksum_verified")
                chk_str = "Verified" if chk else "Failed"
                item_chk = QTableWidgetItem(chk_str)
                item_chk.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if chk:
                    item_chk.setForeground(QColor("#10b981"))
                else:
                    item_chk.setForeground(QColor("#ef4444"))
                self.table.setItem(row_idx, 6, item_chk)

                # 7: Dice
                dice = r.get("dice_coefficient")
                dice_str = f"{dice:.4f}" if dice is not None else "-"
                item_dice = QTableWidgetItem(dice_str)
                item_dice.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if dice is not None:
                    if dice >= 0.70:
                        item_dice.setForeground(QColor("#10b981"))
                    elif dice >= 0.50:
                        item_dice.setForeground(QColor("#38bdf8"))
                    else:
                        item_dice.setForeground(QColor("#f59e0b"))
                self.table.setItem(row_idx, 7, item_dice)

                # 8: IoU
                iou = r.get("iou")
                iou_str = f"{iou:.4f}" if iou is not None else "-"
                item_iou = QTableWidgetItem(iou_str)
                item_iou.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if iou is not None and iou >= 0.50:
                    item_iou.setForeground(QColor("#38bdf8"))
                self.table.setItem(row_idx, 8, item_iou)

                # 9: PSNR
                psnr = r.get("psnr")
                psnr_str = f"{psnr:.2f} dB" if psnr is not None else "-"
                item_psnr = QTableWidgetItem(psnr_str)
                item_psnr.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_idx, 9, item_psnr)

                # 10: SSIM
                ssim = r.get("ssim")
                ssim_str = f"{ssim:.4f}" if ssim is not None else "-"
                item_ssim = QTableWidgetItem(ssim_str)
                item_ssim.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_idx, 10, item_ssim)
        finally:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)

    def _export_csv(self):
        """Exports the raw records to a user-selected CSV file path."""
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Receiver Evaluation Records",
            str(EXPERIMENTS_DIR / "receiver_evaluation_records.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        src_csv = self.results_manager.csv_path
        if src_csv.exists() and src_csv.stat().st_size > 0:
            shutil.copyfile(src_csv, target_path)
            QMessageBox.information(self, "Export Complete", f"Evaluation records exported to:\n{target_path}")
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
            str(EXPERIMENTS_DIR / "experiment_summary_report.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        # Group by quality
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
            dices = [r["dice_coefficient"] for r in recs if r.get("dice_coefficient") is not None]
            ious = [r["iou"] for r in recs if r.get("iou") is not None]
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
                "Dice_DSC": _fmt(dices),
                "IoU": _fmt(ious),
                "Transmission_Time_Sec": _fmt(tx_times),
                "Measured_Throughput_Kbps": _fmt(thru),
            })

        fieldnames = [
            "Quality_Tier",
            "Trial_Count",
            "Compression_Ratio",
            "PSNR_dB",
            "SSIM",
            "Dice_DSC",
            "IoU",
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
