"""Receiver Experiment Results & Image-Wise Evaluation Telemetry View.

Implements:
1. Flat traceable log of all individual experiment trials.
2. Original Image ID-based experiment grouping and tracking across test images.
3. Dedicated separate image ID-wise comparison of compression methods and network profiles.
4. Export utilities for research reproducibility and paper writing.
"""

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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from common.config import EXPERIMENTS_DIR
from common.logging_config import setup_logger
from common.results_manager import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_INCOMPLETE,
    ResultsManager,
)
from receiver.segmentation_metrics import get_canonical_busi_id
from ui.shared.widgets import MetricBox, SectionHeader

logger = setup_logger("ReceiverResultsView")


class ReceiverResultsView(QWidget):
    """Traceable results log and dedicated original image ID-wise tracking view."""

    FLAT_HEADERS = [
        "Original Image ID",
        "Experiment ID",
        "Method",
        "Quality",
        "Network Profile",
        "Comp Size",
        "Tx Time (s)",
        "Throughput (Kbps)",
        "Checksum",
        "Dice",
        "IoU",
        "PSNR",
        "SSIM",
        "Status",
    ]

    VARIANT_HEADERS = [
        "Experiment ID",
        "Method",
        "Quality",
        "Network Profile",
        "Comp Size",
        "Ratio",
        "Tx Time (s)",
        "Throughput (Kbps)",
        "PSNR (dB)",
        "SSIM",
        "Dice (DSC)",
        "IoU",
        "Status",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_manager = ResultsManager()
        self._all_records: List[Dict[str, Any]] = []
        self._grouped_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._is_dirty = True
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        header = SectionHeader(
            "Receiver Evaluation Records & Image-Wise Telemetry",
            "Traceable records of received ultrasound frames and separate original image ID-wise experimental comparison",
        )
        layout.addWidget(header)

        # Main tabs: 1) Image ID-Wise Results, 2) All Experiment Trials Log
        self.tabs = QTabWidget()
        self.tabs.setObjectName("ResultsTabWidget")

        self.tab_imagewise = QWidget()
        self.tab_flat = QWidget()

        self._init_imagewise_tab()
        self._init_flat_tab()

        self.tabs.addTab(self.tab_imagewise, "Image ID-Wise Results")
        self.tabs.addTab(self.tab_flat, "All Experiment Trials Log")

        layout.addWidget(self.tabs, stretch=1)

        self.load_records(force=True)

    def _init_imagewise_tab(self):
        """Initializes Tab 1: Dedicated Image ID-Wise Comparison View."""
        layout = QVBoxLayout(self.tab_imagewise)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Top Image Selection & Sort Bar
        sel_frame = QFrame()
        sel_frame.setObjectName("CardFrame")
        sel_layout = QHBoxLayout(sel_frame)
        sel_layout.setContentsMargins(12, 8, 12, 8)
        sel_layout.setSpacing(10)

        sel_layout.addWidget(QLabel("Select Original Image:"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(280)
        self.image_combo.currentIndexChanged.connect(self._on_image_selected)
        sel_layout.addWidget(self.image_combo)

        sel_layout.addWidget(QLabel("Sort By:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Order Executed",
            "Highest Dice Similarity (DSC)",
            "Highest IoU (Jaccard Index)",
            "Highest PSNR (Image Fidelity)",
            "Highest SSIM (Structural Quality)",
            "Highest Compression Ratio (Storage)",
            "Fastest Transmission (Low Latency)",
            "Method / Quality",
        ])
        self.sort_combo.currentIndexChanged.connect(self._populate_image_variants_table)
        sel_layout.addWidget(self.sort_combo)

        sel_layout.addStretch()

        self.btn_export_image = QPushButton("Export This Image (CSV)")
        self.btn_export_image.setObjectName("SecondaryBtn")
        self.btn_export_image.clicked.connect(self._export_selected_image_csv)
        sel_layout.addWidget(self.btn_export_image)

        self.btn_export_all_imagewise = QPushButton("Export Per-Image Summary (CSV)")
        self.btn_export_all_imagewise.setObjectName("SecondaryBtn")
        self.btn_export_all_imagewise.clicked.connect(self._export_all_imagewise_summary)
        sel_layout.addWidget(self.btn_export_all_imagewise)

        layout.addWidget(sel_frame)

        # 2. Key Analytics Metric Cards for Selected Original Image
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)

        self.card_image_id = MetricBox("Selected Image ID", "None")
        self.card_accuracy = MetricBox("Clinical Accuracy", "Max: - | Mean: -")
        self.card_fidelity = MetricBox("Reconstruction Fidelity", "PSNR: - | SSIM: -")
        self.card_efficiency = MetricBox("Transmission Telemetry", "CR: - | Time: -")

        cards_layout.addWidget(self.card_image_id, 2)
        cards_layout.addWidget(self.card_accuracy, 2)
        cards_layout.addWidget(self.card_fidelity, 2)
        cards_layout.addWidget(self.card_efficiency, 2)
        layout.addLayout(cards_layout)

        # 3. Image Variants Comparison Table
        self.image_table = QTableWidget(0, len(self.VARIANT_HEADERS))
        self.image_table.setHorizontalHeaderLabels(self.VARIANT_HEADERS)
        self.image_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.image_table.horizontalHeader().setStretchLastSection(False)
        self.image_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.image_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        var_col_widths = [
            350,  # Experiment ID (full run name visible)
            105,  # Method
            70,   # Quality
            125,  # Network Profile
            100,  # Comp Size
            75,   # Ratio
            95,   # Tx Time (s)
            140,  # Throughput (Kbps) (header fully visible)
            85,   # PSNR (dB)
            75,   # SSIM
            90,   # Dice (DSC)
            80,   # IoU
            105,  # Status
        ]
        for c_idx, width in enumerate(var_col_widths):
            if c_idx < len(self.VARIANT_HEADERS):
                self.image_table.setColumnWidth(c_idx, width)
        self.image_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.image_table, stretch=1)

    def _init_flat_tab(self):
        """Initializes Tab 2: Flat view of all experiment trials."""
        layout = QVBoxLayout(self.tab_flat)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Action row
        act_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Records")
        self.refresh_btn.setObjectName("SecondaryBtn")
        self.export_btn = QPushButton("Export All Records (CSV)")
        self.export_btn.setObjectName("SecondaryBtn")
        self.export_summary_btn = QPushButton("Export Quality Summary")
        self.export_summary_btn.setObjectName("SecondaryBtn")

        self.records_count_lbl = QLabel("Total Records: 0 (Showing: 0)")
        self.records_count_lbl.setStyleSheet("color: #94a3b8; font-weight: 600; font-size: 12px;")

        act_row.addWidget(self.refresh_btn)
        act_row.addWidget(self.export_btn)
        act_row.addWidget(self.export_summary_btn)
        act_row.addStretch()
        act_row.addWidget(self.records_count_lbl)
        layout.addLayout(act_row)

        # Filter row
        filter_frame = QFrame()
        filter_frame.setObjectName("CardFrame")
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(10, 6, 10, 6)

        filter_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter by Image ID or Experiment ID...")
        self.search_input.textChanged.connect(self._apply_flat_filters)
        filter_layout.addWidget(self.search_input, stretch=2)

        filter_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["All Qualities", "Q90", "Q70", "Q50", "Q30"])
        self.quality_combo.currentTextChanged.connect(self._apply_flat_filters)
        filter_layout.addWidget(self.quality_combo)

        filter_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["All Methods", "Uniform JPEG", "Region-Adaptive"])
        self.method_combo.currentTextChanged.connect(self._apply_flat_filters)
        filter_layout.addWidget(self.method_combo)

        layout.addWidget(filter_frame)

        # Flat table
        self.flat_table = QTableWidget(0, len(self.FLAT_HEADERS))
        self.flat_table.setHorizontalHeaderLabels(self.FLAT_HEADERS)
        self.flat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.flat_table.horizontalHeader().setStretchLastSection(False)
        self.flat_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.flat_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        flat_col_widths = [
            180,  # Original Image ID
            350,  # Experiment ID
            105,  # Method
            70,   # Quality
            125,  # Network Profile
            100,  # Comp Size
            95,   # Tx Time (s)
            140,  # Throughput (Kbps)
            85,   # Checksum
            90,   # Dice
            80,   # IoU
            85,   # PSNR
            75,   # SSIM
            105,  # Status
        ]
        for col_idx, width in enumerate(flat_col_widths):
            if col_idx < len(self.FLAT_HEADERS):
                self.flat_table.setColumnWidth(col_idx, width)
        self.flat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.flat_table, stretch=1)

        self.refresh_btn.clicked.connect(lambda: self.load_records(force=True))
        self.export_btn.clicked.connect(self._export_csv)
        self.export_summary_btn.clicked.connect(self._export_summary)

    def mark_dirty(self):
        """Flags records as modified so the tables refresh when viewed."""
        self._is_dirty = True

    def load_records(self, force: bool = False, *args, **kwargs):
        """Loads records from database with caching."""
        if force or self._is_dirty or not self._all_records:
            try:
                self._all_records = self.results_manager.fetch_all()
                self._group_records_by_original_image()
                self._is_dirty = False
            except Exception as e:
                logger.error(f"Error fetching records: {e}")

        self._apply_flat_filters()
        self._refresh_image_combo()

    def _group_records_by_original_image(self):
        """Groups all experiment variants under their persistent original_image_id."""
        self._grouped_records.clear()
        for r in self._all_records:
            orig_id = r.get("original_image_id")
            if not orig_id:
                raw_img = str(r.get("image_id") or "")
                orig_id, _, _, _ = get_canonical_busi_id(raw_img)
            self._grouped_records[orig_id].append(r)

    def _refresh_image_combo(self):
        """Populates the original image selector combo box."""
        current_sel = self.image_combo.currentText()
        self.image_combo.blockSignals(True)
        self.image_combo.clear()

        sorted_orig_ids = sorted(self._grouped_records.keys())
        for orig_id in sorted_orig_ids:
            variants = self._grouped_records[orig_id]
            self.image_combo.addItem(f"{orig_id} ({len(variants)} trials)", userData=orig_id)

        self.image_combo.blockSignals(False)

        # Restore previous selection if possible
        idx = self.image_combo.findText(current_sel)
        if idx >= 0:
            self.image_combo.setCurrentIndex(idx)
        elif self.image_combo.count() > 0:
            self.image_combo.setCurrentIndex(0)

        self._populate_image_variants_table()

    def _on_image_selected(self):
        self._populate_image_variants_table()

    def _populate_image_variants_table(self):
        """Populates the variant comparison table for the selected original image."""
        if self.image_combo.currentIndex() < 0:
            return

        selected_orig_id = self.image_combo.currentData()
        if not selected_orig_id:
            return

        variants = list(self._grouped_records.get(selected_orig_id, []))
        if not variants:
            self.image_table.setRowCount(0)
            self.card_image_id.set_value("No trials")
            self.card_accuracy.set_value("-")
            self.card_fidelity.set_value("-")
            self.card_efficiency.set_value("-")
            return

        # Sort variants based on selected criteria
        sort_mode = self.sort_combo.currentText()
        if "Highest Dice" in sort_mode:
            variants.sort(key=lambda x: (x.get("dice_coefficient") or 0.0), reverse=True)
        elif "Highest IoU" in sort_mode:
            variants.sort(key=lambda x: (x.get("iou") or 0.0), reverse=True)
        elif "Highest PSNR" in sort_mode:
            variants.sort(key=lambda x: (x.get("psnr") or 0.0), reverse=True)
        elif "Highest SSIM" in sort_mode:
            variants.sort(key=lambda x: (x.get("ssim") or 0.0), reverse=True)
        elif "Highest Compression Ratio" in sort_mode:
            variants.sort(key=lambda x: (x.get("compression_ratio") or 0.0), reverse=True)
        elif "Fastest Transmission" in sort_mode:
            variants.sort(key=lambda x: (x.get("transmission_time_sec") or 9999.0))
        elif "Method / Quality" in sort_mode:
            variants.sort(key=lambda x: (str(x.get("compression_method")), -(x.get("compression_quality") or 0)))

        # Compute summary metrics for this image ID across all variants
        dices = [v["dice_coefficient"] for v in variants if v.get("dice_coefficient") is not None]
        ious = [v["iou"] for v in variants if v.get("iou") is not None]
        psnrs = [v["psnr"] for v in variants if v.get("psnr") is not None]
        ssims = [v["ssim"] for v in variants if v.get("ssim") is not None]
        ratios = [v["compression_ratio"] for v in variants if v.get("compression_ratio") is not None]
        times = [v["transmission_time_sec"] for v in variants if v.get("transmission_time_sec") is not None]

        # 1. Image ID & count card
        base_stem = selected_orig_id.split("/")[-1] if "/" in selected_orig_id else selected_orig_id
        self.card_image_id.set_value(base_stem, subtitle=f"{selected_orig_id} • {len(variants)} trials")

        # 2. Clinical Accuracy card
        if dices:
            max_d = max(dices)
            mean_d = float(np.mean(dices))
            mean_i = float(np.mean(ious)) if ious else 0.0
            self.card_accuracy.set_value(
                f"Max DSC: {max_d:.4f}",
                subtitle=f"Mean DSC: {mean_d:.4f}  |  Mean IoU: {mean_i:.4f}",
                color="#10b981",
            )
        else:
            self.card_accuracy.set_value("Unverified", subtitle="No ground truth mask found", color="#f59e0b")

        # 3. Fidelity card
        if psnrs and ssims:
            mean_p = float(np.mean(psnrs))
            mean_s = float(np.mean(ssims))
            self.card_fidelity.set_value(
                f"PSNR: {mean_p:.1f} dB",
                subtitle=f"Mean SSIM: {mean_s:.4f}",
                color="#38bdf8",
            )
        else:
            self.card_fidelity.set_value("-", subtitle="Fidelity not computed")

        # 4. Efficiency card
        if ratios and times:
            mean_cr = float(np.mean(ratios))
            mean_t = float(np.mean(times))
            self.card_efficiency.set_value(
                f"Ratio: {mean_cr:.1f}×",
                subtitle=f"Mean Latency: {mean_t:.4f}s",
                color="#38bdf8",
            )
        else:
            self.card_efficiency.set_value("-", subtitle="Efficiency metrics pending")

        # Populate Table
        self.image_table.setUpdatesEnabled(False)
        self.image_table.blockSignals(True)
        try:
            self.image_table.setRowCount(len(variants))
            for row_idx, r in enumerate(variants):
                # 0: Experiment ID
                full_exp = str(r.get("experiment_id") or "")
                item_exp = QTableWidgetItem(full_exp)
                item_exp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_exp.setToolTip(full_exp)
                self.image_table.setItem(row_idx, 0, item_exp)

                # 1: Method
                method = r.get("compression_method") or "NORMAL"
                item_m = QTableWidgetItem(method)
                item_m.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.image_table.setItem(row_idx, 1, item_m)

                # 2: Quality Config
                q_val = r.get("compression_quality")
                item_q = QTableWidgetItem(f"Q{q_val}" if q_val is not None else "-")
                item_q.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.image_table.setItem(row_idx, 2, item_q)

                # 3: Network Profile
                prof = r.get("network_profile") or "Unshaped"
                item_prof = QTableWidgetItem(prof)
                item_prof.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.image_table.setItem(row_idx, 3, item_prof)

                # 4: Comp Size
                sz = r.get("compressed_file_size")
                item_sz = QTableWidgetItem(f"{sz:,} B" if sz is not None else "-")
                item_sz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.image_table.setItem(row_idx, 4, item_sz)

                # 5: Compression Ratio
                cr = r.get("compression_ratio")
                item_cr = QTableWidgetItem(f"{cr:.2f}×" if cr is not None else "-")
                item_cr.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.image_table.setItem(row_idx, 5, item_cr)

                # 6: Tx Time (s)
                tx = r.get("transmission_time_sec")
                item_tx = QTableWidgetItem(f"{tx:.4f}" if tx is not None else "-")
                item_tx.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.image_table.setItem(row_idx, 6, item_tx)

                # 7: Throughput (Kbps)
                thru = r.get("measured_throughput_kbps")
                item_thru = QTableWidgetItem(f"{thru:,.1f}" if thru is not None else "-")
                item_thru.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.image_table.setItem(row_idx, 7, item_thru)

                # 8: PSNR (dB)
                psnr = r.get("psnr")
                item_psnr = QTableWidgetItem(f"{psnr:.2f}" if psnr is not None else "-")
                item_psnr.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.image_table.setItem(row_idx, 8, item_psnr)

                # 9: SSIM
                ssim = r.get("ssim")
                item_ssim = QTableWidgetItem(f"{ssim:.4f}" if ssim is not None else "-")
                item_ssim.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.image_table.setItem(row_idx, 9, item_ssim)

                # 10: Dice
                dice = r.get("dice_coefficient")
                item_dice = QTableWidgetItem(f"{dice:.4f}" if dice is not None else "-")
                item_dice.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if dice is not None:
                    if dice >= 0.70:
                        item_dice.setForeground(QColor("#10b981"))
                    elif dice >= 0.50:
                        item_dice.setForeground(QColor("#38bdf8"))
                    else:
                        item_dice.setForeground(QColor("#f59e0b"))
                self.image_table.setItem(row_idx, 10, item_dice)

                # 11: IoU
                iou = r.get("iou")
                item_iou = QTableWidgetItem(f"{iou:.4f}" if iou is not None else "-")
                item_iou.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if iou is not None and iou >= 0.50:
                    item_iou.setForeground(QColor("#38bdf8"))
                self.image_table.setItem(row_idx, 11, item_iou)

                # 12: Status
                stat = r.get("experiment_status") or STATUS_COMPLETED
                item_stat = QTableWidgetItem(stat)
                item_stat.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if stat == STATUS_COMPLETED:
                    item_stat.setForeground(QColor("#10b981"))
                elif stat == STATUS_INCOMPLETE:
                    item_stat.setForeground(QColor("#f59e0b"))
                else:
                    item_stat.setForeground(QColor("#ef4444"))
                self.image_table.setItem(row_idx, 12, item_stat)
        finally:
            self.image_table.setUpdatesEnabled(True)
            self.image_table.blockSignals(False)

    def _apply_flat_filters(self):
        """Filters in-memory records and populates Tab 2 table cells smoothly."""
        search_txt = self.search_input.text().strip().lower()
        q_sel = self.quality_combo.currentText()
        m_sel = self.method_combo.currentText()

        filtered = []
        for r in self._all_records:
            exp_id = str(r.get("experiment_id") or "").lower()
            img_id = str(r.get("image_id") or "").lower()
            orig_id = str(r.get("original_image_id") or "").lower()

            if search_txt and (search_txt not in exp_id and search_txt not in img_id and search_txt not in orig_id):
                continue

            q_val = r.get("compression_quality")
            if q_sel != "All Qualities":
                target_q = int(q_sel.replace("Q", ""))
                if q_val != target_q:
                    continue

            method = str(r.get("compression_method") or "").upper()
            if m_sel == "Uniform JPEG" and "REGION" in method:
                continue
            if m_sel == "Region-Adaptive" and "REGION" not in method:
                continue

            filtered.append(r)

        self.records_count_lbl.setText(f"Total Records: {len(self._all_records)} (Showing: {len(filtered)})")

        self.flat_table.setUpdatesEnabled(False)
        self.flat_table.blockSignals(True)
        try:
            self.flat_table.setRowCount(len(filtered))
            for row_idx, r in enumerate(filtered):
                orig_id = r.get("original_image_id") or r.get("image_id", "")
                self.flat_table.setItem(row_idx, 0, QTableWidgetItem(str(orig_id)))

                full_exp = str(r.get("experiment_id") or "")
                item_exp = QTableWidgetItem(full_exp)
                item_exp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_exp.setToolTip(full_exp)
                self.flat_table.setItem(row_idx, 1, item_exp)

                item_m = QTableWidgetItem(str(r.get("compression_method") or "NORMAL"))
                item_m.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.flat_table.setItem(row_idx, 2, item_m)

                q_val = r.get("compression_quality")
                item_q = QTableWidgetItem(f"Q{q_val}" if q_val is not None else "-")
                item_q.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.flat_table.setItem(row_idx, 3, item_q)

                prof = r.get("network_profile") or "Unshaped"
                item_prof = QTableWidgetItem(prof)
                item_prof.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.flat_table.setItem(row_idx, 4, item_prof)

                comp_sz = r.get("compressed_file_size")
                item_sz = QTableWidgetItem(f"{comp_sz:,} B" if comp_sz is not None else "-")
                item_sz.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.flat_table.setItem(row_idx, 5, item_sz)

                tx_time = r.get("transmission_time_sec")
                item_tx = QTableWidgetItem(f"{tx_time:.4f}" if tx_time is not None else "-")
                item_tx.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.flat_table.setItem(row_idx, 6, item_tx)

                thru = r.get("measured_throughput_kbps")
                item_thru = QTableWidgetItem(f"{thru:,.1f}" if thru is not None else "-")
                item_thru.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.flat_table.setItem(row_idx, 7, item_thru)

                chk = r.get("checksum_verified")
                item_chk = QTableWidgetItem("Verified" if chk else "Failed")
                item_chk.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_chk.setForeground(QColor("#10b981") if chk else QColor("#ef4444"))
                self.flat_table.setItem(row_idx, 8, item_chk)

                dice = r.get("dice_coefficient")
                item_dice = QTableWidgetItem(f"{dice:.4f}" if dice is not None else "-")
                item_dice.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if dice is not None and dice >= 0.70:
                    item_dice.setForeground(QColor("#10b981"))
                self.flat_table.setItem(row_idx, 9, item_dice)

                iou = r.get("iou")
                item_iou = QTableWidgetItem(f"{iou:.4f}" if iou is not None else "-")
                item_iou.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.flat_table.setItem(row_idx, 10, item_iou)

                psnr = r.get("psnr")
                item_psnr = QTableWidgetItem(f"{psnr:.2f}" if psnr is not None else "-")
                item_psnr.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.flat_table.setItem(row_idx, 11, item_psnr)

                ssim = r.get("ssim")
                item_ssim = QTableWidgetItem(f"{ssim:.4f}" if ssim is not None else "-")
                item_ssim.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.flat_table.setItem(row_idx, 12, item_ssim)

                stat = r.get("experiment_status") or STATUS_COMPLETED
                item_stat = QTableWidgetItem(stat)
                item_stat.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if stat == STATUS_COMPLETED:
                    item_stat.setForeground(QColor("#10b981"))
                elif stat == STATUS_INCOMPLETE:
                    item_stat.setForeground(QColor("#f59e0b"))
                else:
                    item_stat.setForeground(QColor("#ef4444"))
                self.flat_table.setItem(row_idx, 13, item_stat)
        finally:
            self.flat_table.setUpdatesEnabled(True)
            self.flat_table.blockSignals(False)

    def _export_csv(self):
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Receiver Evaluation Records",
            str(EXPERIMENTS_DIR / "receiver_evaluation_records.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return
        try:
            shutil.copy2(self.results_manager.csv_path, target_path)
            QMessageBox.information(self, "Export Complete", f"Exported {len(self._all_records)} records to {target_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV: {e}")

    def _export_selected_image_csv(self):
        selected_orig_id = self.image_combo.currentData()
        if not selected_orig_id:
            return
        variants = self._grouped_records.get(selected_orig_id, [])
        if not variants:
            QMessageBox.warning(self, "No Records", f"No records found for {selected_orig_id}")
            return

        clean_slug = selected_orig_id.replace("/", "_").replace(" ", "_")
        target_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Comparison for {selected_orig_id}",
            str(EXPERIMENTS_DIR / f"{clean_slug}_comparison.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        try:
            keys = [
                "original_image_id", "experiment_id", "compression_method",
                "compression_quality", "network_profile", "original_file_size",
                "compressed_file_size", "compression_ratio", "transmission_time_sec",
                "measured_throughput_kbps", "psnr", "ssim", "dice_coefficient",
                "iou", "experiment_status"
            ]
            with open(target_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for v in variants:
                    writer.writerow(v)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(variants)} variant records for '{selected_orig_id}' to:\n{target_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export image variants: {e}")

    def _export_all_imagewise_summary(self):
        """Exports a consolidated summary across all unique original images with their metric averages."""
        if not self._grouped_records:
            QMessageBox.warning(self, "No Records", "No records available to summarize.")
            return

        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Per-Image Summary",
            str(EXPERIMENTS_DIR / "image_wise_summary.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        try:
            summary_rows = []
            for orig_id, variants in sorted(self._grouped_records.items()):
                if not variants:
                    continue
                dices = [x["dice_coefficient"] for x in variants if x.get("dice_coefficient") is not None]
                ious = [x["iou"] for x in variants if x.get("iou") is not None]
                psnrs = [x["psnr"] for x in variants if x.get("psnr") is not None]
                ssims = [x["ssim"] for x in variants if x.get("ssim") is not None]
                ratios = [x["compression_ratio"] for x in variants if x.get("compression_ratio") is not None]
                times = [x["transmission_time_sec"] for x in variants if x.get("transmission_time_sec") is not None]

                summary_rows.append({
                    "original_image_id": orig_id,
                    "total_trials": len(variants),
                    "max_dice": round(float(np.max(dices)), 4) if dices else None,
                    "mean_dice": round(float(np.mean(dices)), 4) if dices else None,
                    "max_iou": round(float(np.max(ious)), 4) if ious else None,
                    "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
                    "mean_psnr_db": round(float(np.mean(psnrs)), 2) if psnrs else None,
                    "mean_ssim": round(float(np.mean(ssims)), 4) if ssims else None,
                    "mean_comp_ratio": round(float(np.mean(ratios)), 2) if ratios else None,
                    "mean_tx_time_sec": round(float(np.mean(times)), 4) if times else None,
                    "min_tx_time_sec": round(float(np.min(times)), 4) if times else None,
                })

            summary_keys = list(summary_rows[0].keys())
            with open(target_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_keys)
                writer.writeheader()
                writer.writerows(summary_rows)

            QMessageBox.information(
                self,
                "Summary Export Complete",
                f"Exported summary across {len(summary_rows)} original images to:\n{target_path}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export summary: {e}")

    def _export_summary(self):
        """Exports statistical summary across compression qualities."""
        if not self._all_records:
            QMessageBox.warning(self, "No Records", "No records available.")
            return

        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Summary Report",
            str(EXPERIMENTS_DIR / "receiver_summary_report.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        try:
            q_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for r in self._all_records:
                q = r.get("compression_quality")
                if q is not None:
                    q_groups[int(q)].append(r)

            summary_rows = []
            for q, group in sorted(q_groups.items()):
                dices = [r["dice_coefficient"] for r in group if r.get("dice_coefficient") is not None]
                ious = [r["iou"] for r in group if r.get("iou") is not None]
                psnrs = [r["psnr"] for r in group if r.get("psnr") is not None]
                ssims = [r["ssim"] for r in group if r.get("ssim") is not None]
                ratios = [r["compression_ratio"] for r in group if r.get("compression_ratio") is not None]
                times = [r["transmission_time_sec"] for r in group if r.get("transmission_time_sec") is not None]

                summary_rows.append({
                    "quality": f"Q{q}",
                    "samples": len(group),
                    "mean_dice": round(float(np.mean(dices)), 4) if dices else "-",
                    "mean_iou": round(float(np.mean(ious)), 4) if ious else "-",
                    "mean_psnr_db": round(float(np.mean(psnrs)), 2) if psnrs else "-",
                    "mean_ssim": round(float(np.mean(ssims)), 4) if ssims else "-",
                    "mean_comp_ratio": round(float(np.mean(ratios)), 2) if ratios else "-",
                    "mean_tx_time_sec": round(float(np.mean(times)), 4) if times else "-",
                })

            fieldnames = ["quality", "samples", "mean_dice", "mean_iou", "mean_psnr_db", "mean_ssim", "mean_comp_ratio", "mean_tx_time_sec"]
            with open(target_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

            QMessageBox.information(self, "Export Complete", f"Exported summary across {len(q_groups)} quality levels to:\n{target_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export summary: {e}")
