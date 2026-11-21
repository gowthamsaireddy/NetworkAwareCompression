"""Receiver Experiment Results & Image-Wise Evaluation Telemetry View.

Implements:
1. Flat traceable log of all individual experiment trials.
2. Original Image ID-based experiment grouping and tracking across 48+ test images.
3. Multi-objective clinical Pareto trade-off optimization and best-configuration analysis.
4. Export utilities for research reproducibility and paper writing.
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
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

DEFAULT_PARETO_WEIGHTS: Dict[str, float] = {
    "dice": 0.35,        # Clinical lesion boundary retention
    "iou": 0.15,         # Jaccard spatial overlap
    "ssim": 0.20,        # Structural perception preservation
    "psnr": 0.10,        # Signal fidelity (normalized / 50 dB)
    "comp_ratio": 0.10,  # Storage/bandwidth efficiency (normalized / 25x)
    "tx_time": 0.10,     # Low network wire latency (normalized <= 2.0s)
}


def calculate_pareto_score(
    dice: Optional[float],
    iou: Optional[float],
    ssim: Optional[float],
    psnr: Optional[float],
    comp_ratio: Optional[float],
    tx_time: Optional[float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Calculates scientifically calibrated Pareto Trade-Off Score in [0, 100].

    Formula:
      Score = 100 * [
          w_dice * Dice +
          w_iou  * IoU +
          w_ssim * SSIM +
          w_psnr * min(1.0, PSNR / 50.0) +
          w_cr   * min(1.0, CR / 25.0) +
          w_tx   * max(0.0, 1.0 - (T_tx / 2.0))
      ]
    """
    w = weights or DEFAULT_PARETO_WEIGHTS
    norm_dice = float(dice) if dice is not None else 0.0
    norm_iou = float(iou) if iou is not None else 0.0
    norm_ssim = float(ssim) if ssim is not None else 0.0
    norm_psnr = min(max(float(psnr) / 50.0, 0.0), 1.0) if psnr is not None else 0.0
    norm_cr = min(max(float(comp_ratio) / 25.0, 0.0), 1.0) if comp_ratio is not None else 0.0
    norm_time = max(0.0, 1.0 - (float(tx_time) / 2.0)) if tx_time is not None else 0.0

    raw = (
        w.get("dice", 0.35) * norm_dice
        + w.get("iou", 0.15) * norm_iou
        + w.get("ssim", 0.20) * norm_ssim
        + w.get("psnr", 0.10) * norm_psnr
        + w.get("comp_ratio", 0.10) * norm_cr
        + w.get("tx_time", 0.10) * norm_time
    )
    return round(float(np.clip(raw * 100.0, 0.0, 100.0)), 2)


class ParetoWeightsDialog(QDialog):
    """Allows user to customize weights for Pareto multi-objective ranking."""

    def __init__(self, current_weights: Dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Pareto Trade-Off Weights")
        self.resize(460, 340)
        self.weights = dict(current_weights)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel("Adjust Multi-Objective Optimization Weights (%):\nWeights determine the clinical trade-off score ranking.")
        header.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setSpacing(8)

        self.spinners: Dict[str, QDoubleSpinBox] = {}
        labels = [
            ("dice", "Dice Similarity (DSC) Weight %:"),
            ("iou", "Intersection over Union (IoU) Weight %:"),
            ("ssim", "Structural Similarity (SSIM) Weight %:"),
            ("psnr", "PSNR Fidelity Weight %:"),
            ("comp_ratio", "Compression Ratio Weight %:"),
            ("tx_time", "Transmission Latency Weight %:"),
        ]

        for row, (key, lbl_text) in enumerate(labels):
            lbl = QLabel(lbl_text)
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 100.0)
            sp.setSingleStep(5.0)
            sp.setValue(round(self.weights.get(key, 0.1) * 100.0, 1))
            grid.addWidget(lbl, row, 0)
            grid.addWidget(sp, row, 1)
            self.spinners[key] = sp

        layout.addLayout(grid)

        self.btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btn_box.accepted.connect(self._validate_and_accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

    def _validate_and_accept(self):
        total = sum(sp.value() for sp in self.spinners.values())
        if total <= 0:
            QMessageBox.warning(self, "Invalid Weights", "Sum of weights must be greater than 0%.")
            return
        # Normalize weights to sum to 1.0
        self.weights = {k: sp.value() / total for k, sp in self.spinners.items()}
        self.accept()


class ReceiverResultsView(QWidget):
    """Traceable results log, original image-wise tracking, and Pareto best-config analysis."""

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
        "Quality Config",
        "Network Profile",
        "Comp Size",
        "Ratio",
        "Tx Time (s)",
        "Throughput",
        "PSNR (dB)",
        "SSIM",
        "Dice (DSC)",
        "IoU",
        "Pareto Score",
        "Status",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results_manager = ResultsManager()
        self._all_records: List[Dict[str, Any]] = []
        self._grouped_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._pareto_weights = dict(DEFAULT_PARETO_WEIGHTS)
        self._is_dirty = True
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        header = SectionHeader(
            "Receiver Evaluation Records & Image-Wise Telemetry",
            "Traceable records of received ultrasound frames, original image tracking, and multi-objective Pareto optimization",
        )
        layout.addWidget(header)

        # Main tabs: 1) Flat All Trials, 2) Image-Wise Comparison & Best Config Analysis
        self.tabs = QTabWidget()
        self.tabs.setObjectName("ResultsTabWidget")

        self.tab_flat = QWidget()
        self.tab_imagewise = QWidget()

        self._init_flat_tab()
        self._init_imagewise_tab()

        self.tabs.addTab(self.tab_imagewise, "Image-Wise Comparison & Best Configuration Analysis")
        self.tabs.addTab(self.tab_flat, "All Experiment Trials Log")

        layout.addWidget(self.tabs, stretch=1)

        self.load_records(force=True)

    def _init_flat_tab(self):
        """Initializes Tab 1: Flat view of all experiment trials."""
        layout = QVBoxLayout(self.tab_flat)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Action row
        act_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Records")
        self.refresh_btn.setObjectName("SecondaryBtn")
        self.export_btn = QPushButton("Export All Records (CSV)")
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
        self.flat_table.horizontalHeader().setStretchLastSection(True)
        col_widths = [160, 220, 110, 65, 110, 85, 75, 85, 75, 65, 65, 75, 65, 90]
        for col_idx, width in enumerate(col_widths):
            if col_idx < len(self.FLAT_HEADERS):
                self.flat_table.setColumnWidth(col_idx, width)
        self.flat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.flat_table, stretch=1)

        self.refresh_btn.clicked.connect(lambda: self.load_records(force=True))
        self.export_btn.clicked.connect(self._export_csv)
        self.export_summary_btn.clicked.connect(self._export_summary)

    def _init_imagewise_tab(self):
        """Initializes Tab 2: Image-Wise Comparison and Multi-Objective Pareto Analysis."""
        layout = QVBoxLayout(self.tab_imagewise)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Top Image Selection & Strategy Bar
        sel_frame = QFrame()
        sel_frame.setObjectName("CardFrame")
        sel_layout = QHBoxLayout(sel_frame)
        sel_layout.setContentsMargins(12, 8, 12, 8)
        sel_layout.setSpacing(10)

        sel_layout.addWidget(QLabel("Select Original Image:"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(260)
        self.image_combo.currentIndexChanged.connect(self._on_image_selected)
        sel_layout.addWidget(self.image_combo)

        sel_layout.addWidget(QLabel("Ranking Objective:"))
        self.ranking_combo = QComboBox()
        self.ranking_combo.addItems([
            "Recommended: Clinical Pareto Trade-Off Score",
            "Highest Dice Similarity (DSC)",
            "Highest IoU (Jaccard Index)",
            "Highest PSNR (Image Fidelity)",
            "Highest SSIM (Structural Quality)",
            "Highest Compression Ratio (Storage)",
            "Fastest Transmission (Low Latency)",
        ])
        self.ranking_combo.currentIndexChanged.connect(self._populate_image_variants_table)
        sel_layout.addWidget(self.ranking_combo)

        self.btn_weights = QPushButton("Configure Weights...")
        self.btn_weights.setObjectName("SecondaryBtn")
        self.btn_weights.clicked.connect(self._open_weights_dialog)
        sel_layout.addWidget(self.btn_weights)

        sel_layout.addStretch()

        self.btn_export_image = QPushButton("Export This Image (CSV)")
        self.btn_export_image.setObjectName("SecondaryBtn")
        self.btn_export_image.clicked.connect(self._export_selected_image_csv)
        sel_layout.addWidget(self.btn_export_image)

        self.btn_export_all_imagewise = QPushButton("Export All Images Summary")
        self.btn_export_all_imagewise.setObjectName("SecondaryBtn")
        self.btn_export_all_imagewise.clicked.connect(self._export_all_imagewise_summary)
        sel_layout.addWidget(self.btn_export_all_imagewise)

        layout.addWidget(sel_frame)

        # 2. Key Analytics Metric Cards for Selected Original Image
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(10)

        self.card_best_config = MetricBox("Best Overall Configuration", "None")
        self.card_best_score = MetricBox("Pareto Trade-Off Score", "0.0 / 100")
        self.card_best_clinical = MetricBox("Clinical Accuracy (DSC / IoU)", "DSC: - | IoU: -")
        self.card_best_efficiency = MetricBox("Efficiency (Ratio / Tx Time)", "CR: - | Time: -")

        cards_layout.addWidget(self.card_best_config, 2)
        cards_layout.addWidget(self.card_best_score, 1)
        cards_layout.addWidget(self.card_best_clinical, 1)
        cards_layout.addWidget(self.card_best_efficiency, 1)
        layout.addLayout(cards_layout)

        # 3. Transparent Decision Rationale Banner
        self.rationale_frame = QFrame()
        self.rationale_frame.setObjectName("CardFrame")
        r_layout = QVBoxLayout(self.rationale_frame)
        r_layout.setContentsMargins(12, 8, 12, 8)
        r_layout.setSpacing(3)

        self.rationale_formula_lbl = QLabel()
        self.rationale_formula_lbl.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        self.rationale_detail_lbl = QLabel()
        self.rationale_detail_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        r_layout.addWidget(self.rationale_formula_lbl)
        r_layout.addWidget(self.rationale_detail_lbl)
        layout.addWidget(self.rationale_frame)

        # 4. Image Variants Comparison Table
        self.image_table = QTableWidget(0, len(self.VARIANT_HEADERS))
        self.image_table.setHorizontalHeaderLabels(self.VARIANT_HEADERS)
        self.image_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.image_table.horizontalHeader().setStretchLastSection(True)
        var_col_widths = [240, 110, 110, 105, 85, 75, 80, 85, 75, 65, 75, 65, 85, 95]
        for c_idx, width in enumerate(var_col_widths):
            if c_idx < len(self.VARIANT_HEADERS):
                self.image_table.setColumnWidth(c_idx, width)
        self.image_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.image_table, stretch=1)

    def mark_dirty(self):
        """Flags records as modified so the tables refresh when viewed."""
        self._is_dirty = True

    def load_records(self, force: bool = False, *args, **kwargs):
        """Loads records from database with smart caching."""
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
            self.image_combo.addItem(f"{orig_id} ({len(variants)} variants)", userData=orig_id)

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

    def _open_weights_dialog(self):
        dlg = ParetoWeightsDialog(self._pareto_weights, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._pareto_weights = dlg.weights
            self._populate_image_variants_table()

    def _populate_image_variants_table(self):
        """Populates the variant comparison table and computes optimal Pareto trade-off."""
        if self.image_combo.currentIndex() < 0:
            return

        selected_orig_id = self.image_combo.currentData()
        if not selected_orig_id:
            return

        variants = list(self._grouped_records.get(selected_orig_id, []))
        if not variants:
            self.image_table.setRowCount(0)
            self.card_best_config.set_value("No experiments")
            self.card_best_score.set_value("-")
            self.card_best_clinical.set_value("-")
            self.card_best_efficiency.set_value("-")
            return

        # Compute Pareto score for each variant
        for v in variants:
            v["_pareto_score"] = calculate_pareto_score(
                dice=v.get("dice_coefficient"),
                iou=v.get("iou"),
                ssim=v.get("ssim"),
                psnr=v.get("psnr"),
                comp_ratio=v.get("compression_ratio"),
                tx_time=v.get("transmission_time_sec"),
                weights=self._pareto_weights,
            )

        # Sort variants based on selected strategy
        strat = self.ranking_combo.currentText()
        if "Highest Dice" in strat:
            variants.sort(key=lambda x: (x.get("dice_coefficient") or 0.0), reverse=True)
        elif "Highest IoU" in strat:
            variants.sort(key=lambda x: (x.get("iou") or 0.0), reverse=True)
        elif "Highest PSNR" in strat:
            variants.sort(key=lambda x: (x.get("psnr") or 0.0), reverse=True)
        elif "Highest SSIM" in strat:
            variants.sort(key=lambda x: (x.get("ssim") or 0.0), reverse=True)
        elif "Highest Compression Ratio" in strat:
            variants.sort(key=lambda x: (x.get("compression_ratio") or 0.0), reverse=True)
        elif "Fastest Transmission" in strat:
            variants.sort(key=lambda x: (x.get("transmission_time_sec") or 9999.0))
        else:  # Recommended: Clinical Pareto Trade-Off Score
            variants.sort(key=lambda x: x.get("_pareto_score", 0.0), reverse=True)

        best_variant = variants[0]

        # Update Analytics Cards
        best_cfg_name = f"{best_variant.get('compression_method', 'NORMAL')} Q{best_variant.get('compression_quality')} ({best_variant.get('network_profile', 'Unshaped')})"
        best_score = best_variant.get("_pareto_score", 0.0)
        best_dice = best_variant.get("dice_coefficient")
        best_iou = best_variant.get("iou")
        best_cr = best_variant.get("compression_ratio")
        best_tx = best_variant.get("transmission_time_sec")

        self.card_best_config.set_value(best_cfg_name)
        self.card_best_score.set_value(f"{best_score:.1f} / 100")
        dice_str = f"{best_dice:.4f}" if best_dice is not None else "-"
        iou_str = f"{best_iou:.4f}" if best_iou is not None else "-"
        self.card_best_clinical.set_value(f"DSC: {dice_str} | IoU: {iou_str}")

        cr_str = f"{best_cr:.1f}×" if best_cr is not None else "-"
        tx_str = f"{best_tx:.4f}s" if best_tx is not None else "-"
        self.card_best_efficiency.set_value(f"CR: {cr_str} | Time: {tx_str}")

        # Update Rationale Banner
        w = self._pareto_weights
        formula_txt = (
            f"Multi-Objective Optimization Formula: Score = 100 × ["
            f"{w.get('dice', 0.35):.2f}·Dice + {w.get('iou', 0.15):.2f}·IoU + "
            f"{w.get('ssim', 0.20):.2f}·SSIM + {w.get('psnr', 0.10):.2f}·(PSNR/50) + "
            f"{w.get('comp_ratio', 0.10):.2f}·(CR/25) + {w.get('tx_time', 0.10):.2f}·(1-T/2)]"
        )
        self.rationale_formula_lbl.setText(formula_txt)

        rationale_detail = (
            f"Optimal trade-off for {selected_orig_id}: '{best_cfg_name}' achieved top score of {best_score:.1f}/100. "
            f"Preserved segmentation accuracy (DSC: {dice_str}) with {cr_str} size reduction in {tx_str} transmission time."
        )
        self.rationale_detail_lbl.setText(rationale_detail)

        # Populate Table
        self.image_table.setUpdatesEnabled(False)
        self.image_table.blockSignals(True)
        try:
            self.image_table.setRowCount(len(variants))
            for row_idx, r in enumerate(variants):
                is_best = (row_idx == 0)

                # 0: Experiment ID
                exp_prefix = "★ " if is_best else ""
                item_exp = QTableWidgetItem(f"{exp_prefix}{r.get('experiment_id', '')}")
                item_exp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if is_best:
                    item_exp.setForeground(QColor("#fbbf24"))  # Amber star
                    item_exp.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
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

                # 7: Throughput
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

                # 12: Pareto Score
                score = r.get("_pareto_score", 0.0)
                item_score = QTableWidgetItem(f"{score:.1f}")
                item_score.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if is_best:
                    item_score.setForeground(QColor("#34d399"))
                    item_score.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                self.image_table.setItem(row_idx, 12, item_score)

                # 13: Status
                stat = r.get("experiment_status") or STATUS_COMPLETED
                item_stat = QTableWidgetItem(stat)
                item_stat.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if stat == STATUS_COMPLETED:
                    item_stat.setForeground(QColor("#10b981"))
                elif stat == STATUS_INCOMPLETE:
                    item_stat.setForeground(QColor("#f59e0b"))
                else:
                    item_stat.setForeground(QColor("#ef4444"))
                self.image_table.setItem(row_idx, 13, item_stat)
        finally:
            self.image_table.setUpdatesEnabled(True)
            self.image_table.blockSignals(False)

    def _apply_flat_filters(self):
        """Filters in-memory records and populates Tab 1 table cells smoothly."""
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

                item_exp = QTableWidgetItem(str(r.get("experiment_id") or ""))
                item_exp.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
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
                "iou", "_pareto_score", "experiment_status"
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
        """Exports a consolidated summary across all unique original images with their optimal Pareto configurations."""
        if not self._grouped_records:
            QMessageBox.warning(self, "No Records", "No records available to summarize.")
            return

        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export All Images Optimization Summary",
            str(EXPERIMENTS_DIR / "image_wise_best_configurations_summary.csv"),
            "CSV Files (*.csv)",
        )
        if not target_path:
            return

        try:
            summary_rows = []
            for orig_id, variants in sorted(self._grouped_records.items()):
                if not variants:
                    continue
                # Score variants
                for v in variants:
                    v["_pareto_score"] = calculate_pareto_score(
                        dice=v.get("dice_coefficient"),
                        iou=v.get("iou"),
                        ssim=v.get("ssim"),
                        psnr=v.get("psnr"),
                        comp_ratio=v.get("compression_ratio"),
                        tx_time=v.get("transmission_time_sec"),
                        weights=self._pareto_weights,
                    )
                best_v = max(variants, key=lambda x: x.get("_pareto_score", 0.0))
                all_dices = [x["dice_coefficient"] for x in variants if x.get("dice_coefficient") is not None]
                all_times = [x["transmission_time_sec"] for x in variants if x.get("transmission_time_sec") is not None]

                summary_rows.append({
                    "original_image_id": orig_id,
                    "total_variants_tested": len(variants),
                    "best_config_id": best_v.get("experiment_id"),
                    "best_method": best_v.get("compression_method"),
                    "best_quality": best_v.get("compression_quality"),
                    "best_network_profile": best_v.get("network_profile"),
                    "best_pareto_score": best_v.get("_pareto_score"),
                    "best_dice": best_v.get("dice_coefficient"),
                    "best_iou": best_v.get("iou"),
                    "best_psnr": best_v.get("psnr"),
                    "best_ssim": best_v.get("ssim"),
                    "best_compression_ratio": best_v.get("compression_ratio"),
                    "best_tx_time_sec": best_v.get("transmission_time_sec"),
                    "mean_dice_across_variants": round(float(np.mean(all_dices)), 4) if all_dices else None,
                    "min_tx_time_sec": round(float(np.min(all_times)), 4) if all_times else None,
                })

            summary_keys = list(summary_rows[0].keys())
            with open(target_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=summary_keys)
                writer.writeheader()
                writer.writerows(summary_rows)

            QMessageBox.information(
                self,
                "Summary Export Complete",
                f"Exported optimization summary across {len(summary_rows)} original images to:\n{target_path}",
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
