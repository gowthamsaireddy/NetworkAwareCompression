"""Sender Compression View: Dual-Mode Selection (Normal JPEG vs. ML-Based Region-Adaptive).

Provides two explicitly separate, independent compression modes:
- Option A: Normal Compression (Standard uniform JPEG at Q30, Q50, Q70, Q90). Default.
- Option B: ML-Based Region-Adaptive Compression (Preserves lesion + configurable margin at high quality,
  aggressively compresses background tissue, with Gaussian alpha-feathering along the margin boundary).
"""

from io import BytesIO
import json
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from common.logging_config import setup_logger
from sender.compression import CompressionResult, ImageCompressor
from sender.dataset_manager import DatasetItem
from sender.region_adaptive_compressor import RegionAdaptiveCompressor, RegionAdaptiveResult
from shared.constants import (
    COMPRESSION_QUALITIES,
    PROTOCOL_VERSION,
    STATUS_NOT_CALCULATED,
)
from shared.metadata import TransmissionMetadata
from shared.protocol import compute_sha256
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader

logger = setup_logger("CompressionView")


class SenderCompressionView(QWidget):
    """Dual-mode compression workbench supporting Standard JPEG and ML Region-Adaptive compression."""

    compression_completed = pyqtSignal(object, bytes, float, float)  # Emits (metadata, bytes, psnr, ssim)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_item: Optional[DatasetItem] = None
        self.current_image_path: Optional[Path] = None
        self.current_result: Optional[CompressionResult] = None
        self.current_adaptive_result: Optional[RegionAdaptiveResult] = None

        # Region adaptive compressor instance
        self.adaptive_compressor = RegionAdaptiveCompressor()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Image Compression Workbench",
            "Select compression methodology: Option A (Standard JPEG) or Option B (ML-Based Region-Adaptive)",
        )
        layout.addWidget(header)

        # -------------------------------------------------------------
        # Mode Selection Header Frame (Option A vs. Option B)
        # -------------------------------------------------------------
        mode_frame = QFrame()
        mode_frame.setObjectName("CardFrame")
        mode_frame_layout = QVBoxLayout(mode_frame)
        mode_frame_layout.setContentsMargins(16, 14, 16, 14)
        mode_frame_layout.setSpacing(10)

        mode_title = QLabel("Select Active Compression Mode:")
        mode_title.setStyleSheet("font-weight: 700; font-size: 13px; color: #f8fafc;")
        mode_frame_layout.addWidget(mode_title)

        mode_radios_layout = QHBoxLayout()
        self.mode_button_group = QButtonGroup(self)

        self.radio_normal = QRadioButton("Option A: Normal Compression (Standard Uniform JPEG)")
        self.radio_normal.setObjectName("RadioNormal")
        self.radio_normal.setChecked(True)  # Default option to preserve working behavior

        self.radio_adaptive = QRadioButton("Option B: ML-Based Region-Adaptive Compression")
        self.radio_adaptive.setObjectName("RadioAdaptive")

        self.mode_button_group.addButton(self.radio_normal, 1)
        self.mode_button_group.addButton(self.radio_adaptive, 2)

        mode_radios_layout.addWidget(self.radio_normal)
        mode_radios_layout.addWidget(self.radio_adaptive)
        mode_radios_layout.addStretch()

        self.browse_single_btn = QPushButton("Select Image File")
        self.browse_single_btn.setObjectName("SecondaryBtn")
        mode_radios_layout.addWidget(self.browse_single_btn)

        mode_frame_layout.addLayout(mode_radios_layout)
        layout.addWidget(mode_frame)

        # -------------------------------------------------------------
        # Settings Panels (Option A vs Option B)
        # -------------------------------------------------------------
        self.settings_container = QFrame()
        self.settings_container.setObjectName("CardFrame")
        self.settings_layout = QVBoxLayout(self.settings_container)
        self.settings_layout.setContentsMargins(16, 12, 16, 12)

        # --- Sub-panel A: Normal Compression ---
        self.panel_normal = QWidget()
        layout_normal = QHBoxLayout(self.panel_normal)
        layout_normal.setContentsMargins(0, 0, 0, 0)
        layout_normal.setSpacing(12)

        layout_normal.addWidget(QLabel("Uniform JPEG Quality:"))

        self.normal_btn_group = QButtonGroup(self)
        self.radio_q30 = QRadioButton("Q30 (High Compression)")
        self.radio_q50 = QRadioButton("Q50 (Medium-High)")
        self.radio_q70 = QRadioButton("Q70 (Balanced)")
        self.radio_q90 = QRadioButton("Q90 (High Quality)")
        self.radio_q70.setChecked(True)

        self.normal_btn_group.addButton(self.radio_q30, 30)
        self.normal_btn_group.addButton(self.radio_q50, 50)
        self.normal_btn_group.addButton(self.radio_q70, 70)
        self.normal_btn_group.addButton(self.radio_q90, 90)

        layout_normal.addWidget(self.radio_q30)
        layout_normal.addWidget(self.radio_q50)
        layout_normal.addWidget(self.radio_q70)
        layout_normal.addWidget(self.radio_q90)
        layout_normal.addStretch()

        self.settings_layout.addWidget(self.panel_normal)

        # --- Sub-panel B: Region-Adaptive Compression ---
        self.panel_adaptive = QWidget()
        layout_adaptive = QHBoxLayout(self.panel_adaptive)
        layout_adaptive.setContentsMargins(0, 0, 0, 0)
        layout_adaptive.setSpacing(14)

        # Lesion Margin (px)
        layout_adaptive.addWidget(QLabel("Lesion Margin:"))
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(5, 50)
        self.margin_spin.setValue(15)
        self.margin_spin.setSuffix(" px")
        self.margin_spin.setToolTip("Dilation radius in pixels expanding preserved lesion border")
        layout_adaptive.addWidget(self.margin_spin)

        # ROI Quality
        layout_adaptive.addWidget(QLabel("ROI Quality:"))
        self.roi_q_combo = QComboBox()
        self.roi_q_combo.addItems(["85", "90", "95", "100"])
        self.roi_q_combo.setCurrentText("90")
        self.roi_q_combo.setToolTip("JPEG encoding quality for lesion and dilated margin")
        layout_adaptive.addWidget(self.roi_q_combo)

        # Background Quality
        layout_adaptive.addWidget(QLabel("Background Quality:"))
        self.bg_q_combo = QComboBox()
        self.bg_q_combo.addItems(["15", "20", "25", "30", "35"])
        self.bg_q_combo.setCurrentText("20")
        self.bg_q_combo.setToolTip("Aggressive compression factor for non-critical tissue")
        layout_adaptive.addWidget(self.bg_q_combo)

        # ROI Localization Mode
        layout_adaptive.addWidget(QLabel("ROI Locator:"))
        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItems(["ML Model (roi_detector_v1)", "Ground-Truth Mask (Upper Bound)"])
        self.roi_mode_combo.setToolTip("Method used to detect or delineate lesion region")
        layout_adaptive.addWidget(self.roi_mode_combo)

        # Model Version Tag
        lbl_model_ver = QLabel("Model: roi_detector_v1")
        lbl_model_ver.setStyleSheet(
            "background: #1e293b; color: #38bdf8; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;"
        )
        layout_adaptive.addWidget(lbl_model_ver)
        layout_adaptive.addStretch()

        self.settings_layout.addWidget(self.panel_adaptive)
        self.panel_adaptive.setVisible(False)  # Normal is default

        # Execute Button Row
        exec_row = QHBoxLayout()
        self.chk_show_overlay = QCheckBox("Show Preserved ROI Boundary Contour")
        self.chk_show_overlay.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.setVisible(False)

        self.compress_btn = QPushButton("Compress Current Image")
        self.compress_btn.setStyleSheet(
            "background: #0284c7; color: white; font-weight: 700; padding: 8px 18px; border-radius: 6px;"
        )

        exec_row.addWidget(self.chk_show_overlay)
        exec_row.addStretch()
        exec_row.addWidget(self.compress_btn)
        self.settings_layout.addLayout(exec_row)

        layout.addWidget(self.settings_container)

        # -------------------------------------------------------------
        # Two Image Previews: Original vs Compressed
        # -------------------------------------------------------------
        previews_layout = QHBoxLayout()
        self.original_preview = ImagePreviewBox("Original Uncompressed Image", "No image selected")
        self.compressed_preview = ImagePreviewBox("Compressed Image Output", "Not compressed yet")
        previews_layout.addWidget(self.original_preview)
        previews_layout.addWidget(self.compressed_preview)
        layout.addLayout(previews_layout, stretch=1)

        # -------------------------------------------------------------
        # Metrics Display Grid
        # -------------------------------------------------------------
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        self.card_orig_size = MetricBox("Original Size", STATUS_NOT_CALCULATED)
        self.card_comp_size = MetricBox("Compressed Size", STATUS_NOT_CALCULATED)
        self.card_ratio = MetricBox("Compression Ratio", STATUS_NOT_CALCULATED)
        self.card_reduction = MetricBox("Size Reduction %", STATUS_NOT_CALCULATED)
        self.card_psnr = MetricBox("Full Image PSNR", STATUS_NOT_CALCULATED)
        self.card_roi_psnr = MetricBox("Preserved ROI PSNR", "Uniform (N/A)")
        self.card_ssim = MetricBox("SSIM", STATUS_NOT_CALCULATED)

        metrics_grid.addWidget(self.card_orig_size, 0, 0)
        metrics_grid.addWidget(self.card_comp_size, 0, 1)
        metrics_grid.addWidget(self.card_ratio, 0, 2)
        metrics_grid.addWidget(self.card_reduction, 0, 3)
        metrics_grid.addWidget(self.card_psnr, 1, 0)
        metrics_grid.addWidget(self.card_roi_psnr, 1, 1)
        metrics_grid.addWidget(self.card_ssim, 1, 2)

        layout.addLayout(metrics_grid)

        # -------------------------------------------------------------
        # Connect Actions
        # -------------------------------------------------------------
        self.radio_normal.toggled.connect(self._on_mode_toggled)
        self.radio_adaptive.toggled.connect(self._on_mode_toggled)
        self.browse_single_btn.clicked.connect(self._browse_single_image)
        self.compress_btn.clicked.connect(self._handle_compress)
        self.chk_show_overlay.toggled.connect(self._update_adaptive_preview_overlay)

    def _on_mode_toggled(self):
        """Toggles parameter visibility when switching between Option A and Option B."""
        is_normal = self.radio_normal.isChecked()
        self.panel_normal.setVisible(is_normal)
        self.panel_adaptive.setVisible(not is_normal)
        self.chk_show_overlay.setVisible(not is_normal)
        if is_normal:
            self.card_roi_psnr.set_value("Uniform (N/A)")
        else:
            self.card_roi_psnr.set_value(STATUS_NOT_CALCULATED)
        self._reset_metrics()
        self.compressed_preview.clear_image("Ready to compress with " + ("Normal JPEG" if is_normal else "Region-Adaptive"))

    def load_dataset_item(self, item: DatasetItem):
        """Loads a DatasetItem selected from DatasetView."""
        self.current_item = item
        self.current_image_path = item.image_path
        if item.image_path.exists():
            pix = QPixmap(str(item.image_path))
            self.original_preview.set_pixmap(
                pix,
                caption=f"ID: {item.image_id} | Category: {item.category.upper()} | {item.width}×{item.height} | {item.file_size_bytes:,} B",
            )
            self.compressed_preview.clear_image("Not compressed yet")
            self._reset_metrics()

    def _browse_single_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Ultrasound Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            p = Path(file_path)
            if "_mask" in p.stem.lower():
                QMessageBox.warning(
                    self,
                    "Mask Selected",
                    "The selected file appears to be a ground-truth mask. Ground-truth masks must NEVER be compressed.",
                )
                return

            with Image.open(p) as img:
                w, h = img.size

            category = "benign" if "benign" in p.stem.lower() else ("malignant" if "malignant" in p.stem.lower() else "normal")
            item = DatasetItem(
                image_id=f"custom/{p.stem}",
                image_path=p,
                category=category,
                mask_paths=[],
                file_size_bytes=p.stat().st_size,
                width=w,
                height=h,
            )
            self.load_dataset_item(item)

    def _reset_metrics(self):
        self.card_orig_size.set_value(STATUS_NOT_CALCULATED)
        self.card_comp_size.set_value(STATUS_NOT_CALCULATED)
        self.card_ratio.set_value(STATUS_NOT_CALCULATED)
        self.card_reduction.set_value(STATUS_NOT_CALCULATED)
        self.card_psnr.set_value(STATUS_NOT_CALCULATED)
        if not self.radio_normal.isChecked():
            self.card_roi_psnr.set_value(STATUS_NOT_CALCULATED)
        else:
            self.card_roi_psnr.set_value("Uniform (N/A)")
        self.card_ssim.set_value(STATUS_NOT_CALCULATED)

    def _update_adaptive_preview_overlay(self):
        """Redraws preview with or without contour overlay when toggle changed."""
        res = self.current_adaptive_result
        if res is None:
            return

        display_rgb = res.fused_array.copy()
        if self.chk_show_overlay.isChecked() and np.any(res.roi_mask > 0):
            # Draw crisp cyan contour of dilated margin
            contours, _ = cv2.findContours(res.roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_rgb, contours, -1, (56, 189, 248), 2, lineType=cv2.LINE_AA)

        h, w, ch = display_rgb.shape
        q_img = QImage(display_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        comp_pix = QPixmap.fromImage(q_img)
        caption = (
            f"Region-Adaptive (Margin {res.margin_px}px, ROI Q{res.roi_quality}, BG Q{res.bg_quality}) | "
            f"{res.compressed_size_bytes:,} B ({res.compression_ratio:.2f}:1)"
        )
        self.compressed_preview.set_pixmap(comp_pix, caption=caption)

    def _handle_compress(self):
        """Executes compression based on the selected mode: Option A or Option B."""
        if not self.current_image_path or not self.current_image_path.exists():
            QMessageBox.warning(self, "No Image", "Please select an ultrasound image from Dataset Management first.")
            return

        is_normal = self.radio_normal.isChecked()
        img_id = self.current_item.image_id if self.current_item else f"custom/{self.current_image_path.stem}"
        category = self.current_item.category if self.current_item else "benign"

        if is_normal:
            # -------------------------------------------------------------
            # OPTION A: NORMAL COMPRESSION
            # -------------------------------------------------------------
            quality = self.normal_btn_group.checkedId()
            if quality <= 0:
                quality = 70

            try:
                res = ImageCompressor.compress_image(self.current_image_path, quality)
                self.current_result = res
                self.current_adaptive_result = None

                h, w, ch = res.decoded_array.shape
                q_img = QImage(res.decoded_array.data, w, h, ch * w, QImage.Format.Format_RGB888)
                comp_pix = QPixmap.fromImage(q_img)
                caption = (
                    f"Normal JPEG Q{res.quality} | {res.width}×{res.height} | {res.compressed_size_bytes:,} bytes "
                    f"({res.compression_ratio:.2f}:1)"
                )
                self.compressed_preview.set_pixmap(comp_pix, caption=caption)

                # Update genuine measured metric cards
                self.card_orig_size.set_value(f"{res.original_size_bytes:,} B")
                self.card_comp_size.set_value(f"{res.compressed_size_bytes:,} B")
                self.card_ratio.set_value(f"{res.compression_ratio:.2f} : 1")
                self.card_reduction.set_value(f"{res.size_reduction_pct:.2f}%")
                self.card_psnr.set_value(f"{res.psnr:.2f} dB" if res.psnr != float("inf") else "inf")
                self.card_roi_psnr.set_value("Uniform (N/A)")
                self.card_ssim.set_value(f"{res.ssim:.4f}")

                sha = compute_sha256(res.compressed_bytes)
                metadata = TransmissionMetadata(
                    protocol_version=PROTOCOL_VERSION,
                    experiment_id=f"EXP_{int(time.time())}",
                    image_id=img_id,
                    compression_quality=quality,
                    image_width=res.width,
                    image_height=res.height,
                    image_format="JPEG",
                    original_file_size=res.original_size_bytes,
                    compressed_file_size=res.compressed_size_bytes,
                    image_byte_length=res.compressed_size_bytes,
                    sha256_checksum=sha,
                    compression_mode="NORMAL",
                    region_adaptive_margin=0,
                    compression_model_version=None,
                    image_category=category,
                )
                self.compression_completed.emit(metadata, res.compressed_bytes, res.psnr, res.ssim)

            except Exception as e:
                QMessageBox.critical(self, "Compression Error", f"Failed normal compression: {e}")

        else:
            # -------------------------------------------------------------
            # OPTION B: ML-BASED REGION-ADAPTIVE COMPRESSION
            # -------------------------------------------------------------
            margin_px = self.margin_spin.value()
            roi_quality = int(self.roi_q_combo.currentText())
            bg_quality = int(self.bg_q_combo.currentText())
            selected_roi_mode_text = self.roi_mode_combo.currentText()
            roi_mode = "GT_MASK" if "Ground-Truth" in selected_roi_mode_text else "ML_MODEL"

            gt_mask = None
            if self.current_item and self.current_item.mask_paths:
                for mp in self.current_item.mask_paths:
                    if mp.exists():
                        m_arr = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                        if m_arr is not None:
                            gt_mask = m_arr
                            break

            try:
                res_adapt = self.adaptive_compressor.compress_image(
                    image_source=self.current_image_path,
                    margin_px=margin_px,
                    roi_quality=roi_quality,
                    bg_quality=bg_quality,
                    roi_mode=roi_mode,
                    gt_mask=gt_mask,
                )
                self.current_adaptive_result = res_adapt
                self.current_result = None

                self._update_adaptive_preview_overlay()

                # Update genuine measured metric cards
                self.card_orig_size.set_value(f"{res_adapt.original_size_bytes:,} B")
                self.card_comp_size.set_value(f"{res_adapt.compressed_size_bytes:,} B")
                self.card_ratio.set_value(f"{res_adapt.compression_ratio:.2f} : 1")
                self.card_reduction.set_value(f"{res_adapt.size_reduction_pct:.2f}%")
                self.card_psnr.set_value(f"{res_adapt.psnr:.2f} dB")
                self.card_roi_psnr.set_value(f"{res_adapt.roi_psnr:.2f} dB")
                self.card_ssim.set_value(f"{res_adapt.ssim:.4f}")

                sha = compute_sha256(res_adapt.compressed_bytes)
                metadata = TransmissionMetadata(
                    protocol_version=PROTOCOL_VERSION,
                    experiment_id=f"EXP_ADAPT_{int(time.time())}",
                    image_id=img_id,
                    compression_quality=roi_quality,
                    image_width=res_adapt.width,
                    image_height=res_adapt.height,
                    image_format="JPEG",
                    original_file_size=res_adapt.original_size_bytes,
                    compressed_file_size=res_adapt.compressed_size_bytes,
                    image_byte_length=res_adapt.compressed_size_bytes,
                    sha256_checksum=sha,
                    compression_mode="REGION_ADAPTIVE",
                    region_adaptive_margin=margin_px,
                    compression_model_version=res_adapt.model_version,
                    image_category=category,
                )
                self.compression_completed.emit(
                    metadata, res_adapt.compressed_bytes, res_adapt.psnr, res_adapt.ssim
                )

            except Exception as e:
                QMessageBox.critical(self, "Compression Error", f"Failed region-adaptive compression: {e}")
