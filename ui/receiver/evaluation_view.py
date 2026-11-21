"""Receiver Segmentation Evaluation View (Dice & IoU)."""

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from common.config import DATA_DIR
from receiver.segmentation import SegmentationResult
from receiver.segmentation_metrics import (
    create_comparative_overlay,
    evaluate_segmentation,
    get_canonical_busi_id,
    locate_ground_truth_mask,
    locate_original_uncompressed,
)
from shared.constants import STATUS_NOT_CALCULATED
from shared.metadata import TransmissionMetadata
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class ReceiverEvaluationView(QWidget):
    """Comparative evaluation against verified ground-truth masks (Dice and IoU metrics)."""

    evaluation_completed = pyqtSignal(object, object, object, object, object)  # Emits (metadata, dice, iou, psnr, ssim)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.search_roots: List[Path] = [
            DATA_DIR / "BUSI",
            DATA_DIR / "ground_truth",
            DATA_DIR,
        ]
        self.current_result: Optional[SegmentationResult] = None
        self.current_metadata: Optional[TransmissionMetadata] = None
        self.current_comp_path: Optional[Path] = None
        self.current_orig_path: Optional[Path] = None
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Wrap entire evaluation view in a scroll area to maintain structure on all displays
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        content = QWidget()
        content.setObjectName("EvaluationScrollContent")
        layout = QVBoxLayout(content)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        header = SectionHeader(
            "Evaluation & Ground-Truth Comparison",
            "Evaluate segmentation accuracy against verified ground-truth masks using genuine Dice and IoU metrics",
        )
        layout.addWidget(header)

        # Action bar
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("CardFrame")
        ctrl_layout = QHBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)

        self.select_gt_folder_btn = QPushButton("Set Ground-Truth Directory")
        self.select_gt_folder_btn.setObjectName("SecondaryBtn")
        self.eval_btn = QPushButton("Re-evaluate Active Image")

        ctrl_layout.addWidget(self.select_gt_folder_btn)
        ctrl_layout.addWidget(self.eval_btn)
        ctrl_layout.addStretch()

        self.gt_folder_lbl = QLabel(f"GT Search Root: {self.search_roots[0]}")
        self.gt_folder_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        ctrl_layout.addWidget(self.gt_folder_lbl)

        layout.addWidget(ctrl_frame)

        # SECTION 1: Transmitted Ultrasound Comparison (Balanced 2-Card Row)
        sec1_header = QLabel("TRANSMITTED ULTRASOUND COMPARISON")
        sec1_header.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;")
        layout.addWidget(sec1_header)

        trans_grid = QGridLayout()
        trans_grid.setSpacing(12)

        self.orig_box = ImagePreviewBox("Original Uncompressed (Source)", "No image")
        self.comp_box = ImagePreviewBox("Received Compressed (Channel Output)", "No image")

        trans_grid.addWidget(self.orig_box, 0, 0)
        trans_grid.addWidget(self.comp_box, 0, 1)
        layout.addLayout(trans_grid)

        # SECTION 2: Clinical Lesion Segmentation & Verification (Balanced 3-Card Row)
        sec2_header = QLabel("CLINICAL SEGMENTATION & GROUND-TRUTH VERIFICATION")
        sec2_header.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; margin-top: 8px;")
        layout.addWidget(sec2_header)

        seg_grid = QGridLayout()
        seg_grid.setSpacing(12)

        self.gt_box = ImagePreviewBox("Ground-Truth Mask (Verified)", "No ground truth")
        self.pred_box = ImagePreviewBox("Predicted Mask (MGDA U-Net)", "No prediction")
        self.overlay_box = ImagePreviewBox("Comparative Overlay (Yellow=TP, Red=FP, Green=FN)", "No overlay")

        seg_grid.addWidget(self.gt_box, 0, 0)
        seg_grid.addWidget(self.pred_box, 0, 1)
        seg_grid.addWidget(self.overlay_box, 0, 2)
        layout.addLayout(seg_grid)

        # SECTION 3: Quantitative Evaluation Metrics
        sec3_header = QLabel("QUANTITATIVE ACCURACY & COMPRESSION METRICS")
        sec3_header.setStyleSheet("color: #38bdf8; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; margin-top: 8px;")
        layout.addWidget(sec3_header)

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        self.card_dice = MetricBox("Dice Similarity Coefficient (DSC)", STATUS_NOT_CALCULATED)
        self.card_iou = MetricBox("Intersection over Union (IoU)", STATUS_NOT_CALCULATED)
        self.card_gt_match = MetricBox("Ground Truth Status", "Unverified")
        self.card_quality = MetricBox("Compression Quality", STATUS_NOT_CALCULATED)

        metrics_grid.addWidget(self.card_dice, 0, 0)
        metrics_grid.addWidget(self.card_iou, 0, 1)
        metrics_grid.addWidget(self.card_gt_match, 0, 2)
        metrics_grid.addWidget(self.card_quality, 0, 3)

        layout.addLayout(metrics_grid)

        scroll.setWidget(content)
        root_layout.addWidget(scroll)

        # Connect actions
        self.select_gt_folder_btn.clicked.connect(self._select_gt_folder)
        self.eval_btn.clicked.connect(self._re_evaluate)

    def _select_gt_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Ground-Truth Root Directory")
        if folder:
            p = Path(folder)
            if p not in self.search_roots:
                self.search_roots.insert(0, p)
            self.gt_folder_lbl.setText(f"GT Search Root: {p}")
            self._re_evaluate()

    def load_evaluation_data(
        self,
        result: SegmentationResult,
        metadata: TransmissionMetadata,
        comp_image_path: Optional[Path] = None,
    ):
        """Loads inference result and conducts comparative evaluation against ground truth."""
        self.current_result = result
        self.current_metadata = metadata
        self.current_comp_path = comp_image_path

        # 1. Received Compressed Preview
        if comp_image_path and comp_image_path.exists():
            pix = QPixmap(str(comp_image_path))
            self.comp_box.set_pixmap(
                pix,
                caption=f"Q{metadata.compression_quality} | {metadata.image_width}×{metadata.image_height}",
            )
        else:
            # Fallback to processed image
            h, w = result.processed_image.shape
            q_comp = QImage(result.processed_image.data, w, h, w, QImage.Format.Format_Grayscale8)
            self.comp_box.set_pixmap(QPixmap.fromImage(q_comp), caption=f"Q{metadata.compression_quality} (256×256)")

        # 2. Predicted Mask Preview
        h, w = result.predicted_mask.shape
        q_pred = QImage(result.predicted_mask.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.pred_box.set_pixmap(QPixmap.fromImage(q_pred), caption="256 × 256 Binary Prediction")

        # 3. Locate Ground-Truth Mask(s) using robust canonical matching
        gt_mask, mask_id, mask_paths = locate_ground_truth_mask(metadata.image_id, self.search_roots)

        # 4. Search for Original Uncompressed Image
        self._load_original_uncompressed(metadata.image_id)

        # Calculate PSNR and SSIM if original image was found
        psnr_val, ssim_val = self._calculate_fidelity(self.current_orig_path, comp_image_path)

        # 5. Evaluate Metrics & Render Comparative Overlay
        if gt_mask is not None:
            # Ground truth located
            gh, gw = gt_mask.shape
            q_gt = QImage(gt_mask.data, gw, gh, gw, QImage.Format.Format_Grayscale8)
            caption_str = f"256 × 256 Ground Truth ({len(mask_paths)} mask{'s' if len(mask_paths) > 1 else ''})"
            self.gt_box.set_pixmap(QPixmap.fromImage(q_gt), caption=caption_str)

            dice, iou, status = evaluate_segmentation(gt_mask, result.predicted_mask)
            self.card_dice.set_value(f"{dice:.4f}" if dice is not None else STATUS_NOT_CALCULATED, color="#10b981")
            self.card_iou.set_value(f"{iou:.4f}" if iou is not None else STATUS_NOT_CALCULATED, color="#38bdf8")
            self.card_gt_match.set_value(f"GT Matched ({len(mask_paths)})", subtitle=f"Source: {mask_paths[0].name}", color="#38bdf8")

            # Comparative overlay
            comp_overlay = create_comparative_overlay(result.processed_image, gt_mask, result.predicted_mask)
            oh, ow, oc = comp_overlay.shape
            q_comp_overlay = QImage(
                comp_overlay.data,
                ow,
                oh,
                oc * ow,
                QImage.Format.Format_RGB888,
            )
            self.overlay_box.set_pixmap(
                QPixmap.fromImage(q_comp_overlay),
                caption=f"Dice: {dice:.4f} | IoU: {iou:.4f}",
            )
            self.evaluation_completed.emit(metadata, dice, iou, psnr_val, ssim_val)
        else:
            # Zero Dummy-Data policy: Do not calculate or fabricate Dice/IoU
            self.gt_box.clear_image("Ground truth unavailable")
            self.card_dice.set_value(STATUS_NOT_CALCULATED, color="#f59e0b")
            self.card_iou.set_value(STATUS_NOT_CALCULATED, color="#f59e0b")
            self.card_gt_match.set_value("No GT Mask Found", subtitle="Zero dummy data policy enforced", color="#ef4444")

            # Overlay prediction only
            single_overlay = create_comparative_overlay(result.processed_image, None, result.predicted_mask)
            oh, ow, oc = single_overlay.shape
            q_single = QImage(
                single_overlay.data,
                ow,
                oh,
                oc * ow,
                QImage.Format.Format_RGB888,
            )
            self.overlay_box.set_pixmap(QPixmap.fromImage(q_single), caption="Prediction Only (No GT)")
            self.evaluation_completed.emit(metadata, None, None, psnr_val, ssim_val)

        self.card_quality.set_value(f"Q{metadata.compression_quality}")

    def _calculate_fidelity(
        self, orig_path: Optional[Path], comp_path: Optional[Path]
    ) -> Tuple[Optional[float], Optional[float]]:
        if not orig_path or not comp_path or not orig_path.exists() or not comp_path.exists():
            return None, None
        try:
            from sender.image_metrics import calculate_psnr, calculate_ssim

            with Image.open(orig_path) as orig_pil, Image.open(comp_path) as comp_pil:
                orig_rgb = orig_pil.convert("RGB")
                comp_rgb = comp_pil.convert("RGB")
                if orig_rgb.size != comp_rgb.size:
                    comp_rgb = comp_rgb.resize(orig_rgb.size, Image.Resampling.BILINEAR)
                orig_arr = np.array(orig_rgb)
                comp_arr = np.array(comp_rgb)
                psnr = calculate_psnr(orig_arr, comp_arr)
                ssim = calculate_ssim(orig_arr, comp_arr)
                return psnr, ssim
        except Exception:
            return None, None

    def _load_original_uncompressed(self, image_id: str):
        found_orig = locate_original_uncompressed(image_id, self.search_roots)
        self.current_orig_path = found_orig
        if found_orig and found_orig.exists():
            pix = QPixmap(str(found_orig))
            self.orig_box.set_pixmap(pix, caption=f"Original: {found_orig.name}")
        else:
            self.orig_box.clear_image("Original not local")

    def _re_evaluate(self):
        if self.current_result and self.current_metadata:
            self.load_evaluation_data(self.current_result, self.current_metadata, self.current_comp_path)
