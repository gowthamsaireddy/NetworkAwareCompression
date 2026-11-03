"""Receiver Segmentation Evaluation View (Dice & IoU)."""

from pathlib import Path
from typing import List, Optional
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
    QVBoxLayout,
    QWidget,
)
from common.config import DATA_DIR
from receiver.segmentation import SegmentationResult
from receiver.segmentation_metrics import (
    create_comparative_overlay,
    evaluate_segmentation,
    locate_ground_truth_mask,
)
from shared.constants import STATUS_NOT_CALCULATED
from shared.metadata import TransmissionMetadata
from ui.shared.widgets import ImagePreviewBox, MetricBox, SectionHeader


class ReceiverEvaluationView(QWidget):
    """Comparative evaluation against verified ground-truth masks (Dice and IoU metrics)."""

    evaluation_completed = pyqtSignal(object, object, object)  # Emits (metadata, dice, iou)

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
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        header = SectionHeader(
            "Evaluation & Ground-Truth Comparison",
            "Evaluate segmentation accuracy against verified ground-truth masks using genuine Dice and IoU metrics",
        )
        layout.addWidget(header)

        # Action bar
        ctrl_frame = QFrame()
        ctrl_frame.setObjectName("CardFrame")
        ctrl_layout = QHBoxLayout(ctrl_frame)

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

        # 5-Panel Display:
        # Top row: Original Image | Received Compressed | Ground-Truth Mask
        # Bottom row: Predicted Mask | Comparative Overlay
        preview_grid = QGridLayout()
        preview_grid.setSpacing(10)

        self.orig_box = ImagePreviewBox("Original Uncompressed", "No image")
        self.comp_box = ImagePreviewBox("Received Compressed", "No image")
        self.gt_box = ImagePreviewBox("Ground-Truth Mask", "No ground truth")
        self.pred_box = ImagePreviewBox("Predicted Mask (256×256)", "No prediction")
        self.overlay_box = ImagePreviewBox("Comparative Overlay (Yellow=TP, Red=FP, Green=FN)", "No overlay")

        preview_grid.addWidget(self.orig_box, 0, 0)
        preview_grid.addWidget(self.comp_box, 0, 1)
        preview_grid.addWidget(self.gt_box, 0, 2)
        preview_grid.addWidget(self.pred_box, 1, 0)
        preview_grid.addWidget(self.overlay_box, 1, 1, 1, 2)

        layout.addLayout(preview_grid, stretch=1)

        # Quantitative Metrics Grid
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(12)

        self.card_dice = MetricBox("Dice Similarity Coefficient", STATUS_NOT_CALCULATED)
        self.card_iou = MetricBox("Intersection over Union (IoU)", STATUS_NOT_CALCULATED)
        self.card_gt_match = MetricBox("Ground Truth Status", "Unverified")
        self.card_quality = MetricBox("Compression Quality", STATUS_NOT_CALCULATED)

        metrics_grid.addWidget(self.card_dice, 0, 0)
        metrics_grid.addWidget(self.card_iou, 0, 1)
        metrics_grid.addWidget(self.card_gt_match, 0, 2)
        metrics_grid.addWidget(self.card_quality, 0, 3)

        layout.addLayout(metrics_grid)

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

        # 3. Locate Ground-Truth Mask
        gt_mask = locate_ground_truth_mask(metadata.image_id, self.search_roots)

        # 4. Search for Original Uncompressed Image
        self._load_original_uncompressed(metadata.image_id)

        # 5. Evaluate Metrics & Render Comparative Overlay
        if gt_mask is not None:
            # Ground truth located
            gh, gw = gt_mask.shape
            q_gt = QImage(gt_mask.data, gw, gh, gw, QImage.Format.Format_Grayscale8)
            self.gt_box.set_pixmap(QPixmap.fromImage(q_gt), caption="256 × 256 Binary Ground Truth")

            dice, iou, status = evaluate_segmentation(gt_mask, result.predicted_mask)
            self.card_dice.set_value(f"{dice:.4f}" if dice is not None else STATUS_NOT_CALCULATED)
            self.card_iou.set_value(f"{iou:.4f}" if iou is not None else STATUS_NOT_CALCULATED)
            self.card_gt_match.set_value("Ground Truth Matched")

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
            self.evaluation_completed.emit(metadata, dice, iou)
        else:
            # Zero Dummy-Data policy: Do not calculate or fabricate Dice/IoU
            self.gt_box.clear_image("Ground truth unavailable")
            self.card_dice.set_value(STATUS_NOT_CALCULATED)
            self.card_iou.set_value(STATUS_NOT_CALCULATED)
            self.card_gt_match.set_value("No GT Mask Found")

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
            self.evaluation_completed.emit(metadata, None, None)

        self.card_quality.set_value(f"Q{metadata.compression_quality}")

    def _load_original_uncompressed(self, image_id: str):
        clean_name = Path(image_id).stem.lower()
        found_orig: Optional[Path] = None

        for root in self.search_roots:
            if not root.exists():
                continue
            candidates = [
                f for f in root.rglob("*.png")
                if "_mask" not in f.stem.lower() and clean_name in f.stem.lower()
            ]
            if candidates:
                found_orig = candidates[0]
                break

        if found_orig and found_orig.exists():
            pix = QPixmap(str(found_orig))
            self.orig_box.set_pixmap(pix, caption=f"Original: {found_orig.name}")
        else:
            self.orig_box.clear_image("Original not local")

    def _re_evaluate(self):
        if self.current_result and self.current_metadata:
            self.load_evaluation_data(self.current_result, self.current_metadata, self.current_comp_path)
