"""Evaluation metrics for lesion segmentation: Dice Coefficient and Intersection over Union (IoU).

Calculates genuine metrics between binary predicted masks and binary ground truth.
Enforces strict Zero Dummy-Data policy: returns None if ground truth is unavailable.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from common.logging_config import setup_logger

logger = setup_logger("SegmentationMetrics")


def compute_dice_coefficient(ground_truth: np.ndarray, prediction: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculates the Dice Similarity Coefficient between two binary masks.

    Formula: (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
    """
    gt_bin = (ground_truth > 0).astype(np.float32)
    pred_bin = (prediction > 0).astype(np.float32)

    intersection = np.sum(gt_bin * pred_bin)
    total_area = np.sum(gt_bin) + np.sum(pred_bin)

    if total_area == 0:
        # Both masks completely empty (no lesion) -> perfect match
        return 1.0

    dice = (2.0 * intersection + smooth) / (total_area + smooth)
    return float(np.clip(dice, 0.0, 1.0))


def compute_iou(ground_truth: np.ndarray, prediction: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculates Intersection over Union (Jaccard Index) between two binary masks.

    Formula: (|X n Y| + smooth) / (|X U Y| + smooth)
    """
    gt_bin = (ground_truth > 0).astype(np.float32)
    pred_bin = (prediction > 0).astype(np.float32)

    intersection = np.sum(gt_bin * pred_bin)
    union = np.sum(gt_bin) + np.sum(pred_bin) - intersection

    if union == 0:
        return 1.0

    iou = (intersection + smooth) / (union + smooth)
    return float(np.clip(iou, 0.0, 1.0))


def locate_ground_truth_mask(image_id: str, search_roots: List[Path]) -> Optional[np.ndarray]:
    """Locates and loads ground-truth mask(s) for a given image_id across search roots.

    Combines multiple lesion masks if present using bitwise maximum.
    Resizes combined mask to target 256x256 using nearest-neighbor interpolation.
    Returns None if no ground truth mask exists.
    """
    clean_id = Path(image_id).stem.lower().replace("_mask", "")

    for root in search_roots:
        if not root.exists():
            continue

        # Look for matching mask files
        matching_masks = [
            f for f in root.rglob("*.png")
            if "_mask" in f.stem.lower() and clean_id in f.stem.lower()
        ]

        if matching_masks:
            combined: Optional[np.ndarray] = None
            for mp in sorted(matching_masks):
                with Image.open(mp) as img:
                    arr = np.array(img.convert("L"))
                    bin_mask = (arr > 0).astype(np.uint8) * 255
                    if combined is None:
                        combined = bin_mask
                    else:
                        if combined.shape == bin_mask.shape:
                            combined = np.maximum(combined, bin_mask)

            if combined is not None:
                # Resize to 256x256 with nearest neighbor to preserve binary edges
                resized_gt = cv2.resize(combined, (256, 256), interpolation=cv2.INTER_NEAREST)
                return (resized_gt > 0).astype(np.uint8) * 255

    return None


def create_comparative_overlay(
    base_gray_256: np.ndarray,
    ground_truth_256: Optional[np.ndarray],
    prediction_256: np.ndarray,
) -> np.ndarray:
    """Generates a color-coded comparative overlay:

    - Green: Ground-Truth mask only (False Negative)
    - Red: Predicted mask only (False Positive)
    - Yellow: Overlap (True Positive)
    """
    if len(base_gray_256.shape) == 2:
        base_rgb = cv2.cvtColor(base_gray_256, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_gray_256.copy()

    overlay = base_rgb.copy()

    if ground_truth_256 is None:
        # Prediction only (tint red)
        pred_idx = prediction_256 > 0
        overlay[pred_idx] = (0.5 * base_rgb[pred_idx] + 0.5 * np.array([239, 68, 68])).astype(np.uint8)
        return overlay

    gt_bin = ground_truth_256 > 0
    pred_bin = prediction_256 > 0

    tp = gt_bin & pred_bin       # True Positives: Yellow
    fp = (~gt_bin) & pred_bin    # False Positives: Red
    fn = gt_bin & (~pred_bin)    # False Negatives: Green

    # Apply tints
    if np.any(tp):
        overlay[tp] = (0.4 * base_rgb[tp] + 0.6 * np.array([234, 179, 8])).astype(np.uint8)
    if np.any(fp):
        overlay[fp] = (0.4 * base_rgb[fp] + 0.6 * np.array([239, 68, 68])).astype(np.uint8)
    if np.any(fn):
        overlay[fn] = (0.4 * base_rgb[fn] + 0.6 * np.array([34, 197, 94])).astype(np.uint8)

    return overlay


def evaluate_segmentation(
    ground_truth: Optional[np.ndarray],
    prediction: np.ndarray,
) -> Tuple[Optional[float], Optional[float], str]:
    """Evaluates prediction against ground truth.

    Returns (dice, iou, status_message).
    If ground_truth is None, returns (None, None, 'Ground truth unavailable')
    per zero-dummy data policy.
    """
    if ground_truth is None:
        return None, None, "Ground truth mask unavailable: Dice and IoU cannot be calculated."

    if ground_truth.shape != prediction.shape:
        return None, None, f"Shape mismatch between GT {ground_truth.shape} and Pred {prediction.shape}."

    dice = compute_dice_coefficient(ground_truth, prediction)
    iou = compute_iou(ground_truth, prediction)
    return dice, iou, "Evaluation complete"
