"""Evaluation metrics for lesion segmentation: Dice Coefficient and Intersection over Union (IoU).

Calculates genuine metrics between binary predicted masks and binary ground truth.
Returns None if ground truth is unavailable (Zero Dummy-Data Policy).
"""

from typing import Optional, Tuple
import numpy as np


def compute_dice_coefficient(ground_truth: np.ndarray, prediction: np.ndarray, smooth: float = 1e-6) -> float:
    """Calculates the Dice Similarity Coefficient between two binary masks.

    Formula: (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
    """
    gt_bin = (ground_truth > 0).astype(np.float32)
    pred_bin = (prediction > 0).astype(np.float32)

    intersection = np.sum(gt_bin * pred_bin)
    total_area = np.sum(gt_bin) + np.sum(pred_bin)

    if total_area == 0:
        # Both masks empty -> perfect match
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


def evaluate_segmentation(
    ground_truth: Optional[np.ndarray],
    prediction: np.ndarray,
) -> Tuple[Optional[float], Optional[float], str]:
    """Evaluates prediction against ground truth.

    Returns (dice, iou, status_message).
    If ground_truth is None or empty, returns (None, None, 'Ground truth unavailable')
    per zero-dummy data policy.
    """
    if ground_truth is None:
        return None, None, "Ground truth mask unavailable: Dice and IoU cannot be calculated."

    if ground_truth.shape != prediction.shape:
        return None, None, f"Shape mismatch between GT {ground_truth.shape} and Pred {prediction.shape}."

    dice = compute_dice_coefficient(ground_truth, prediction)
    iou = compute_iou(ground_truth, prediction)
    return dice, iou, "Evaluation complete"

