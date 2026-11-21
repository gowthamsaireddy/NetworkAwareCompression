"""Evaluation metrics for lesion segmentation: Dice Coefficient and Intersection over Union (IoU).

Calculates genuine metrics between binary predicted masks and binary ground truth.
Enforces strict Zero Dummy-Data policy: returns None if ground truth is unavailable.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from common.logging_config import setup_logger

logger = setup_logger("SegmentationMetrics")


def get_canonical_busi_id(raw_id: str) -> Tuple[str, str, str, str]:
    """Extracts canonical BUSI dataset identification from any image_id variant.

    Robustly handles:
      - 'benign/benign (1)' -> orig_id='benign/benign (1)', base_stem='benign (1)', cat='benign', orig_filename='benign (1).png'
      - 'benign (1)' -> orig_id='benign/benign (1)', base_stem='benign (1)', cat='benign', orig_filename='benign (1).png'
      - 'benign_benign (1)' -> orig_id='benign/benign (1)', base_stem='benign (1)', cat='benign', orig_filename='benign (1).png'
      - 'benign_benign (148)_Q90_RUN_...' -> orig_id='benign/benign (148)', base_stem='benign (148)', cat='benign', orig_filename='benign (148).png'
      - 'BG70_Satellite_benign_benign__148_' -> orig_id='benign/benign (148)', base_stem='benign (148)', cat='benign', orig_filename='benign (148).png'
      - 'malignant/malignant (23)' -> orig_id='malignant/malignant (23)', base_stem='malignant (23)', cat='malignant', orig_filename='malignant (23).png'
      - 'normal/normal (5)' -> orig_id='normal/normal (5)', base_stem='normal (5)', cat='normal', orig_filename='normal (5).png'

    Returns:
        (original_image_id, base_stem, category, original_filename)
    """
    s = str(raw_id).strip().replace("\\", "/")
    tail = s.split("/")[-1].strip() if "/" in s else s

    # 1. Determine category: benign, malignant, normal
    cat = "benign"
    low_s = s.lower()
    for c in ("malignant", "normal", "benign"):
        if c in low_s:
            cat = c
            break

    # 2. Extract BUSI index number (e.g. '(148)', '__148_', '_148_', or bare '148')
    num_match = re.search(r"\(\s*(\d+)\s*\)", tail)
    if not num_match:
        num_match = re.search(r"__(\d+)_", tail)
    if not num_match:
        num_match = re.search(r"[\s_](\d+)(?:[^\d]|$)", tail)

    if num_match:
        num = num_match.group(1)
        base_stem = f"{cat} ({num})"
    else:
        # Fallback to tail without extension
        base_stem = Path(tail).stem

    orig_id = f"{cat}/{base_stem}"
    orig_filename = f"{base_stem}.png"
    return orig_id, base_stem, cat, orig_filename


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


def locate_ground_truth_mask(
    image_id: str, search_roots: List[Path]
) -> Tuple[Optional[np.ndarray], Optional[str], List[Path]]:
    """Locates and loads ground-truth mask(s) for a given image_id across search roots.

    Performs exact base stem matching to eliminate false substring matches.
    Combines multiple lesion masks (e.g., _mask.png, _mask_1.png) using bitwise maximum.
    Resizes combined mask to target 256x256 using nearest-neighbor interpolation.

    Returns:
        (combined_mask_256, mask_identifier, matching_file_paths)
        or (None, None, []) if no ground truth mask exists.
    """
    _, base_stem, _, _ = get_canonical_busi_id(image_id)
    target_base = base_stem.lower().strip()

    matching_masks: List[Path] = []

    for root in search_roots:
        if not root.exists():
            continue

        for f in root.rglob("*.png"):
            stem_low = f.stem.lower()
            if "_mask" in stem_low:
                candidate_base = re.sub(r"_mask.*$", "", stem_low, flags=re.IGNORECASE).strip()
                if candidate_base == target_base:
                    if f not in matching_masks:
                        matching_masks.append(f)

        if matching_masks:
            break

    if not matching_masks:
        logger.warning(f"No ground-truth mask found for '{image_id}' (canonical stem: '{base_stem}').")
        return None, None, []

    combined: Optional[np.ndarray] = None
    for mp in sorted(matching_masks):
        try:
            with Image.open(mp) as img:
                arr = np.array(img.convert("L"))
                bin_mask = (arr > 0).astype(np.uint8) * 255
                if combined is None:
                    combined = bin_mask
                else:
                    if combined.shape == bin_mask.shape:
                        combined = np.maximum(combined, bin_mask)
        except Exception as e:
            logger.error(f"Error loading mask {mp}: {e}")

    if combined is not None:
        resized_gt = cv2.resize(combined, (256, 256), interpolation=cv2.INTER_NEAREST)
        final_mask = (resized_gt > 0).astype(np.uint8) * 255
        mask_id = ", ".join(m.name for m in matching_masks)
        logger.info(f"Matched {len(matching_masks)} ground-truth mask(s) for '{image_id}': {mask_id}")
        return final_mask, mask_id, matching_masks

    return None, None, []


def locate_original_uncompressed(image_id: str, search_roots: List[Path]) -> Optional[Path]:
    """Locates the original uncompressed source image for image_id across search roots."""
    _, base_stem, _, _ = get_canonical_busi_id(image_id)
    target_base = base_stem.lower().strip()

    for root in search_roots:
        if not root.exists():
            continue

        for f in root.rglob("*.png"):
            stem_low = f.stem.lower()
            if "_mask" not in stem_low and stem_low == target_base:
                return f

    logger.warning(f"Original uncompressed image not found for '{image_id}' (canonical stem: '{base_stem}').")
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
