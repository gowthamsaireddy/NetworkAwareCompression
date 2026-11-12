"""Baseline U-Net Lesion Segmentation Evaluation on Held-Out Test Split.

Evaluates uncompressed original BUSI test images:
1. Loads models/best_unet.pth in frozen evaluation mode via receiver.model_manager.ModelManager.
2. Uses the 117 held-out test images from data/BUSI/splits.json exactly once.
3. Uses the receiver's SegmentationEngine (256x256, grayscale, float32 [0..1], 0.5 threshold).
4. Compares predictions with ground-truth masks (combining multi-masks via np.maximum).
5. Explicitly documents and handles empty mask cases (True Negatives = 1.0, False Positives/Negatives = 0.0).
6. Exports results to results/test_baseline.csv and results/test_baseline_summary.json.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from receiver.model_manager import ModelManager
from receiver.segmentation import SegmentationEngine, SegmentationResult
from receiver.segmentation_metrics import compute_dice_coefficient, compute_iou


def load_ground_truth_for_item(item: Dict, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Loads ground-truth mask for a dataset item, combining multi-masks via np.maximum."""
    mask_paths = [PROJECT_ROOT / p for p in item.get("mask_paths", [])]
    category = item.get("category", "")

    if not mask_paths or category == "normal":
        # Normal scan: strictly empty mask (0 lesion pixels)
        return np.zeros(target_size, dtype=np.uint8)

    combined: Optional[np.ndarray] = None
    for mp in mask_paths:
        if not mp.exists():
            continue
        m_raw = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m_raw is None:
            m_raw = np.array(Image.open(mp).convert("L"))

        if combined is None:
            combined = m_raw
        else:
            combined = np.maximum(combined, m_raw)

    if combined is None:
        return np.zeros(target_size, dtype=np.uint8)

    # Resize to target 256x256 using nearest-neighbor to prevent fractional mask edges
    resized_gt = cv2.resize(combined, target_size, interpolation=cv2.INTER_NEAREST)
    return (resized_gt > 127).astype(np.uint8) * 255


def run_baseline_evaluation(
    checkpoint_path: Path = PROJECT_ROOT / "models" / "best_unet.pth",
    splits_path: Path = PROJECT_ROOT / "data" / "BUSI" / "splits.json",
    results_dir: Path = PROJECT_ROOT / "results",
) -> Dict:
    """Executes baseline evaluation across all 117 held-out test samples."""
    print("=" * 80)
    print("BASELINE U-NET TEST SET SEGMENTATION EVALUATION")
    print("=" * 80)

    # 1. Verify files
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found at: {splits_path}")

    results_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = results_dir / "test_baseline.csv"
    summary_output_path = results_dir / "test_baseline_summary.json"

    # 2. Load Model via receiver's ModelManager
    model_mgr = ModelManager()
    success, msg = model_mgr.load_checkpoint(checkpoint_path)
    if not success or not model_mgr.is_loaded:
        raise RuntimeError(f"ModelManager failed to load checkpoint: {msg}")

    print(f"[+] Loaded model checkpoint: {checkpoint_path.name}")
    print(f"[+] Compute Device: {model_mgr.device} ({torch.cuda.get_device_name(0) if model_mgr.device.type == 'cuda' else 'CPU'})")
    print(f"[+] Model Status: {model_mgr.get_status()}")

    # 3. Initialize receiver's SegmentationEngine
    engine = SegmentationEngine(model_mgr)

    # 4. Load held-out test split
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    test_items = splits_data.get("test", [])
    expected_count = 117
    actual_count = len(test_items)
    print(f"[+] Held-out test items: {actual_count} samples (Expected: {expected_count})")
    assert actual_count == expected_count, f"Test count mismatch: expected {expected_count}, got {actual_count}"

    records = []
    skipped_samples = []
    start_eval_time = time.perf_counter()

    for idx, item in enumerate(test_items, start=1):
        image_id = item["image_id"]
        img_path = PROJECT_ROOT / item["image_path"]
        category = item["category"]

        if not img_path.exists():
            err_msg = f"Missing file: {img_path}"
            print(f"[-] ERROR on sample {idx}: {err_msg}")
            skipped_samples.append({"image_id": image_id, "error": err_msg})
            continue

        # Step A: Run receiver inference
        ok, res, err = engine.run_inference(img_path, image_id=image_id, threshold=0.5)
        if not ok or res is None:
            err_msg = f"Inference failed for {image_id}: {err}"
            print(f"[-] ERROR on sample {idx}: {err_msg}")
            skipped_samples.append({"image_id": image_id, "error": err_msg})
            continue

        # Step B: Load Ground Truth mask (256x256, {0, 255})
        gt_mask = load_ground_truth_for_item(item, target_size=(256, 256))
        pred_mask = res.predicted_mask  # (256, 256), {0, 255}

        # Step C: Compute Metrics
        dice = compute_dice_coefficient(gt_mask, pred_mask)
        iou = compute_iou(gt_mask, pred_mask)

        gt_bin = gt_mask > 0
        pred_bin = pred_mask > 0

        tp_pixels = int(np.sum(gt_bin & pred_bin))
        fp_pixels = int(np.sum((~gt_bin) & pred_bin))
        fn_pixels = int(np.sum(gt_bin & (~pred_bin)))
        tn_pixels = int(np.sum((~gt_bin) & (~pred_bin)))
        gt_pixels = int(np.sum(gt_bin))
        pred_pixels = int(np.sum(pred_bin))

        record = {
            "sample_index": idx,
            "image_id": image_id,
            "category": category,
            "image_path": item["image_path"],
            "original_width": item["width"],
            "original_height": item["height"],
            "original_size_bytes": item["file_size_bytes"],
            "gt_mask_count": len(item.get("mask_paths", [])),
            "gt_lesion_pixels": gt_pixels,
            "pred_lesion_pixels": pred_pixels,
            "true_positive_pixels": tp_pixels,
            "false_positive_pixels": fp_pixels,
            "false_negative_pixels": fn_pixels,
            "true_negative_pixels": tn_pixels,
            "dice": dice,
            "iou": iou,
            "inference_time_ms": res.inference_time_ms,
        }
        records.append(record)

    total_eval_duration = time.perf_counter() - start_eval_time

    # 5. Build DataFrame and Compute Aggregates
    df = pd.DataFrame(records)
    df.to_csv(csv_output_path, index=False)
    print(f"[+] Saved per-sample evaluation results to: {csv_output_path}")

    # Summary by Category
    cat_summary = {}
    for cat in ["benign", "malignant", "normal"]:
        sub_df = df[df["category"] == cat]
        if not sub_df.empty:
            cat_summary[cat] = {
                "count": len(sub_df),
                "dice_mean": float(sub_df["dice"].mean()),
                "dice_std": float(sub_df["dice"].std()),
                "dice_median": float(sub_df["dice"].median()),
                "iou_mean": float(sub_df["iou"].mean()),
                "iou_std": float(sub_df["iou"].std()),
                "iou_median": float(sub_df["iou"].median()),
                "avg_inference_time_ms": float(sub_df["inference_time_ms"].mean()),
            }

    # Lesion-only (Benign + Malignant where GT > 0)
    lesion_df = df[df["category"].isin(["benign", "malignant"])]
    lesion_summary = {
        "count": len(lesion_df),
        "dice_mean": float(lesion_df["dice"].mean()),
        "dice_std": float(lesion_df["dice"].std()),
        "dice_median": float(lesion_df["dice"].median()),
        "iou_mean": float(lesion_df["iou"].mean()),
        "iou_std": float(lesion_df["iou"].std()),
        "iou_median": float(lesion_df["iou"].median()),
        "avg_inference_time_ms": float(lesion_df["inference_time_ms"].mean()),
    }

    # Overall Summary (All 117 test samples)
    overall_summary = {
        "count": len(df),
        "dice_mean": float(df["dice"].mean()),
        "dice_std": float(df["dice"].std()),
        "dice_median": float(df["dice"].median()),
        "iou_mean": float(df["iou"].mean()),
        "iou_std": float(df["iou"].std()),
        "iou_median": float(df["iou"].median()),
        "avg_inference_time_ms": float(df["inference_time_ms"].mean()),
    }

    summary_payload = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_evaluated": checkpoint_path.name,
        "device": str(model_mgr.device),
        "total_test_samples_evaluated": len(df),
        "total_skipped_samples": len(skipped_samples),
        "skipped_details": skipped_samples,
        "total_evaluation_time_sec": total_eval_duration,
        "empty_mask_policy": (
            "When ground-truth is empty (healthy/normal scan): if prediction is also empty (True Negative), "
            "Dice=1.0 and IoU=1.0; if prediction contains false positives, Dice=0.0 and IoU=0.0."
        ),
        "overall_metrics_all_samples": overall_summary,
        "lesion_cases_only_benign_malignant": lesion_summary,
        "category_breakdown": cat_summary,
    }

    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[+] Saved evaluation summary to: {summary_output_path}")

    # 6. Display Clean Academic Table
    print("\n" + "=" * 80)
    print("HELD-OUT TEST SPLIT BASELINE EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Cohort / Category':<28} | {'N':<5} | {'Dice (Mean ± Std)':<20} | {'IoU (Mean ± Std)':<20} | {'Avg Latency'}")
    print("-" * 80)

    for cat in ["benign", "malignant", "normal"]:
        stats = cat_summary.get(cat, {})
        d_str = f"{stats.get('dice_mean', 0.0):.4f} ± {stats.get('dice_std', 0.0):.4f}"
        j_str = f"{stats.get('iou_mean', 0.0):.4f} ± {stats.get('iou_std', 0.0):.4f}"
        lat_str = f"{stats.get('avg_inference_time_ms', 0.0):.2f} ms"
        print(f"{cat.capitalize():<28} | {stats.get('count', 0):<5} | {d_str:<20} | {j_str:<20} | {lat_str}")

    print("-" * 80)
    # Lesion-only
    d_l_str = f"{lesion_summary['dice_mean']:.4f} ± {lesion_summary['dice_std']:.4f}"
    j_l_str = f"{lesion_summary['iou_mean']:.4f} ± {lesion_summary['iou_std']:.4f}"
    lat_l_str = f"{lesion_summary['avg_inference_time_ms']:.2f} ms"
    print(f"{'Lesion Cases (B + M)':<28} | {lesion_summary['count']:<5} | {d_l_str:<20} | {j_l_str:<20} | {lat_l_str}")

    # Overall
    d_o_str = f"{overall_summary['dice_mean']:.4f} ± {overall_summary['dice_std']:.4f}"
    j_o_str = f"{overall_summary['iou_mean']:.4f} ± {overall_summary['iou_std']:.4f}"
    lat_o_str = f"{overall_summary['avg_inference_time_ms']:.2f} ms"
    print(f"{'Overall (All Test Samples)':<28} | {overall_summary['count']:<5} | {d_o_str:<20} | {j_o_str:<20} | {lat_o_str}")
    print("=" * 80)
    print(f"Total Test Set Evaluation Time: {total_eval_duration:.2f} seconds.")
    print(f"Total Samples Evaluated: {len(df)}/117 (Skipped: {len(skipped_samples)})")
    print("=" * 80 + "\n")

    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline segmentation on BUSI test split.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "best_unet.pth")
    parser.add_argument("--splits", type=Path, default=PROJECT_ROOT / "data" / "BUSI" / "splits.json")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    run_baseline_evaluation(
        checkpoint_path=args.checkpoint,
        splits_path=args.splits,
        results_dir=args.results_dir,
    )

