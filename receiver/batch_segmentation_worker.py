"""High-Throughput Background Worker for Batch U-Net Segmentation & Evaluation.

Prevents GUI freezes by running U-Net inference, ground-truth matching,
metric evaluation (Dice, IoU, PSNR, SSIM), and database persistence on a dedicated QThread.

Pre-indexes ground-truth masks and source images once into memory to eliminate
repeated recursive disk scans across 480+ images.
"""

from datetime import datetime
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from common.config import DATA_DIR
from common.logging_config import setup_logger
from common.results_manager import (
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    ExperimentRecord,
    ResultsManager,
)
from receiver.segmentation import SegmentationEngine, SegmentationResult
from receiver.segmentation_metrics import evaluate_segmentation, get_canonical_busi_id
from sender.image_metrics import calculate_psnr, calculate_ssim
from shared.metadata import TransmissionMetadata

logger = setup_logger("BatchSegmentationWorker")


class BatchSegmentationWorker(QThread):
    """Executes high-throughput background U-Net inference across all received images."""

    progress_updated = pyqtSignal(int, int, str, object, object, float)  # curr, total, image_id, dice, iou, latency_ms
    batch_finished = pyqtSignal(int, int, float, float, str)  # processed, total, avg_dice, avg_iou, summary

    def __init__(
        self,
        engine: SegmentationEngine,
        items: List[Tuple[TransmissionMetadata, Path]],
        results_manager: ResultsManager,
        rx_telemetry: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.engine = engine
        self.items = list(items)
        self.results_manager = results_manager
        self.rx_telemetry = dict(rx_telemetry) if rx_telemetry else {}
        self._stop_requested = False

    def stop(self):
        """Cooperatively stops the background worker."""
        self._stop_requested = True

    def run(self):
        t_batch_start = time.perf_counter()
        total = len(self.items)

        logger.info(f"Starting background batch segmentation across {total} received images...")

        # 1. Pre-index masks and original uncompressed images ONCE using canonical base stems
        search_roots = [
            DATA_DIR / "BUSI",
            DATA_DIR / "BUSI" / "benign",
            DATA_DIR / "BUSI" / "malignant",
            DATA_DIR / "BUSI" / "normal",
            DATA_DIR,
        ]

        mask_map: Dict[str, List[Path]] = {}
        orig_map: Dict[str, Path] = {}

        for root in search_roots:
            if not root.exists():
                continue
            for f in root.rglob("*.png"):
                stem_lower = f.stem.lower()
                if "_mask" in stem_lower:
                    base = re.sub(r"_mask.*$", "", stem_lower, flags=re.IGNORECASE).strip()
                    mask_map.setdefault(base, []).append(f)
                else:
                    orig_map[stem_lower] = f

        logger.info(f"Pre-indexed {len(mask_map)} ground-truth masks and {len(orig_map)} original references in memory.")

        processed = 0
        dices: List[float] = []
        ious: List[float] = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for idx, (metadata, save_path) in enumerate(self.items, 1):
            if self._stop_requested:
                logger.info("Batch segmentation cancelled by user.")
                break

            if not save_path.exists():
                continue

            # Run frozen U-Net inference on background thread
            success, result, msg = self.engine.run_inference(save_path, metadata.image_id)
            if not success or result is None:
                continue

            # Canonical O(1) in-memory ground-truth lookup
            orig_id, base_stem, cat, orig_fn = get_canonical_busi_id(metadata.image_id)
            target_key = base_stem.lower().strip()
            matching_masks = mask_map.get(target_key, [])

            dice_val: Optional[float] = None
            iou_val: Optional[float] = None

            if matching_masks:
                combined = None
                for mp in sorted(matching_masks):
                    try:
                        m_raw = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                        if m_raw is not None:
                            bin_m = (m_raw > 0).astype(np.uint8) * 255
                            combined = bin_m if combined is None else np.maximum(combined, bin_m)
                    except Exception:
                        pass

                if combined is not None:
                    resized_gt = cv2.resize(combined, (256, 256), interpolation=cv2.INTER_NEAREST)
                    gt_bin = (resized_gt > 0).astype(np.uint8) * 255
                    dice_val, iou_val, _ = evaluate_segmentation(gt_bin, result.predicted_mask)
                    if dice_val is not None:
                        dices.append(dice_val)
                    if iou_val is not None:
                        ious.append(iou_val)

            # Canonical O(1) fidelity lookup
            orig_p = orig_map.get(target_key)
            psnr_val, ssim_val = None, None
            if orig_p and orig_p.exists() and save_path.exists():
                try:
                    with Image.open(orig_p) as orig_pil, Image.open(save_path) as comp_pil:
                        orig_rgb = orig_pil.convert("RGB")
                        comp_rgb = comp_pil.convert("RGB")
                        if orig_rgb.size != comp_rgb.size:
                            comp_rgb = comp_rgb.resize(orig_rgb.size, Image.Resampling.BILINEAR)
                        orig_arr = np.array(orig_rgb)
                        comp_arr = np.array(comp_rgb)
                        psnr_val = calculate_psnr(orig_arr, comp_arr)
                        ssim_val = calculate_ssim(orig_arr, comp_arr)
                except Exception:
                    pass

            # Telemetry lookup
            tx_time_val, thru_val = None, None
            if metadata.experiment_id in self.rx_telemetry:
                cached_time, cached_thru = self.rx_telemetry[metadata.experiment_id]
                if cached_time > 0.0:
                    tx_time_val = cached_time
                if cached_thru > 0.0:
                    thru_val = cached_thru

            ratio = (
                (metadata.original_file_size / metadata.compressed_file_size)
                if metadata.compressed_file_size > 0
                else 1.0
            )

            gt_stat = "MATCHED" if matching_masks else "MISSING_GT"
            exp_stat = (
                STATUS_COMPLETED
                if (dice_val is not None and psnr_val is not None)
                else STATUS_INCOMPLETE
            )

            rec = ExperimentRecord(
                experiment_id=metadata.experiment_id,
                timestamp=now_str,
                image_id=metadata.image_id,
                original_image_id=orig_id,
                original_image_filename=orig_fn,
                image_category=cat,
                compression_method=metadata.compression_mode,
                compression_quality=metadata.compression_quality,
                region_adaptive_margin=metadata.region_adaptive_margin,
                compression_model_version=metadata.compression_model_version,
                original_file_size=metadata.original_file_size,
                compressed_file_size=metadata.compressed_file_size,
                compression_ratio=ratio,
                psnr=psnr_val,
                ssim=ssim_val,
                bandwidth_limit=None,
                packet_loss_rate=None,
                network_delay_ms=None,
                theoretical_transmission_time_sec=None,
                transmission_time_sec=tx_time_val,
                measured_throughput_kbps=thru_val,
                checksum_verified=True,
                dice_coefficient=dice_val,
                iou=iou_val,
                ground_truth_status=gt_stat,
                experiment_status=exp_stat,
                completion_time=now_str,
                inference_time_ms=result.inference_time_ms,
                notes=f"Batch evaluated: {metadata.compression_mode}",
            )

            # Persist record directly to SQLite in background without per-item CSV re-serialization
            self.results_manager.record_or_update_experiment(rec, sync_csv=False)
            processed += 1

            self.progress_updated.emit(
                processed,
                total,
                metadata.image_id,
                dice_val,
                iou_val,
                result.inference_time_ms,
            )

        # Synchronize CSV once at batch completion
        self.results_manager.sync_csv()

        t_elapsed = time.perf_counter() - t_batch_start
        avg_d = float(np.mean(dices)) if dices else 0.0
        avg_i = float(np.mean(ious)) if ious else 0.0
        summary = (
            f"Successfully segmented and evaluated {processed}/{total} images in {t_elapsed:.1f}s "
            f"({(processed / max(t_elapsed, 0.001)):.1f} imgs/sec).\n"
            f"Mean Dice (DSC): {avg_d:.4f} | Mean IoU: {avg_i:.4f}."
        )
        logger.info(summary)
        self.batch_finished.emit(processed, total, avg_d, avg_i, summary)
