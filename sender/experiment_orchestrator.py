"""Experiment Orchestrator integrating Sender, Compression, Network Profiles, and Logging.

Executes reproducible research experiments:
1. Normal Compression (Fixed JPEG Q30, Q50, Q70, Q90 under controlled network profiles).
2. ML-Based Region-Adaptive Compression (Lesion-margin preservation + background compression).
3. Dynamic Adaptive Compression (Bandwidth-reactive rate allocation).

Enforces strict Controlled Experimental Execution:
- Explicit user confirmation before starting any batch.
- Pre-run estimation of trial count and duration.
- Pause, resume, and cancellation capability.
- Deduplication option to skip already completed experiments.
"""

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from common.logging_config import setup_logger
from common.results_manager import ExperimentRecord, ResultsManager
from sender.adaptive_controller import AdaptiveController, AdaptiveDecision
from sender.compression import CompressionResult, ImageCompressor
from sender.dataset_manager import DatasetItem
from sender.network_control import (
    NetworkController,
    NetworkProfile,
    compute_theoretical_transmission_time,
)
from sender.region_adaptive_compressor import RegionAdaptiveCompressor, RegionAdaptiveResult
from sender.tcp_client import TCPTransmissionClient, TransmissionReport
from shared.constants import PROTOCOL_VERSION
from shared.metadata import TransmissionMetadata
from shared.protocol import compute_sha256

logger = setup_logger("ExperimentOrchestrator")


class ExperimentOrchestrator:
    """Coordinates end-to-end compression, network emulation, transmission, and experiment recording."""

    def __init__(
        self,
        tcp_client: TCPTransmissionClient,
        network_controller: NetworkController,
        adaptive_controller: AdaptiveController,
        results_manager: Optional[ResultsManager] = None,
    ):
        self.tcp_client = tcp_client
        self.network_controller = network_controller
        self.adaptive_controller = adaptive_controller
        self.results_manager = results_manager or ResultsManager()
        self.adaptive_compressor = RegionAdaptiveCompressor()

    @staticmethod
    def estimate_batch_run(
        items_count: int,
        compression_method: str,
        qualities_count: int,
        profiles: List[Optional[NetworkProfile]],
        avg_image_bytes: int = 35000,
    ) -> Tuple[int, float]:
        """Calculates total planned trial count and estimated runtime in seconds."""
        trials_per_item = max(qualities_count, 1)
        total_trials = items_count * len(profiles) * trials_per_item

        # Processing overhead per trial (compression + verification + logging): ~0.06s (Normal) vs ~0.10s (Adaptive)
        base_proc_sec = 0.06 if compression_method == "NORMAL" else 0.10

        total_est_sec = 0.0
        for prof in profiles:
            bw_bps = prof.bandwidth_bps if prof else 100_000_000.0  # Local unshaped: ~100 Mbps
            delay_s = (prof.delay_ms / 1000.0) if prof else 0.001
            loss_factor = 1.0 + (prof.loss_percent / 100.0) if prof else 1.0
            wire_time = (avg_image_bytes * 8.0) / max(bw_bps, 1000.0)
            trial_sec = base_proc_sec + (wire_time * loss_factor) + (delay_s * 2)
            total_est_sec += trial_sec * items_count * trials_per_item

        return total_trials, round(total_est_sec, 1)

    def run_single_trial(
        self,
        item: DatasetItem,
        quality: int = 70,
        compression_method: str = "NORMAL",
        region_adaptive_margin: int = 15,
        adaptive_roi_quality: int = 90,
        adaptive_bg_quality: int = 20,
        roi_mode: str = "ML_MODEL",
        profile: Optional[NetworkProfile] = None,
        experiment_id_prefix: str = "TRIAL",
        is_adaptive: bool = False,
        adaptive_decision: Optional[AdaptiveDecision] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[bool, Optional[ExperimentRecord], str]:
        """Executes a single end-to-end trial (compression -> transmission -> verification -> logging)."""
        now_ts = int(time.time() * 1000)
        exp_id = f"{experiment_id_prefix}_{now_ts}_{compression_method[:4]}_Q{quality}"

        # 1. Step 1: Compress image (Normal JPEG vs Region-Adaptive)
        try:
            if compression_method == "REGION_ADAPTIVE":
                gt_mask = None
                if roi_mode == "GT_MASK" and item.mask_paths:
                    for mp in item.mask_paths:
                        if mp.exists():
                            m_arr = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                            if m_arr is not None:
                                gt_mask = m_arr
                                break

                adapt_res = self.adaptive_compressor.compress_image(
                    image_source=item.image_path,
                    margin_px=region_adaptive_margin,
                    roi_quality=adaptive_roi_quality,
                    bg_quality=adaptive_bg_quality,
                    roi_mode=roi_mode,
                    gt_mask=gt_mask,
                )
                comp_bytes = adapt_res.compressed_bytes
                orig_size = adapt_res.original_size_bytes
                comp_size = adapt_res.compressed_size_bytes
                comp_ratio = adapt_res.compression_ratio
                psnr = adapt_res.psnr
                ssim = adapt_res.ssim
                img_w = adapt_res.width
                img_h = adapt_res.height
                model_ver = adapt_res.model_version
                margin_val = region_adaptive_margin
                params_str = json.dumps({
                    "roi_quality": adaptive_roi_quality,
                    "bg_quality": adaptive_bg_quality,
                    "margin_px": region_adaptive_margin,
                    "roi_mode": roi_mode,
                })
            else:
                norm_res = ImageCompressor.compress_image(item.image_path, quality=quality)
                comp_bytes = norm_res.compressed_bytes
                orig_size = norm_res.original_size_bytes
                comp_size = norm_res.compressed_size_bytes
                comp_ratio = norm_res.compression_ratio
                psnr = norm_res.psnr
                ssim = norm_res.ssim
                img_w = norm_res.width
                img_h = norm_res.height
                model_ver = None
                margin_val = 0
                params_str = json.dumps({"quality": quality})

        except Exception as e:
            err = f"Compression failed for {item.image_id}: {e}"
            logger.error(err)
            return False, None, err

        # 2. Step 2: Prepare transmission metadata
        sha = compute_sha256(comp_bytes)
        meta = TransmissionMetadata(
            protocol_version=PROTOCOL_VERSION,
            experiment_id=exp_id,
            image_id=item.image_id,
            compression_quality=quality if compression_method == "NORMAL" else adaptive_roi_quality,
            image_width=img_w,
            image_height=img_h,
            image_format="JPEG",
            original_file_size=orig_size,
            compressed_file_size=comp_size,
            image_byte_length=comp_size,
            sha256_checksum=sha,
            compression_mode=compression_method,
            region_adaptive_margin=margin_val,
            compression_model_version=model_ver,
            image_category=item.category,
        )

        # 3. Step 3: Transmit over TCP and await verified ACK
        report: TransmissionReport = self.tcp_client.send_image(
            meta,
            comp_bytes,
            progress_callback=progress_callback,
        )

        if not report.is_success:
            err = f"Transmission unverified for {item.image_id}: {report.error_message or f'ACK code {report.ack_status}'}"
            logger.warning(err)
            return False, None, err

        # 4. Step 4: Update EWMA bandwidth tracker with genuine measured throughput
        self.adaptive_controller.update_measured_throughput(report.throughput_kbps)

        # 5. Step 5: Calculate theoretical wire time
        bw_str = profile.bandwidth if profile else "unshaped"
        loss_pct = profile.loss_percent if profile else 0.0
        delay_ms = float(profile.delay_ms) if profile else 0.0
        bw_bps = profile.bandwidth_bps if profile else 0.0
        theo_time = compute_theoretical_transmission_time(comp_size, bw_bps)

        notes = (
            f"Mode: {'Adaptive (' + adaptive_decision.mode + ')' if is_adaptive and adaptive_decision else compression_method}"
        )

        # 6. Step 6: Log complete record to SQLite & CSV
        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=report.timestamp,
            image_id=item.image_id,
            image_category=item.category,
            compression_method=compression_method,
            compression_quality=quality if compression_method == "NORMAL" else adaptive_roi_quality,
            compression_params=params_str,
            region_adaptive_margin=margin_val,
            compression_model_version=model_ver,
            original_file_size=orig_size,
            compressed_file_size=comp_size,
            transmitted_bytes=report.bytes_sent,
            received_bytes=report.received_bytes,
            compression_ratio=comp_ratio,
            psnr=psnr,
            ssim=ssim,
            bandwidth_limit=bw_str,
            packet_loss_rate=loss_pct,
            network_delay_ms=delay_ms,
            theoretical_transmission_time_sec=theo_time,
            transmission_time_sec=report.wire_time_sec,
            measured_throughput_kbps=report.throughput_kbps,
            connection_time_ms=report.connection_time_ms,
            round_trip_time_sec=report.round_trip_time_sec,
            effective_bandwidth_kbps=report.effective_bandwidth_kbps,
            decoding_time_ms=report.decoding_time_ms,
            verification_time_ms=report.verification_time_ms,
            inference_time_ms=report.inference_time_ms,
            checksum_verified=report.checksum_verified,
            dice_coefficient=report.dice,
            iou=report.iou,
            experiment_status="SUCCESS" if report.is_success else "FAILED",
            error_message=report.error_message,
            transmission_mode="ADAPTIVE" if is_adaptive else "BATCH_CONTROLLED",
            notes=notes,
        )

        self.results_manager.record_experiment(record)
        logger.info(
            f"Logged trial {exp_id} for {item.image_id} ({compression_method} Q{quality}): "
            f"Wire {report.wire_time_sec:.4f}s, {report.throughput_kbps:.2f} Kbps, PSNR={psnr:.2f}dB."
        )
        return True, record, "Trial completed and verified"

    def run_batch_experiment(
        self,
        items: List[DatasetItem],
        compression_method: str = "NORMAL",
        fixed_qualities: Optional[List[int]] = None,
        adaptive_margins: Optional[List[int]] = None,
        adaptive_roi_quality: int = 90,
        adaptive_bg_quality: int = 20,
        roi_mode: str = "ML_MODEL",
        profiles: Optional[List[Optional[NetworkProfile]]] = None,
        skip_completed: bool = True,
        max_batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str, Optional[ExperimentRecord]], None]] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
        pause_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[int, int, int, List[ExperimentRecord], str]:
        """Executes a controlled sequential experimental batch with pause/resume and skip-completed."""
        qualities = fixed_qualities or [70]
        margins = adaptive_margins or [15]
        net_profiles = profiles if profiles is not None else [None]

        # Configurable batch slice limit
        if max_batch_size and max_batch_size > 0:
            target_items = items[:max_batch_size]
        else:
            target_items = items

        # Calculate planned trial parameters
        param_sweep = qualities if compression_method == "NORMAL" else margins
        total_trials = len(target_items) * len(net_profiles) * len(param_sweep)

        completed = 0
        skipped = 0
        failed = 0
        records: List[ExperimentRecord] = []
        current_step = 0

        logger.info(
            f"Starting controlled batch: {len(target_items)} items, method='{compression_method}', "
            f"{len(net_profiles)} profile(s), {total_trials} total scheduled trials."
        )

        try:
            for prof in net_profiles:
                if stop_requested and stop_requested():
                    break

                bw_str = prof.bandwidth if prof else "unshaped"
                # Apply network traffic shaping profile if specified
                try:
                    if prof:
                        self.network_controller.apply_profile(prof)
                    else:
                        self.network_controller.reset()
                except Exception as e:
                    logger.warning(f"Network control failed (continuing unshaped): {e}")

                for item in target_items:
                    if stop_requested and stop_requested():
                        break

                    for param in param_sweep:
                        # Cooperative pause loop
                        while pause_requested and pause_requested():
                            if stop_requested and stop_requested():
                                break
                            time.sleep(0.3)

                        if stop_requested and stop_requested():
                            break

                        current_step += 1
                        q_val = param if compression_method == "NORMAL" else adaptive_roi_quality
                        m_val = param if compression_method == "REGION_ADAPTIVE" else 0

                        # Check deduplication if requested
                        if skip_completed and self.results_manager.is_experiment_completed(
                            image_id=item.image_id,
                            compression_method=compression_method,
                            compression_quality=q_val,
                            bandwidth_limit=bw_str,
                        ):
                            skipped += 1
                            step_desc = f"[{current_step}/{total_trials}] {item.image_id} - Skipped (Already Completed)"
                            if progress_callback:
                                progress_callback(current_step, total_trials, step_desc, None)
                            continue

                        step_desc = (
                            f"[{current_step}/{total_trials}] {item.image_id} "
                            f"({compression_method} {'Q' + str(param) if compression_method == 'NORMAL' else 'M' + str(param) + 'px'})"
                        )

                        success, rec, msg = self.run_single_trial(
                            item=item,
                            quality=q_val,
                            compression_method=compression_method,
                            region_adaptive_margin=m_val,
                            adaptive_roi_quality=adaptive_roi_quality,
                            adaptive_bg_quality=adaptive_bg_quality,
                            roi_mode=roi_mode,
                            profile=prof,
                            experiment_id_prefix="BATCH",
                        )

                        if success and rec:
                            completed += 1
                            records.append(rec)
                        else:
                            failed += 1

                        if progress_callback:
                            progress_callback(current_step, total_trials, step_desc, rec)

        finally:
            # Guarantee network traffic shaping cleanup upon completion, failure, or cancel
            try:
                self.network_controller.reset()
            except Exception:
                pass

        summary = (
            f"Batch finished: {completed} completed, {skipped} skipped, {failed} failed "
            f"out of {current_step} attempted trials."
        )
        logger.info(summary)
        return completed, skipped, failed, records, summary
