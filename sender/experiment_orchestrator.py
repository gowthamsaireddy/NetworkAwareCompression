"""Experiment Orchestrator integrating Sender, Compression, Network Profiles, and Logging.

Executes reproducible research experiments:
1. Fixed-Quality Evaluation (Q30, Q50, Q70, Q90 under controlled network profiles).
2. Adaptive-Quality Evaluation (dynamic or profile-based rate allocation).
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple

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

    def run_single_trial(
        self,
        item: DatasetItem,
        quality: int,
        profile: Optional[NetworkProfile] = None,
        experiment_id_prefix: str = "TRIAL",
        is_adaptive: bool = False,
        adaptive_decision: Optional[AdaptiveDecision] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[bool, Optional[ExperimentRecord], str]:
        """Executes a single end-to-end trial (compression -> transmission -> verification -> logging)."""
        now_ts = int(time.time())
        exp_id = f"{experiment_id_prefix}_{now_ts}"

        # 1. Step 1: Compress image
        try:
            comp_res = ImageCompressor.compress_image(item.image_path, quality=quality)
        except Exception as e:
            err = f"Compression failed for {item.image_id}: {e}"
            logger.error(err)
            return False, None, err

        # 2. Step 2: Prepare transmission metadata
        sha = compute_sha256(comp_res.compressed_bytes)
        meta = TransmissionMetadata(
            protocol_version=PROTOCOL_VERSION,
            experiment_id=exp_id,
            image_id=item.image_id,
            compression_quality=quality,
            image_width=comp_res.width,
            image_height=comp_res.height,
            image_format="JPEG",
            original_file_size=comp_res.original_size_bytes,
            compressed_file_size=comp_res.compressed_size_bytes,
            image_byte_length=comp_res.compressed_size_bytes,
            sha256_checksum=sha,
        )

        # 3. Step 3: Transmit over TCP and await verified ACK
        report: TransmissionReport = self.tcp_client.send_image(
            meta,
            comp_res.compressed_bytes,
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
        theo_time = compute_theoretical_transmission_time(comp_res.compressed_size_bytes, bw_bps)

        notes = f"Mode: {'Adaptive (' + adaptive_decision.mode + ')' if is_adaptive and adaptive_decision else 'Fixed Rate'}"

        # 6. Step 6: Log complete record to SQLite & CSV
        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=report.timestamp,
            image_id=item.image_id,
            compression_quality=quality,
            original_file_size=comp_res.original_size_bytes,
            compressed_file_size=comp_res.compressed_size_bytes,
            compression_ratio=comp_res.compression_ratio,
            psnr=comp_res.psnr,
            ssim=comp_res.ssim,
            bandwidth_limit=bw_str,
            packet_loss_rate=loss_pct,
            network_delay_ms=delay_ms,
            theoretical_transmission_time_sec=theo_time,
            transmission_time_sec=report.transmission_time_sec,
            measured_throughput_kbps=report.throughput_kbps,
            checksum_verified=report.checksum_verified,
            dice_coefficient=None,
            iou=None,
            notes=notes,
        )

        self.results_manager.record_experiment(record)
        logger.info(
            f"Logged trial {exp_id} for {item.image_id} (Q{quality}): {report.transmission_time_sec:.4f}s, "
            f"{report.throughput_kbps:.2f} Kbps, PSNR={comp_res.psnr:.2f}dB."
        )
        return True, record, "Trial completed and verified"

    def run_adaptive_trial(
        self,
        item: DatasetItem,
        profile: Optional[NetworkProfile] = None,
        manual_kbps: Optional[float] = None,
        use_ewma: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[bool, Optional[ExperimentRecord], str]:
        """Executes an adaptive trial where quality is automatically selected by the controller."""
        # Determine target quality
        bw_bps = profile.bandwidth_bps if profile else None
        decision = self.adaptive_controller.evaluate_with_fallback(
            manual_kbps=manual_kbps,
            profile_bw_bps=bw_bps,
            use_ewma=use_ewma,
        )

        return self.run_single_trial(
            item=item,
            quality=decision.quality,
            profile=profile,
            experiment_id_prefix="ADAPTIVE",
            is_adaptive=True,
            adaptive_decision=decision,
            progress_callback=progress_callback,
        )

    def run_batch_experiment(
        self,
        items: List[DatasetItem],
        quality_strategy: str = "fixed",  # "fixed" or "adaptive"
        fixed_qualities: Optional[List[int]] = None,
        profiles: Optional[List[Optional[NetworkProfile]]] = None,
        progress_callback: Optional[Callable[[int, int, str, Optional[ExperimentRecord]], None]] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[int, int, List[ExperimentRecord], str]:
        """Executes automated multi-image experimental runs with progress tracking and interruption safety.

        Guarantees:
        - Prevents duplicate experiment IDs by using atomic timestamps.
        - Preserves previous historical records (append/update policy).
        - Automatically restores network settings to unshaped upon completion or abort.
        """
        qualities = fixed_qualities or [70]
        net_profiles = profiles if profiles is not None else [None]

        # Calculate total trials
        trials_per_item = len(qualities) if quality_strategy == "fixed" else 1
        total_trials = len(items) * len(net_profiles) * trials_per_item

        completed = 0
        failed = 0
        records: List[ExperimentRecord] = []
        current_step = 0

        logger.info(
            f"Starting batch experiment: {len(items)} items, strategy='{quality_strategy}', "
            f"{len(net_profiles)} profile(s), {total_trials} total scheduled trials."
        )

        try:
            for prof in net_profiles:
                if stop_requested and stop_requested():
                    break

                # Apply network traffic shaping profile if specified
                if prof:
                    self.network_controller.apply_profile(prof)
                else:
                    self.network_controller.reset()

                for item_idx, item in enumerate(items):
                    if stop_requested and stop_requested():
                        break

                    if quality_strategy == "fixed":
                        for q in qualities:
                            if stop_requested and stop_requested():
                                break

                            current_step += 1
                            step_desc = f"[{current_step}/{total_trials}] {item.image_id} (Q{q})"
                            success, rec, msg = self.run_single_trial(
                                item=item,
                                quality=q,
                                profile=prof,
                                experiment_id_prefix="EXP",
                            )
                            if success and rec:
                                completed += 1
                                records.append(rec)
                            else:
                                failed += 1

                            if progress_callback:
                                progress_callback(current_step, total_trials, step_desc, rec)

                    else:  # Adaptive Mode
                        current_step += 1
                        step_desc = f"[{current_step}/{total_trials}] {item.image_id} (Adaptive)"
                        success, rec, msg = self.run_adaptive_trial(
                            item=item,
                            profile=prof,
                            use_ewma=True,
                        )
                        if success and rec:
                            completed += 1
                            records.append(rec)
                        else:
                            failed += 1

                        if progress_callback:
                            progress_callback(current_step, total_trials, step_desc, rec)

        finally:
            # Ensure network is reset back to unshaped state
            self.network_controller.reset()

        summary = f"Batch finished: {completed} succeeded, {failed} failed out of {current_step} attempted."
        logger.info(summary)
        return completed, failed, records, summary
