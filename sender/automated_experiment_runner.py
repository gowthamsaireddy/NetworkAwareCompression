"""Automated Experiment Runner for One-Click Transmission Workflow.

Executes the full automated cross-product matrix:
Selected Test Images (10) × Supported Compression Configurations (8) × Predefined Network Profiles (6)
Total: 480 end-to-end transmissions over TCP link with ACK verification and unique run telemetry.

Compression Configurations Included:
1. Uniform JPEG Q30 (Aggressive JPEG across entire image)
2. Uniform JPEG Q50 (Medium JPEG across entire image)
3. Uniform JPEG Q70 (Balanced JPEG across entire image)
4. Uniform JPEG Q90 (High-Fidelity JPEG across entire image)
5. Region-Adaptive (ROI Q90 + BG Q20) [Canonical ML lesion-preservation]
6. Region-Adaptive (ROI Q90 + BG Q30) [ROI preserved, background at Q30 JPEG]
7. Region-Adaptive (ROI Q90 + BG Q50) [ROI preserved, background at Q50 JPEG]
8. Region-Adaptive (ROI Q90 + BG Q70) [ROI preserved, background at Q70 JPEG]

Network Emulation Profiles Included:
1. Hospital LAN (100 Mbps, 2 ms delay, 0.5 ms jitter, 0.01% loss)
2. Broadband (50 Mbps, 20 ms delay, 5 ms jitter, 0.1% loss)
3. 4G LTE (10 Mbps, 40 ms delay, 15 ms jitter, 0.5% loss)
4. 3G Cellular (2 Mbps, 100 ms delay, 30 ms jitter, 1.0% loss)
5. Rural Link (512 Kbps, 150 ms delay, 40 ms jitter, 2.0% loss)
6. Satellite (10 Mbps, 300 ms delay, 30 ms jitter, 0.5% loss)
"""

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from common.config import DATA_DIR, EXPERIMENTS_DIR
from common.logging_config import setup_logger
from common.results_manager import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    ExperimentRecord,
    ResultsManager,
)
from receiver.segmentation_metrics import get_canonical_busi_id
from sender.compression import CompressionResult, ImageCompressor
from sender.dataset_manager import DatasetItem, DatasetManager
from sender.network_control import (
    PRESET_PROFILES,
    NetworkController,
    NetworkProfile,
    compute_theoretical_transmission_time,
)
from sender.region_adaptive_compressor import RegionAdaptiveCompressor, RegionAdaptiveResult
from sender.tcp_client import TCPTransmissionClient, TransmissionReport
from shared.constants import COMPRESSION_QUALITIES, PROTOCOL_VERSION
from shared.metadata import TransmissionMetadata
from shared.protocol import compute_sha256

logger = setup_logger("AutomatedExperimentRunner")


@dataclass
class CompressionConfig:
    """Represents a specific compression methodology and parameter configuration."""

    config_id: str
    display_name: str
    method: str  # "NORMAL" or "REGION_ADAPTIVE"
    quality: int  # JPEG quality for NORMAL, ROI quality for REGION_ADAPTIVE
    bg_quality: int = 0  # Background JPEG quality for REGION_ADAPTIVE
    margin_px: int = 0  # Margin expansion in pixels for REGION_ADAPTIVE
    roi_mode: str = "ML_MODEL"  # "ML_MODEL" or "GT_MASK"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_supported_compression_configs() -> List[CompressionConfig]:
    """Returns every implemented and supported compression configuration in the project.

    Includes:
    - JPEG compression at all 4 available quality levels (Q30, Q50, Q70, Q90)
    - ROI-based compression alone (ROI Q90 + BG Q20 baseline)
    - JPEG compression combined with ROI-based compression (ROI Q90 + BG Q30, Q50, Q70)
    """
    return [
        # 1. Uniform JPEG compression alone across available quality levels
        CompressionConfig(
            config_id="JPEG_Q30",
            display_name="Uniform JPEG Q30",
            method="NORMAL",
            quality=30,
            description="Uniform JPEG compression across entire ultrasound frame at quality 30",
        ),
        CompressionConfig(
            config_id="JPEG_Q50",
            display_name="Uniform JPEG Q50",
            method="NORMAL",
            quality=50,
            description="Uniform JPEG compression across entire ultrasound frame at quality 50",
        ),
        CompressionConfig(
            config_id="JPEG_Q70",
            display_name="Uniform JPEG Q70",
            method="NORMAL",
            quality=70,
            description="Uniform JPEG compression across entire ultrasound frame at quality 70",
        ),
        CompressionConfig(
            config_id="JPEG_Q90",
            display_name="Uniform JPEG Q90",
            method="NORMAL",
            quality=90,
            description="Uniform JPEG compression across entire ultrasound frame at quality 90",
        ),
        # 2. ROI-based compression alone (ML-preservation with standard background)
        CompressionConfig(
            config_id="ROI_Q90_BG20",
            display_name="Region-Adaptive (ROI Q90 + BG Q20)",
            method="REGION_ADAPTIVE",
            quality=90,
            bg_quality=20,
            margin_px=15,
            roi_mode="ML_MODEL",
            description="ML ROI preserved at Q90 (+15px margin, feathered), background compressed at Q20",
        ),
        # 3. JPEG compression combined with ROI-based compression
        CompressionConfig(
            config_id="ROI_Q90_BG30",
            display_name="Region-Adaptive (ROI Q90 + BG Q30)",
            method="REGION_ADAPTIVE",
            quality=90,
            bg_quality=30,
            margin_px=15,
            roi_mode="ML_MODEL",
            description="ML ROI preserved at Q90 (+15px margin), background compressed at Q30 JPEG",
        ),
        CompressionConfig(
            config_id="ROI_Q90_BG50",
            display_name="Region-Adaptive (ROI Q90 + BG Q50)",
            method="REGION_ADAPTIVE",
            quality=90,
            bg_quality=50,
            margin_px=15,
            roi_mode="ML_MODEL",
            description="ML ROI preserved at Q90 (+15px margin), background compressed at Q50 JPEG",
        ),
        CompressionConfig(
            config_id="ROI_Q90_BG70",
            display_name="Region-Adaptive (ROI Q90 + BG Q70)",
            method="REGION_ADAPTIVE",
            quality=90,
            bg_quality=70,
            margin_px=15,
            roi_mode="ML_MODEL",
            description="ML ROI preserved at Q90 (+15px margin), background compressed at Q70 JPEG",
        ),
    ]


def load_test_dataset_items(count: int = 10, project_root: Optional[Path] = None) -> List[DatasetItem]:
    """Loads exactly `count` items from the held-out test split in data/splits.json."""
    root = project_root or Path(__file__).resolve().parent.parent

    # Try standard split locations
    candidates = [
        root / "data" / "splits.json",
        root / "data" / "BUSI" / "splits.json",
    ]

    splits_file = None
    for c in candidates:
        if c.exists():
            splits_file = c
            break

    if not splits_file:
        logger.warning("splits.json not found. Scanning dataset directly for test items...")
        dm = DatasetManager(root / "data" / "BUSI")
        items = dm.scan_dataset()
        return items[:count]

    with open(splits_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_raw = data.get("test", [])
    if not test_raw:
        raise ValueError(f"No test items found in splits file {splits_file}")

    items: List[DatasetItem] = []
    for d in test_raw:
        img_p = root / d["image_path"]
        if not img_p.exists():
            # Try alternate path if slashes differ
            norm_rel = d["image_path"].replace("\\", "/")
            img_p = root / norm_rel
            if not img_p.exists():
                logger.warning(f"Test image path does not exist on disk: {img_p}")
                continue

        mask_paths = [root / p.replace("\\", "/") for p in d.get("mask_paths", [])]

        item = DatasetItem(
            image_id=str(d["image_id"]),
            image_path=img_p,
            category=str(d.get("category", "unknown")),
            mask_paths=mask_paths,
            file_size_bytes=int(d.get("file_size_bytes", img_p.stat().st_size if img_p.exists() else 0)),
            width=int(d.get("width", 256)),
            height=int(d.get("height", 256)),
        )
        items.append(item)
        if len(items) >= count:
            break

    logger.info(f"Loaded {len(items)} held-out test images from {splits_file}")
    return items


class AutomatedExperimentRunner:
    """Orchestrates the complete One-Click Automated Transmission sequence."""

    def __init__(
        self,
        tcp_client: TCPTransmissionClient,
        network_controller: Optional[NetworkController] = None,
        results_manager: Optional[ResultsManager] = None,
        output_base_dir: Optional[Path] = None,
    ):
        self.tcp_client = tcp_client
        self.network_controller = network_controller or NetworkController()
        self.results_manager = results_manager or ResultsManager()
        self.output_base_dir = output_base_dir or (EXPERIMENTS_DIR / "runs")
        self.region_adaptive_compressor = RegionAdaptiveCompressor()

    def run_all_experiments(
        self,
        items: List[DatasetItem],
        compression_configs: Optional[List[CompressionConfig]] = None,
        network_profiles: Optional[List[NetworkProfile]] = None,
        progress_callback: Optional[
            Callable[[int, int, str, str, str, int, int, Optional[ExperimentRecord]], None]
        ] = None,
        stop_requested: Optional[Callable[[], bool]] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Executes the full automated matrix of experiments.

        Args:
            items: Exactly selected test images (typically 10 images).
            compression_configs: List of compression configs (defaults to all 8 supported).
            network_profiles: List of network profiles (defaults to all 6 PRESET_PROFILES).
            progress_callback: Callback receiving (current, total, image_id, config_name, profile_name, success_count, fail_count, record).
            stop_requested: Callable returning True if user clicks Stop.

        Returns:
            Tuple of (is_completed, summary_message, run_manifest_dict).
        """
        configs = compression_configs or get_supported_compression_configs()
        profiles = network_profiles or list(PRESET_PROFILES)

        total_transmissions = len(items) * len(configs) * len(profiles)
        now_dt = datetime.now()
        run_id = f"RUN_{now_dt.strftime('%Y%m%d_%H%M%S')}"

        # 1. Create unique run output directory on sender
        run_dir = self.output_base_dir / run_id
        transmitted_dir = run_dir / "transmitted_images"
        transmitted_dir.mkdir(parents=True, exist_ok=True)

        run_csv_path = run_dir / "experiment_records.csv"
        failed_json_path = run_dir / "failed_transmissions.json"
        summary_json_path = run_dir / "run_summary.json"

        logger.info(
            f"=== Starting Automated Transmission Sequence: Run ID '{run_id}' ===\n"
            f"• Images: {len(items)}\n"
            f"• Compression Configs: {len(configs)}\n"
            f"• Network Profiles: {len(profiles)}\n"
            f"• Total Planned Transmissions: {total_transmissions}\n"
            f"• Output Directory: {run_dir}"
        )

        completed_count = 0
        failed_count = 0
        current_step = 0
        records: List[ExperimentRecord] = []
        failed_records: List[Dict[str, Any]] = []
        t_run_start = time.perf_counter()

        # Prepare run-specific CSV writer
        csv_file = open(run_csv_path, "w", newline="", encoding="utf-8")
        csv_writer = None

        try:
            # We group by network profile to avoid excessive kernel qdisc re-creation overhead
            for prof in profiles:
                if stop_requested and stop_requested():
                    logger.warning("Stop requested by user before network profile application.")
                    break

                # Apply network profile
                prof_applied, prof_msg = self.network_controller.apply_profile(prof)
                logger.info(f"Applied network profile '{prof.name}': {prof_msg}")

                bw_str = prof.bandwidth
                loss_pct = float(prof.loss_percent)
                delay_ms = float(prof.delay_ms)
                jitter_ms = float(prof.jitter_ms)
                bw_bps = prof.bandwidth_bps
                prof_slug = re.sub(r"[^\w\.-]", "_", prof.name)

                for cfg in configs:
                    if stop_requested and stop_requested():
                        break

                    cfg_slug = re.sub(r"[^\w\.-]", "_", cfg.config_id)

                    for item_idx, item in enumerate(items, 1):
                        if stop_requested and stop_requested():
                            break

                        current_step += 1
                        clean_img_id = re.sub(r"[^\w\s\(\)\.-]", "_", item.image_id).strip()
                        unique_exp_id = f"{run_id}_{cfg_slug}_{prof_slug}_{clean_img_id}"

                        # --- Step A: Compression ---
                        comp_bytes: Optional[bytes] = None
                        orig_size = item.file_size_bytes
                        comp_size = 0
                        comp_ratio = 1.0
                        psnr_val = 0.0
                        ssim_val = 0.0
                        img_w = item.width
                        img_h = item.height
                        model_ver: Optional[str] = None
                        margin_val = 0
                        params_str = ""

                        try:
                            if cfg.method == "REGION_ADAPTIVE":
                                res_adapt = self.region_adaptive_compressor.compress_image(
                                    image_source=item.image_path,
                                    margin_px=cfg.margin_px,
                                    roi_quality=cfg.quality,
                                    bg_quality=cfg.bg_quality,
                                    roi_mode=cfg.roi_mode,
                                )
                                comp_bytes = res_adapt.compressed_bytes
                                orig_size = res_adapt.original_size_bytes
                                comp_size = res_adapt.compressed_size_bytes
                                comp_ratio = res_adapt.compression_ratio
                                psnr_val = res_adapt.psnr
                                ssim_val = res_adapt.ssim
                                img_w = res_adapt.width
                                img_h = res_adapt.height
                                model_ver = res_adapt.model_version
                                margin_val = cfg.margin_px
                                params_str = json.dumps({
                                    "roi_quality": cfg.quality,
                                    "bg_quality": cfg.bg_quality,
                                    "margin_px": cfg.margin_px,
                                    "roi_mode": cfg.roi_mode,
                                })
                            else:
                                res_norm = ImageCompressor.compress_image(
                                    image_source=item.image_path,
                                    quality=cfg.quality,
                                )
                                comp_bytes = res_norm.compressed_bytes
                                orig_size = res_norm.original_size_bytes
                                comp_size = res_norm.compressed_size_bytes
                                comp_ratio = res_norm.compression_ratio
                                psnr_val = res_norm.psnr
                                ssim_val = res_norm.ssim
                                img_w = res_norm.width
                                img_h = res_norm.height
                                model_ver = None
                                margin_val = 0
                                params_str = json.dumps({"quality": cfg.quality})

                        except Exception as e:
                            err_msg = f"Compression failed for {item.image_id} ({cfg.config_id}): {e}"
                            logger.error(err_msg)
                            failed_count += 1
                            failed_records.append({
                                "step": current_step,
                                "image_id": item.image_id,
                                "config": cfg.to_dict(),
                                "profile": prof.name,
                                "error": err_msg,
                            })
                            if progress_callback:
                                progress_callback(
                                    current_step,
                                    total_transmissions,
                                    item.image_id,
                                    cfg.display_name,
                                    prof.name,
                                    completed_count,
                                    failed_count,
                                    None,
                                )
                            continue

                        # --- Step B: Save transmitted compressed image to run output dir ---
                        transmitted_filename = f"{clean_img_id}_{cfg_slug}_{prof_slug}.jpg"
                        transmitted_path = transmitted_dir / transmitted_filename
                        try:
                            with open(transmitted_path, "wb") as f_out:
                                f_out.write(comp_bytes)
                        except Exception as e:
                            logger.warning(f"Could not save copy to run dir: {e}")

                        # --- Step C: Prepare transmission metadata ---
                        sha_val = compute_sha256(comp_bytes)
                        orig_id, _, cat, orig_fn = get_canonical_busi_id(item.image_id)
                        meta = TransmissionMetadata(
                            protocol_version=PROTOCOL_VERSION,
                            experiment_id=unique_exp_id,
                            image_id=item.image_id,
                            original_image_id=orig_id,
                            original_image_filename=orig_fn,
                            compression_quality=cfg.quality,
                            image_width=img_w,
                            image_height=img_h,
                            image_format="JPEG",
                            original_file_size=orig_size,
                            compressed_file_size=comp_size,
                            image_byte_length=comp_size,
                            sha256_checksum=sha_val,
                            compression_mode=cfg.method,
                            region_adaptive_margin=margin_val,
                            compression_model_version=model_ver,
                            image_category=item.category,
                        )

                        # --- Step D: Transmit over TCP and await verified ACK ---
                        report: TransmissionReport = self.tcp_client.send_image(meta, comp_bytes)

                        theo_time = compute_theoretical_transmission_time(comp_size, bw_bps)
                        status_str = STATUS_COMPLETED if report.is_success else STATUS_FAILED

                        record = ExperimentRecord(
                            experiment_id=unique_exp_id,
                            timestamp=report.timestamp,
                            image_id=item.image_id,
                            original_image_id=orig_id,
                            original_image_filename=orig_fn,
                            image_category=item.category,
                            compression_method=cfg.method,
                            compression_quality=cfg.quality,
                            compression_params=params_str,
                            region_adaptive_margin=margin_val,
                            compression_model_version=model_ver,
                            original_file_size=orig_size,
                            compressed_file_size=comp_size,
                            transmitted_bytes=report.bytes_sent,
                            received_bytes=report.received_bytes,
                            compression_ratio=comp_ratio,
                            psnr=psnr_val,
                            ssim=ssim_val,
                            bandwidth_limit=bw_str,
                            packet_loss_rate=loss_pct,
                            network_delay_ms=delay_ms,
                            network_jitter_ms=jitter_ms,
                            network_profile=prof.name,
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
                            ground_truth_status="MATCHED" if report.dice is not None else "UNVERIFIED",
                            experiment_status=status_str,
                            error_message=report.error_message,
                            completion_time=report.timestamp,
                            transmission_mode="AUTOMATED_ALL",
                            notes=f"Auto Run: {run_id} | Config: {cfg.config_id} | Profile: {prof.name}",
                        )

                        # Record in global SQLite & CSV
                        self.results_manager.record_experiment(record)

                        # Write to run-specific CSV
                        rec_dict = record.to_dict()
                        if csv_writer is None:
                            csv_writer = csv.DictWriter(csv_file, fieldnames=list(rec_dict.keys()))
                            csv_writer.writeheader()
                        csv_writer.writerow(rec_dict)
                        csv_file.flush()

                        if report.is_success:
                            completed_count += 1
                            records.append(record)
                        else:
                            failed_count += 1
                            failed_records.append({
                                "step": current_step,
                                "image_id": item.image_id,
                                "experiment_id": unique_exp_id,
                                "config": cfg.to_dict(),
                                "profile": prof.name,
                                "error": report.error_message or f"ACK status: {report.ack_status}",
                            })

                        # Telemetry progress update
                        if progress_callback:
                            progress_callback(
                                current_step,
                                total_transmissions,
                                item.image_id,
                                cfg.display_name,
                                prof.name,
                                completed_count,
                                failed_count,
                                record,
                            )

        finally:
            csv_file.close()

            # Restore network configuration unconditionally
            logger.info("Restoring original network traffic configuration...")
            try:
                self.network_controller.reset()
            except Exception as e:
                logger.warning(f"Error resetting network: {e}")

        t_run_end = time.perf_counter()
        total_duration_sec = max(t_run_end - t_run_start, 0.001)

        # Write failed transmissions log
        if failed_records:
            with open(failed_json_path, "w", encoding="utf-8") as f:
                json.dump(failed_records, f, indent=2)

        # Write comprehensive run summary
        summary_manifest = {
            "run_id": run_id,
            "timestamp_start": now_dt.isoformat(),
            "timestamp_end": datetime.now().isoformat(),
            "duration_seconds": round(total_duration_sec, 2),
            "total_planned_transmissions": total_transmissions,
            "total_attempted_transmissions": current_step,
            "successful_transmissions": completed_count,
            "failed_transmissions": failed_count,
            "selected_test_images": [item.image_id for item in items],
            "compression_configs": [c.to_dict() for c in configs],
            "network_profiles": [p.name for p in profiles],
            "output_directory": str(run_dir),
            "transmitted_images_directory": str(transmitted_dir),
            "records_csv": str(run_csv_path),
            "failed_records_json": str(failed_json_path) if failed_records else None,
            "is_stopped_early": bool(stop_requested and stop_requested()),
        }

        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_manifest, f, indent=2)

        summary_msg = (
            f"Automated Sequence '{run_id}' finished in {total_duration_sec:.1f}s: "
            f"{completed_count} successful, {failed_count} failed out of {current_step}/{total_transmissions} transmissions.\n"
            f"Outputs saved to: {run_dir}"
        )
        logger.info(summary_msg)

        is_completed = (completed_count + failed_count) == total_transmissions
        return is_completed, summary_msg, summary_manifest
