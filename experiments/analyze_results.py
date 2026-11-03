"""Research Results Analyzer & Rate-Distortion Plot Generator.

Reads genuine experimental logs from SQLite/CSV and generates publication-quality plots:
1. Rate-Distortion Curves (PSNR & SSIM vs. JPEG Quality).
2. Clinical Segmentation Degradation (Dice & IoU vs. JPEG Quality).
3. Network Latency: Theoretical vs. Measured Wire Time across Profiles.
4. Comprehensive Metric Summary Table (Mean ± Std Dev) for academic publications.

Strictly adheres to Zero Dummy-Data Policy:
- Excludes uncalculated or empty fields.
- Reports clear diagnostic notices if data is insufficient rather than generating fabricated graphs.
"""

import argparse
from pathlib import Path
import sqlite3
import sys

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import EXPERIMENTS_DIR


def load_experiment_data(db_path: Path) -> pd.DataFrame:
    """Loads experimental records from SQLite into a Pandas DataFrame."""
    if not db_path.exists():
        print(f"[!] Database file does not exist: {db_path}")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM experiments ORDER BY id ASC", conn)
        return df
    finally:
        conn.close()


def generate_rate_distortion_plot(df: pd.DataFrame, output_path: Path) -> bool:
    """Generates PSNR and SSIM vs. Compression Quality curve."""
    valid_df = df.dropna(subset=["compression_quality", "psnr", "ssim"])
    if valid_df.empty:
        print("[-] Skipping Rate-Distortion plot: No valid PSNR/SSIM measurements recorded.")
        return False

    grouped = valid_df.groupby("compression_quality").agg({
        "psnr": ["mean", "std"],
        "ssim": ["mean", "std"],
        "compression_ratio": ["mean", "std"],
    }).reset_index()

    qualities = grouped["compression_quality"].values
    psnr_mean = grouped["psnr"]["mean"].values
    psnr_std = grouped["psnr"]["std"].fillna(0.0).values
    ssim_mean = grouped["ssim"]["mean"].values
    ssim_std = grouped["ssim"]["std"].fillna(0.0).values

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    color_psnr = "#0284c7"
    ax1.set_xlabel("JPEG Compression Quality Factor (Q)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Peak Signal-to-Noise Ratio (PSNR in dB)", color=color_psnr, fontsize=11, fontweight="bold")
    ax1.errorbar(qualities, psnr_mean, yerr=psnr_std, fmt="-o", color=color_psnr, capsize=5, label="PSNR (dB)", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color_psnr)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_ssim = "#10b981"
    ax2.set_ylabel("Structural Similarity Index (SSIM)", color=color_ssim, fontsize=11, fontweight="bold")
    ax2.errorbar(qualities, ssim_mean, yerr=ssim_std, fmt="-s", color=color_ssim, capsize=5, label="SSIM", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color_ssim)

    plt.title("Ultrasound Image Rate-Distortion Profile Across JPEG Quality Factors", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved Rate-Distortion plot to {output_path}")
    return True


def generate_segmentation_performance_plot(df: pd.DataFrame, output_path: Path) -> bool:
    """Generates Dice and IoU vs. Compression Quality curve."""
    valid_df = df.dropna(subset=["compression_quality", "dice_coefficient", "iou"])
    if valid_df.empty:
        print("[-] Skipping Segmentation Performance plot: No genuine Dice/IoU evaluations recorded yet.")
        return False

    grouped = valid_df.groupby("compression_quality").agg({
        "dice_coefficient": ["mean", "std"],
        "iou": ["mean", "std"],
    }).reset_index()

    qualities = grouped["compression_quality"].values
    dice_mean = grouped["dice_coefficient"]["mean"].values
    dice_std = grouped["dice_coefficient"]["std"].fillna(0.0).values
    iou_mean = grouped["iou"]["mean"].values
    iou_std = grouped["iou"]["std"].fillna(0.0).values

    plt.figure(figsize=(8, 5), dpi=300)
    plt.errorbar(qualities, dice_mean, yerr=dice_std, fmt="-o", color="#8b5cf6", capsize=5, label="Dice Similarity Coefficient", linewidth=2)
    plt.errorbar(qualities, iou_mean, yerr=iou_std, fmt="-^", color="#f59e0b", capsize=5, label="Intersection over Union (IoU)", linewidth=2)

    plt.xlabel("JPEG Compression Quality Factor (Q)", fontsize=11, fontweight="bold")
    plt.ylabel("Segmentation Accuracy Metric [0.0 - 1.0]", fontsize=11, fontweight="bold")
    plt.title("U-Net Lesion Segmentation Robustness Under Lossy Compression", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved Segmentation Performance plot to {output_path}")
    return True


def generate_latency_comparison_plot(df: pd.DataFrame, output_path: Path) -> bool:
    """Generates Theoretical vs. Measured Transmission Latency comparison."""
    valid_df = df.dropna(subset=["bandwidth_limit", "transmission_time_sec"])
    # Filter out unshaped or zero
    valid_df = valid_df[valid_df["bandwidth_limit"] != "unshaped"]
    if valid_df.empty:
        print("[-] Skipping Latency comparison plot: No shaped network trials recorded.")
        return False

    grouped = valid_df.groupby("bandwidth_limit").agg({
        "transmission_time_sec": ["mean", "std"],
        "theoretical_transmission_time_sec": "mean",
    }).reset_index()

    bw_labels = grouped["bandwidth_limit"].values
    measured_mean = grouped["transmission_time_sec"]["mean"].values
    measured_std = grouped["transmission_time_sec"]["std"].fillna(0.0).values
    theo_mean = grouped["theoretical_transmission_time_sec"]["mean"].values

    x = np.arange(len(bw_labels))
    width = 0.35

    plt.figure(figsize=(8, 5), dpi=300)
    plt.bar(x - width / 2, theo_mean, width, label="Theoretical Wire Time (s)", color="#94a3b8")
    plt.bar(x + width / 2, measured_mean, width, yerr=measured_std, capsize=4, label="Measured Transmission Time (s)", color="#3b82f6")

    plt.xlabel("Emulated Network Bandwidth Profile", fontsize=11, fontweight="bold")
    plt.ylabel("Transmission Latency (Seconds)", fontsize=11, fontweight="bold")
    plt.title("Theoretical Wire Transit vs. Measured Socket Transmission Latency", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, bw_labels)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved Latency comparison plot to {output_path}")
    return True


def print_summary_table(df: pd.DataFrame) -> None:
    """Prints a structured academic summary table of all recorded experiments."""
    if df.empty:
        print("[!] No experimental records found.")
        return

    print("\n" + "=" * 90)
    print(" " * 25 + "EXPERIMENTAL RESULTS SUMMARY TABLE")
    print("=" * 90)

    grouped = df.groupby("compression_quality").agg({
        "id": "count",
        "original_file_size": "mean",
        "compressed_file_size": "mean",
        "compression_ratio": "mean",
        "psnr": "mean",
        "ssim": "mean",
        "transmission_time_sec": "mean",
        "measured_throughput_kbps": "mean",
        "dice_coefficient": "mean",
        "iou": "mean",
    }).rename(columns={"id": "Trials"})

    print(grouped.to_string())
    print("=" * 90)
    print(f"Total Logged Trials: {len(df)}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze Network-Aware Compression experiment logs.")
    parser.add_argument(
        "--db",
        type=Path,
        default=EXPERIMENTS_DIR / "experiments.sqlite",
        help="Path to experiments.sqlite",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=EXPERIMENTS_DIR / "plots",
        help="Output directory for generated plots",
    )
    args = parser.parse_args()

    df = load_experiment_data(args.db)
    if df.empty:
        print("No experiments recorded yet in the database.")
        sys.exit(0)

    print_summary_table(df)
    generate_rate_distortion_plot(df, args.outdir / "rate_distortion_curves.png")
    generate_segmentation_performance_plot(df, args.outdir / "segmentation_accuracy_curves.png")
    generate_latency_comparison_plot(df, args.outdir / "latency_vs_bandwidth.png")


if __name__ == "__main__":
    main()
