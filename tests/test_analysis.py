"""Tests for Experimental Results Analysis and Rate-Distortion Plot Generator."""

from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from common.results_manager import ExperimentRecord, ResultsManager
from experiments.analyze_results import (
    generate_latency_comparison_plot,
    generate_rate_distortion_plot,
    generate_segmentation_performance_plot,
    load_experiment_data,
    print_summary_table,
)


class TestResultsAnalysis(unittest.TestCase):
    """Verifies scientific plotting, data grouping, and Zero Dummy-Data guardrails."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="nac_test_analysis_")
        self.db_path = Path(self.temp_dir) / "test_exp.sqlite"
        self.csv_path = Path(self.temp_dir) / "test_exp.csv"
        self.plots_dir = Path(self.temp_dir) / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_manager = ResultsManager(db_path=self.db_path, csv_path=self.csv_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_database_handling(self):
        df = load_experiment_data(self.db_path)
        self.assertTrue(df.empty)

        # Plot generators must return False without crashing when data is absent
        self.assertFalse(generate_rate_distortion_plot(df, self.plots_dir / "rd.png"))
        self.assertFalse(generate_segmentation_performance_plot(df, self.plots_dir / "seg.png"))
        self.assertFalse(generate_latency_comparison_plot(df, self.plots_dir / "lat.png"))

    def test_populated_plots_generation(self):
        # Insert genuine records across Q30, Q50, Q70, Q90
        for i, q in enumerate([30, 50, 70, 90]):
            rec = ExperimentRecord(
                experiment_id=f"EXP_{q}_{i}",
                timestamp=f"2026-11-04 12:0{i}:00",
                image_id=f"benign ({i+1})",
                compression_quality=q,
                original_file_size=100000,
                compressed_file_size=100000 // (100 - q),
                compression_ratio=(100 - q) / 1.0,
                psnr=25.0 + (q / 5.0),
                ssim=0.70 + (q / 300.0),
                bandwidth_limit="2mbit",
                packet_loss_rate=0.5,
                network_delay_ms=20.0,
                theoretical_transmission_time_sec=0.08,
                transmission_time_sec=0.10 + (0.01 * i),
                measured_throughput_kbps=1950.0,
                checksum_verified=True,
                dice_coefficient=0.75 + (q / 400.0),
                iou=0.65 + (q / 450.0),
                notes="Analysis Test Run",
            )
            self.results_manager.record_experiment(rec)

        df = load_experiment_data(self.db_path)
        self.assertEqual(len(df), 4)

        # Generate plots
        rd_path = self.plots_dir / "rate_distortion_curves.png"
        seg_path = self.plots_dir / "segmentation_accuracy_curves.png"
        lat_path = self.plots_dir / "latency_vs_bandwidth.png"

        self.assertTrue(generate_rate_distortion_plot(df, rd_path))
        self.assertTrue(generate_segmentation_performance_plot(df, seg_path))
        self.assertTrue(generate_latency_comparison_plot(df, lat_path))

        self.assertTrue(rd_path.exists())
        self.assertGreater(rd_path.stat().st_size, 1000)
        self.assertTrue(seg_path.exists())
        self.assertGreater(seg_path.stat().st_size, 1000)
        self.assertTrue(lat_path.exists())
        self.assertGreater(lat_path.stat().st_size, 1000)

        # Print table should execute cleanly
        print_summary_table(df)


if __name__ == "__main__":
    unittest.main()
