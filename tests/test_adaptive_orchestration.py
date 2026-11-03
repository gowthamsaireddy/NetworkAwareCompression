"""Unit and Integration Tests for Adaptive Controller and Experiment Orchestration.

Verifies:
1. Adaptive quality controller decision thresholds and boundary conditions.
2. Dynamic throughput estimation with EWMA smoothing.
3. Fallback and input source prioritization logic.
4. End-to-end ExperimentOrchestrator execution, verified ACK checks, and SQLite/CSV logging.
"""

import os
from pathlib import Path
import shutil
import socket
import tempfile
import threading
import time
import unittest

import numpy as np
from PIL import Image

from common.results_manager import ResultsManager
from receiver.tcp_server import TCPReceiverServer
from sender.adaptive_controller import (
    AdaptiveController,
    AdaptiveDecision,
    AdaptiveThresholds,
)
from sender.dataset_manager import DatasetItem
from sender.experiment_orchestrator import ExperimentOrchestrator
from sender.network_control import NetworkController, NetworkProfile
from sender.tcp_client import TCPTransmissionClient
from shared.constants import ACK_OK
from shared.metadata import AcknowledgmentMessage
from shared.protocol import recv_image_frame, send_ack_frame


class TestAdaptiveController(unittest.TestCase):
    """Tests provisional adaptive quality controller logic and EWMA calculations."""

    def setUp(self):
        self.controller = AdaptiveController(
            thresholds=AdaptiveThresholds(
                q90_min_kbps=10000.0,
                q70_min_kbps=2000.0,
                q50_min_kbps=512.0,
            ),
            ewma_alpha=0.3,
        )

    def test_provisional_threshold_decisions(self):
        # Tier 1: >= 10 Mbps -> Q90
        d1 = self.controller.determine_quality(15000.0)
        self.assertEqual(d1.quality, 90)
        self.assertIn("Q90", d1.explanation)

        d1_edge = self.controller.determine_quality(10000.0)
        self.assertEqual(d1_edge.quality, 90)

        # Tier 2: >= 2 Mbps and < 10 Mbps -> Q70
        d2 = self.controller.determine_quality(5000.0)
        self.assertEqual(d2.quality, 70)
        self.assertIn("Q70", d2.explanation)

        d2_edge = self.controller.determine_quality(2000.0)
        self.assertEqual(d2_edge.quality, 70)

        # Tier 3: >= 512 Kbps and < 2 Mbps -> Q50
        d3 = self.controller.determine_quality(1000.0)
        self.assertEqual(d3.quality, 50)
        self.assertIn("Q50", d3.explanation)

        d3_edge = self.controller.determine_quality(512.0)
        self.assertEqual(d3_edge.quality, 50)

        # Tier 4: < 512 Kbps -> Q30
        d4 = self.controller.determine_quality(511.9)
        self.assertEqual(d4.quality, 30)
        self.assertIn("Q30", d4.explanation)

        d4_low = self.controller.determine_quality(64.0)
        self.assertEqual(d4_low.quality, 30)

    def test_custom_thresholds(self):
        custom = AdaptiveController(
            thresholds=AdaptiveThresholds(
                q90_min_kbps=20000.0,
                q70_min_kbps=5000.0,
                q50_min_kbps=1000.0,
            )
        )
        self.assertEqual(custom.determine_quality(15000.0).quality, 70)
        self.assertEqual(custom.determine_quality(25000.0).quality, 90)
        self.assertEqual(custom.determine_quality(800.0).quality, 30)

    def test_ewma_throughput_smoothing(self):
        self.assertIsNone(self.controller.estimated_bandwidth_kbps)

        # First measurement seeds the estimator
        est1 = self.controller.update_measured_throughput(1000.0)
        self.assertAlmostEqual(est1, 1000.0, places=3)
        self.assertAlmostEqual(self.controller.estimated_bandwidth_kbps, 1000.0, places=3)

        # Second measurement applies alpha=0.3: 0.3 * 2000 + 0.7 * 1000 = 600 + 700 = 1300.0
        est2 = self.controller.update_measured_throughput(2000.0)
        self.assertAlmostEqual(est2, 1300.0, places=3)

        # Non-positive values do not corrupt estimate
        est_invalid = self.controller.update_measured_throughput(0.0)
        self.assertAlmostEqual(est_invalid, 1300.0, places=3)

        # Reset clears estimate
        self.controller.reset_estimation()
        self.assertIsNone(self.controller.estimated_bandwidth_kbps)

    def test_fallback_input_prioritization(self):
        # 1. No inputs -> default fallback (10 Mbps -> Q90)
        d_default = self.controller.evaluate_with_fallback()
        self.assertEqual(d_default.quality, 90)
        self.assertIn("Fallback", d_default.mode)

        # 2. Manual input provided
        d_manual = self.controller.evaluate_with_fallback(manual_kbps=800.0)
        self.assertEqual(d_manual.quality, 50)
        self.assertIn("Manual", d_manual.mode)

        # 3. Profile provided overrides manual
        d_profile = self.controller.evaluate_with_fallback(
            manual_kbps=800.0,
            profile_bw_bps=500_000.0,  # 500 Kbps (< 512 Kbps) -> Q30
        )
        self.assertEqual(d_profile.quality, 30)
        self.assertIn("Profile", d_profile.mode)

        # 4. EWMA requested and available overrides profile and manual
        self.controller.update_measured_throughput(12000.0)  # EWMA: 12 Mbps
        d_ewma = self.controller.evaluate_with_fallback(
            manual_kbps=800.0,
            profile_bw_bps=500_000.0,
            use_ewma=True,
        )
        self.assertEqual(d_ewma.quality, 90)
        self.assertIn("EWMA", d_ewma.mode)


class TestExperimentOrchestrator(unittest.TestCase):
    """Tests end-to-end trial orchestration, TCP transmission, and results recording."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="nac_test_orch_")
        cls.db_path = Path(cls.temp_dir) / "test_experiments.sqlite"
        cls.csv_path = Path(cls.temp_dir) / "test_experiments.csv"
        cls.save_dir = Path(cls.temp_dir) / "received_images"
        cls.save_dir.mkdir(parents=True, exist_ok=True)

        # Find a free dynamic port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            cls.server_port = s.getsockname()[1]

        # Start live TCPReceiverServer
        cls.server = TCPReceiverServer(host="127.0.0.1", port=cls.server_port, save_dir=cls.save_dir, client_timeout=5.0)
        started, msg = cls.server.start()
        assert started, f"Could not start server on port {cls.server_port}: {msg}"
        time.sleep(0.1)

        # Create a test ultrasound-like image
        cls.img_path = Path(cls.temp_dir) / "benign (1).png"
        arr = np.random.RandomState(42).randint(30, 220, (256, 256), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        img.save(cls.img_path)

        cls.item = DatasetItem(
            image_id="benign (1)",
            category="benign",
            image_path=cls.img_path,
            mask_paths=[],
            file_size_bytes=cls.img_path.stat().st_size,
            width=256,
            height=256,
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        time.sleep(0.05)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_run_adaptive_trial_success(self):
        results_mgr = ResultsManager(db_path=self.db_path, csv_path=self.csv_path)
        tcp_client = TCPTransmissionClient(host="127.0.0.1", port=self.server_port, timeout=5.0)
        net_ctrl = NetworkController()
        adapt_ctrl = AdaptiveController()

        orchestrator = ExperimentOrchestrator(
            tcp_client=tcp_client,
            network_controller=net_ctrl,
            adaptive_controller=adapt_ctrl,
            results_manager=results_mgr,
        )

        try:
            # Profile: 2 Mbps -> should select Q70
            prof = NetworkProfile(
                name="2 Mbps",
                bandwidth="2mbit",
                delay_ms=10,
                loss_percent=0.0,
            )

            success, record, msg = orchestrator.run_adaptive_trial(
                item=self.item,
                profile=prof,
            )

            self.assertTrue(success, f"Trial failed: {msg}")
            self.assertIsNotNone(record)
            self.assertEqual(record.compression_quality, 70)
            self.assertEqual(record.image_id, "benign (1)")
            self.assertTrue(record.checksum_verified)
            self.assertIn("Adaptive", record.notes)
            self.assertGreater(record.transmission_time_sec, 0.0)
            self.assertGreater(record.measured_throughput_kbps, 0.0)

            # Verify SQLite record persistence
            loaded_records = results_mgr.fetch_all()
            self.assertEqual(len(loaded_records), 1)
            self.assertEqual(loaded_records[0]["experiment_id"], record.experiment_id)
            self.assertEqual(loaded_records[0]["compression_quality"], 70)

            # Verify CSV persistence
            self.assertTrue(self.csv_path.exists())
            with open(self.csv_path, "r", encoding="utf-8") as f:
                csv_content = f.read()
                self.assertIn(record.experiment_id, csv_content)
                self.assertIn("benign (1)", csv_content)

            # Verify EWMA was updated in controller
            self.assertIsNotNone(adapt_ctrl.estimated_bandwidth_kbps)
            self.assertGreater(adapt_ctrl.estimated_bandwidth_kbps, 0.0)

        finally:
            tcp_client.disconnect()


if __name__ == "__main__":
    unittest.main()
