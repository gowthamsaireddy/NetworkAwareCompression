"""Tests for Phase 4: Network Emulation Profiles, Theoretical Calculation, and Experiment Logging."""

from pathlib import Path
import tempfile
import unittest

from common.results_manager import ExperimentRecord, ResultsManager
from sender.network_control import (
    NetworkController,
    NetworkProfile,
    PRESET_PROFILES,
    compute_theoretical_transmission_time,
    parse_bandwidth_to_bps,
)


class TestNetworkEmulationProfiles(unittest.TestCase):
    """Tests preset profiles, rate conversion, and theoretical transmission calculations."""

    def test_preset_profiles_exist(self):
        profile_names = [p.name for p in PRESET_PROFILES]
        self.assertEqual(len(PRESET_PROFILES), 4)
        self.assertTrue(any("10 Mbps" in n for n in profile_names))
        self.assertTrue(any("2 Mbps" in n for n in profile_names))
        self.assertTrue(any("512 Kbps" in n for n in profile_names))
        self.assertTrue(any("128 Kbps" in n for n in profile_names))

    def test_bandwidth_parsing(self):
        self.assertEqual(parse_bandwidth_to_bps("10mbit"), 10_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("2mbit"), 2_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("512kbit"), 512_000.0)
        self.assertEqual(parse_bandwidth_to_bps("128kbit"), 128_000.0)
        self.assertEqual(parse_bandwidth_to_bps("1gbps"), 1_000_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("invalid"), 0.0)

    def test_theoretical_transmission_time(self):
        # 100,000 bytes over 10 Mbps (10,000,000 bps)
        # Time = (100,000 * 8) / 10,000,000 = 800,000 / 10,000,000 = 0.080 s
        t_10m = compute_theoretical_transmission_time(100_000, 10_000_000.0)
        self.assertAlmostEqual(t_10m, 0.080, places=4)

        # 100,000 bytes over 128 Kbps (128,000 bps)
        # Time = (100,000 * 8) / 128,000 = 800,000 / 128,000 = 6.250 s
        t_128k = compute_theoretical_transmission_time(100_000, 128_000.0)
        self.assertAlmostEqual(t_128k, 6.250, places=4)

        # Zero bandwidth returns None
        self.assertIsNone(compute_theoretical_transmission_time(100_000, 0.0))

    def test_network_controller_interface_detection(self):
        controller = NetworkController()
        interfaces = controller.detect_interfaces()
        self.assertIsInstance(interfaces, list)
        self.assertGreater(len(interfaces), 0)

    def test_network_controller_verification_policy(self):
        """Must never claim traffic shaping is active without verification."""
        controller = NetworkController()
        status = controller.verify_status()
        self.assertIsInstance(status, str)

        if not controller.is_linux:
            success, msg = controller.apply_profile(PRESET_PROFILES[0])
            self.assertFalse(success)
            self.assertIn("only supported on Linux", msg)
            self.assertIsNone(controller.current_profile)


class TestExperimentalLogging(unittest.TestCase):
    """Tests SQLite and CSV persistence of experimental telemetry and theoretical vs measured metrics."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_exp.sqlite"
        self.csv_path = Path(self.temp_dir.name) / "test_exp.csv"
        self.mgr = ResultsManager(db_path=self.db_path, csv_path=self.csv_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_record_and_retrieve_experiment(self):
        record = ExperimentRecord(
            experiment_id="EXP_BENCH_001",
            timestamp="2026-11-04 12:00:00",
            image_id="benign/benign (1)",
            compression_quality=70,
            original_file_size=200000,
            compressed_file_size=40000,
            compression_ratio=5.0,
            psnr=36.42,
            ssim=0.9412,
            bandwidth_limit="10mbit",
            packet_loss_rate=0.0,
            network_delay_ms=5.0,
            theoretical_transmission_time_sec=0.032,
            transmission_time_sec=0.041,
            measured_throughput_kbps=7804.88,
            checksum_verified=True,
            dice_coefficient=None,
            iou=None,
            notes="Physical trial",
        )

        self.mgr.record_experiment(record)

        # Verify SQLite retrieval
        rows = self.mgr.fetch_all()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["experiment_id"], "EXP_BENCH_001")
        self.assertEqual(r["image_id"], "benign/benign (1)")
        self.assertEqual(r["compression_quality"], 70)
        self.assertAlmostEqual(r["psnr"], 36.42, places=2)
        self.assertAlmostEqual(r["theoretical_transmission_time_sec"], 0.032, places=3)
        self.assertAlmostEqual(r["transmission_time_sec"], 0.041, places=3)
        self.assertEqual(r["checksum_verified"], 1)

        # Verify CSV contents
        self.assertTrue(self.csv_path.exists())
        with open(self.csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), 2)  # Header + 1 record
        self.assertIn("EXP_BENCH_001", lines[1])
        self.assertIn("36.42", lines[1])


if __name__ == "__main__":
    unittest.main()
