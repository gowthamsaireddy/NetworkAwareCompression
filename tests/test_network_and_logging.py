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
        self.assertEqual(len(PRESET_PROFILES), 6)
        profile_map = {p.name: p for p in PRESET_PROFILES}

        expected_profiles = [
            "Hospital LAN",
            "Broadband",
            "4G LTE",
            "3G Cellular",
            "Rural Link",
            "Satellite",
        ]
        for name in expected_profiles:
            self.assertIn(name, profile_map)

        # 1. Hospital LAN: 100 Mbps, 2 ms, 0.5 ms, 0.01%
        hosp = profile_map["Hospital LAN"]
        self.assertEqual(hosp.bandwidth_bps, 100_000_000.0)
        self.assertEqual(hosp.delay_ms, 2.0)
        self.assertEqual(hosp.jitter_ms, 0.5)
        self.assertEqual(hosp.loss_percent, 0.01)

        # 2. Broadband: 50 Mbps, 20 ms, 5 ms, 0.1%
        broad = profile_map["Broadband"]
        self.assertEqual(broad.bandwidth_bps, 50_000_000.0)
        self.assertEqual(broad.delay_ms, 20.0)
        self.assertEqual(broad.jitter_ms, 5.0)
        self.assertEqual(broad.loss_percent, 0.1)

        # 3. 4G LTE: 10 Mbps, 40 ms, 15 ms, 0.5%
        lte = profile_map["4G LTE"]
        self.assertEqual(lte.bandwidth_bps, 10_000_000.0)
        self.assertEqual(lte.delay_ms, 40.0)
        self.assertEqual(lte.jitter_ms, 15.0)
        self.assertEqual(lte.loss_percent, 0.5)

        # 4. 3G Cellular: 2 Mbps, 100 ms, 30 ms, 1%
        g3 = profile_map["3G Cellular"]
        self.assertEqual(g3.bandwidth_bps, 2_000_000.0)
        self.assertEqual(g3.delay_ms, 100.0)
        self.assertEqual(g3.jitter_ms, 30.0)
        self.assertEqual(g3.loss_percent, 1.0)

        # 5. Rural Link: 512 Kbps, 150 ms, 40 ms, 2%
        rural = profile_map["Rural Link"]
        self.assertEqual(rural.bandwidth_bps, 512_000.0)
        self.assertEqual(rural.delay_ms, 150.0)
        self.assertEqual(rural.jitter_ms, 40.0)
        self.assertEqual(rural.loss_percent, 2.0)

        # 6. Satellite: 10 Mbps, 300 ms, 30 ms, 0.5%
        sat = profile_map["Satellite"]
        self.assertEqual(sat.bandwidth_bps, 10_000_000.0)
        self.assertEqual(sat.delay_ms, 300.0)
        self.assertEqual(sat.jitter_ms, 30.0)
        self.assertEqual(sat.loss_percent, 0.5)

    def test_bandwidth_parsing(self):
        self.assertEqual(parse_bandwidth_to_bps("100 Mbps"), 100_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("100mbit"), 100_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("50 Mbps"), 50_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("50mbit"), 50_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("10 Mbps"), 10_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("10mbit"), 10_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("2 Mbps"), 2_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("2mbit"), 2_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("512 Kbps"), 512_000.0)
        self.assertEqual(parse_bandwidth_to_bps("512kbit"), 512_000.0)
        self.assertEqual(parse_bandwidth_to_bps("1gbps"), 1_000_000_000.0)
        self.assertEqual(parse_bandwidth_to_bps("invalid"), 0.0)

    def test_theoretical_transmission_time(self):
        # 100,000 bytes over 100 Mbps (100,000,000 bps)
        # Time = (100,000 * 8) / 100,000,000 = 800,000 / 100,000,000 = 0.008 s
        t_100m = compute_theoretical_transmission_time(100_000, 100_000_000.0)
        self.assertAlmostEqual(t_100m, 0.008, places=4)

        # 100,000 bytes over 512 Kbps (512,000 bps)
        # Time = (100,000 * 8) / 512,000 = 800,000 / 512,000 = 1.5625 s
        t_512k = compute_theoretical_transmission_time(100_000, 512_000.0)
        self.assertAlmostEqual(t_512k, 1.5625, places=4)

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
            for p in PRESET_PROFILES:
                success, msg = controller.apply_profile(p)
                self.assertFalse(success)
                self.assertIn("only supported on Linux", msg)
                self.assertIsNone(controller.current_profile)

    def test_network_controller_reset(self):
        controller = NetworkController()
        success, msg = controller.reset_configuration()
        self.assertIsInstance(success, bool)
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

    def test_record_and_retrieve_experiment_with_network_profile(self):
        record = ExperimentRecord(
            experiment_id="EXP_PROFILE_001",
            timestamp="2026-11-21 12:00:00",
            image_id="benign/benign (1)",
            compression_quality=90,
            original_file_size=200000,
            compressed_file_size=40000,
            compression_ratio=5.0,
            psnr=38.50,
            ssim=0.9612,
            bandwidth_limit="100mbit",
            packet_loss_rate=0.01,
            network_delay_ms=2.0,
            network_jitter_ms=0.5,
            network_profile="Hospital LAN",
            theoretical_transmission_time_sec=0.0032,
            transmission_time_sec=0.0045,
            measured_throughput_kbps=71111.11,
            checksum_verified=True,
            dice_coefficient=0.852,
            iou=0.742,
            notes="Testing Hospital LAN profile logging",
        )

        self.mgr.record_experiment(record)

        # Verify SQLite retrieval
        rows = self.mgr.fetch_all()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["experiment_id"], "EXP_PROFILE_001")
        self.assertEqual(r["image_id"], "benign/benign (1)")
        self.assertEqual(r["compression_quality"], 90)
        self.assertEqual(r["network_profile"], "Hospital LAN")
        self.assertAlmostEqual(r["network_delay_ms"], 2.0, places=2)
        self.assertAlmostEqual(r["network_jitter_ms"], 0.5, places=2)
        self.assertAlmostEqual(r["packet_loss_rate"], 0.01, places=3)
        self.assertAlmostEqual(r["theoretical_transmission_time_sec"], 0.0032, places=4)
        self.assertEqual(r["checksum_verified"], 1)

        # Verify CSV contents
        self.assertTrue(self.csv_path.exists())
        with open(self.csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.assertEqual(len(lines), 2)  # Header + 1 record
        header = lines[0].split(",")
        self.assertIn("network_profile", header)
        self.assertIn("network_jitter_ms", header)
        self.assertIn("Hospital LAN", lines[1])


if __name__ == "__main__":
    unittest.main()
