"""Unit and Integration Tests for One-Click Automated Experiment Runner and UI.

Verifies:
1. Generation and validity of all 8 supported compression configurations.
2. Loading of exactly 10 held-out test images from data/splits.json.
3. Verification of 480 total automated transmissions (10 images x 8 configs x 6 network profiles).
4. AutomatedExperimentRunner execution, unique output run directory creation, transmitted image saving, CSV logging, and summary generation.
5. Network restoration in finally block even upon stop or error.
6. SenderTransmissionView presence and connection of 'Run All Compression × Network Experiments' button, Stop button, telemetry displays, and progress tracking.
"""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from common.results_manager import ResultsManager
from sender.automated_experiment_runner import (
    AutomatedExperimentRunner,
    CompressionConfig,
    get_supported_compression_configs,
    load_test_dataset_items,
)
from sender.dataset_manager import DatasetItem
from sender.network_control import PRESET_PROFILES, NetworkController, NetworkProfile
from sender.tcp_client import TCPTransmissionClient, TransmissionReport
from shared.metadata import TransmissionMetadata


class TestAutomatedExperimentMatrix(unittest.TestCase):
    """Verifies compression combinations, network profiles, and dataset loading."""

    def test_supported_compression_configs(self):
        configs = get_supported_compression_configs()
        self.assertEqual(len(configs), 8, "Expected exactly 8 supported compression configurations")

        # 4 Uniform JPEG
        norm_cfgs = [c for c in configs if c.method == "NORMAL"]
        self.assertEqual(len(norm_cfgs), 4)
        norm_qualities = [c.quality for c in norm_cfgs]
        self.assertEqual(norm_qualities, [30, 50, 70, 90])

        # 4 Region-Adaptive ROI combinations
        adapt_cfgs = [c for c in configs if c.method == "REGION_ADAPTIVE"]
        self.assertEqual(len(adapt_cfgs), 4)
        for c in adapt_cfgs:
            self.assertEqual(c.quality, 90, "ROI quality must be preserved at Q90")
            self.assertEqual(c.margin_px, 15, "Standard margin is 15px")
        bg_qualities = [c.bg_quality for c in adapt_cfgs]
        self.assertEqual(bg_qualities, [20, 30, 50, 70])

    def test_preset_network_profiles_count(self):
        self.assertEqual(len(PRESET_PROFILES), 6, "Expected exactly 6 predefined network emulation profiles")
        names = [p.name for p in PRESET_PROFILES]
        expected = ["Hospital LAN", "Broadband", "4G LTE", "3G Cellular", "Rural Link", "Satellite"]
        self.assertEqual(names, expected)

    def test_total_planned_transmissions_calculation(self):
        items_count = 10
        configs_count = len(get_supported_compression_configs())  # 8
        profiles_count = len(PRESET_PROFILES)  # 6
        total = items_count * configs_count * profiles_count
        self.assertEqual(total, 480, "Total automated transmissions must be exactly 480")

    def test_load_test_dataset_items(self):
        items = load_test_dataset_items(count=10)
        self.assertEqual(len(items), 10, "Must load exactly 10 test items")
        for it in items:
            self.assertIsInstance(it, DatasetItem)
            self.assertTrue(it.image_path.exists(), f"Image path must exist: {it.image_path}")
            self.assertIn(it.category.lower(), ["benign", "malignant", "normal"])


class TestAutomatedExperimentRunner(unittest.TestCase):
    """Verifies execution, output artifact creation, and error handling of AutomatedExperimentRunner."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="nac_test_auto_")
        self.out_base = Path(self.temp_dir) / "runs"
        self.db_path = Path(self.temp_dir) / "test.db"
        self.csv_path = Path(self.temp_dir) / "test.csv"
        self.results_mgr = ResultsManager(db_path=self.db_path, csv_path=self.csv_path)

        # Create 2 mock test image files
        self.test_items = []
        for i in range(2):
            img_p = Path(self.temp_dir) / f"test_img_{i}.png"
            arr = np.full((128, 128), 70 + i * 20, dtype=np.uint8)
            Image.fromarray(arr).save(img_p)
            self.test_items.append(
                DatasetItem(
                    image_id=f"test/item_{i}",
                    category="benign",
                    image_path=img_p,
                    mask_paths=[],
                    file_size_bytes=img_p.stat().st_size,
                    width=128,
                    height=128,
                )
            )

        # Mock TCP client
        self.mock_tcp = MagicMock(spec=TCPTransmissionClient)
        self.mock_tcp.send_image.return_value = TransmissionReport(
            image_id="test/item_0",
            bytes_sent=1500,
            transmission_time_sec=0.015,
            throughput_kbps=5000.0,
            checksum_verified=True,
            ack_status=0,
            timestamp="2026-11-21 12:00:00",
            wire_time_sec=0.015,
            round_trip_time_sec=0.020,
            connection_time_ms=1.5,
            received_bytes=1524,
        )

        # Mock Network Controller
        self.mock_net = MagicMock(spec=NetworkController)
        self.mock_net.apply_profile.return_value = (True, "Mock applied")
        self.mock_net.reset.return_value = (True, "Mock reset")

        self.runner = AutomatedExperimentRunner(
            tcp_client=self.mock_tcp,
            network_controller=self.mock_net,
            results_manager=self.results_mgr,
            output_base_dir=self.out_base,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_run_subset_creates_run_artifacts(self):
        # Run a small slice: 2 items x 2 configs x 1 profile = 4 trials
        configs = [
            CompressionConfig("JPEG_Q70", "Uniform JPEG Q70", "NORMAL", 70),
            CompressionConfig("ROI_Q90_BG20", "Region-Adaptive", "REGION_ADAPTIVE", 90, bg_quality=20, margin_px=15),
        ]
        profiles = [PRESET_PROFILES[0]]  # Hospital LAN

        progress_calls = []

        def _prog(curr, total, img_id, cfg_name, prof_name, succ, fail, rec):
            progress_calls.append((curr, total, img_id, succ, fail))

        completed, msg, manifest = self.runner.run_all_experiments(
            items=self.test_items,
            compression_configs=configs,
            network_profiles=profiles,
            progress_callback=_prog,
        )

        self.assertTrue(completed)
        self.assertEqual(len(progress_calls), 4)
        self.assertEqual(manifest["successful_transmissions"], 4)
        self.assertEqual(manifest["failed_transmissions"], 0)

        # Check unique run directory created
        run_dir = Path(manifest["output_directory"])
        self.assertTrue(run_dir.exists())
        self.assertTrue((run_dir / "transmitted_images").exists())
        self.assertTrue((run_dir / "experiment_records.csv").exists())
        self.assertTrue((run_dir / "run_summary.json").exists())

        # Check that transmitted compressed images exist on disk
        saved_imgs = list((run_dir / "transmitted_images").glob("*.jpg"))
        self.assertEqual(len(saved_imgs), 4)

        # Check that network reset was called in finally
        self.mock_net.reset.assert_called()

    def test_stop_early_handling(self):
        configs = get_supported_compression_configs()[:2]
        profiles = [PRESET_PROFILES[0]]

        # Stop immediately after first transmission
        call_count = [0]

        def _stop():
            return call_count[0] >= 1

        def _prog(curr, total, img_id, cfg_name, prof_name, succ, fail, rec):
            call_count[0] += 1

        completed, msg, manifest = self.runner.run_all_experiments(
            items=self.test_items,
            compression_configs=configs,
            network_profiles=profiles,
            progress_callback=_prog,
            stop_requested=_stop,
        )

        self.assertFalse(completed)
        self.assertTrue(manifest["is_stopped_early"])
        # Network must be reset even on stop
        self.mock_net.reset.assert_called()


class TestTransmissionViewUI(unittest.TestCase):
    """Verifies that SenderTransmissionView and TestImageSelectionDialog instantiate and expose all controls."""

    @classmethod
    def setUpClass(cls):
        import sys
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_transmission_view_ui_controls(self):
        from ui.sender.transmission_view import SenderTransmissionView

        view = SenderTransmissionView()
        self.assertIsNotNone(view.run_all_experiments_btn)
        self.assertEqual(view.run_all_experiments_btn.text(), "Run All Compression × Network Experiments")
        self.assertTrue(view.run_all_experiments_btn.isEnabled())

        self.assertIsNotNone(view.select_test_images_btn)
        self.assertEqual(view.select_test_images_btn.text(), "Select Test Images...")

        self.assertIsNotNone(view.stop_auto_btn)
        self.assertEqual(view.stop_auto_btn.text(), "Stop Experiments")
        self.assertFalse(view.stop_auto_btn.isEnabled(), "Stop button must be disabled when idle")

        self.assertIsNotNone(view.auto_progress_bar)
        self.assertIsNotNone(view.auto_lbl_current_img)
        self.assertIsNotNone(view.auto_lbl_current_comp)
        self.assertIsNotNone(view.auto_lbl_current_net)
        self.assertIsNotNone(view.auto_lbl_counts)

        # Verify initial items loaded and badge text
        self.assertGreaterEqual(len(view.selected_test_items), 1)
        self.assertIn("Selected Images:", view.lbl_selected_img_count.text())

    def test_image_selection_dialog(self):
        from ui.sender.transmission_view import TestImageSelectionDialog

        items = load_test_dataset_items(count=10)
        dlg = TestImageSelectionDialog(current_selection=items)
        self.assertEqual(len(dlg.selected_items), 10)
        self.assertTrue(dlg.ok_btn.isEnabled())

        # Test clearing selection disables OK button
        dlg._clear_all()
        self.assertEqual(len(dlg.selected_items), 0)
        self.assertFalse(dlg.ok_btn.isEnabled())

        # Test selecting first 10 re-enables OK button
        dlg._select_n(10)
        self.assertEqual(len(dlg.selected_items), 10)
        self.assertTrue(dlg.ok_btn.isEnabled())

        # Test arbitrary selection (e.g. first 25 preset if >= 25 items exist, or all)
        dlg._select_n(25)
        self.assertTrue(len(dlg.selected_items) >= 1)
        self.assertTrue(dlg.ok_btn.isEnabled())

        # Test select all
        dlg._select_n(len(dlg.all_test_items))
        self.assertEqual(len(dlg.selected_items), len(dlg.all_test_items))
        self.assertTrue(dlg.ok_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
