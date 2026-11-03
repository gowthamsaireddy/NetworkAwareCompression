"""Environment, structure, and GUI smoke test suite."""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Use offscreen platform plugin for headless PyQt6 testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"


class TestEnvironmentAndImports(unittest.TestCase):
    """Verifies that all core packages and architecture modules import without error."""

    def test_core_dependencies(self):
        import numpy as np
        import pandas as pd
        import PIL
        import PyQt6
        import skimage
        import torch

        self.assertIsNotNone(np.__version__)
        self.assertIsNotNone(pd.__version__)
        self.assertIsNotNone(PIL.__version__)
        self.assertIsNotNone(PyQt6)
        self.assertIsNotNone(torch.__version__)

    def test_shared_protocol_metadata(self):
        from shared.metadata import TransmissionMetadata, AcknowledgmentMessage

        meta = TransmissionMetadata(
            protocol_version=1,
            experiment_id="EXP_001",
            image_id="benign/benign (1)",
            compression_quality=70,
            image_width=500,
            image_height=500,
            image_format="JPEG",
            original_file_size=120000,
            compressed_file_size=40000,
            image_byte_length=40000,
            sha256_checksum="abc123def456",
        )
        json_str = meta.to_json()
        restored = TransmissionMetadata.from_json(json_str)
        self.assertEqual(restored.image_id, "benign/benign (1)")
        self.assertEqual(restored.compression_quality, 70)

        ack = AcknowledgmentMessage(
            status_code=0,
            image_id="benign/benign (1)",
            sha256_verified=True,
        )
        self.assertTrue(ack.sha256_verified)

    def test_results_manager_init(self):
        from common.results_manager import ResultsManager
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mgr = ResultsManager(db_path=tmp_path / "test.db", csv_path=tmp_path / "test.csv")
            self.assertTrue((tmp_path / "test.db").exists())
            self.assertTrue((tmp_path / "test.csv").exists())

    def test_sender_modules(self):
        from sender.dataset_manager import DatasetManager
        from sender.compression import ImageCompressor
        from sender.network_control import NetworkController
        from sender.tcp_client import TCPTransmissionClient
        from sender.adaptive_controller import AdaptiveController

        dm = DatasetManager()
        nc = NetworkController()
        tc = TCPTransmissionClient()
        ac = AdaptiveController()

        q, _ = ac.determine_quality(15000)
        self.assertEqual(q, 90)

    def test_receiver_modules(self):
        from receiver.tcp_server import TCPReceiverServer
        from receiver.integrity import IntegrityChecker
        from receiver.model_manager import ModelManager
        from receiver.segmentation_metrics import compute_dice_coefficient, compute_iou
        import numpy as np

        srv = TCPReceiverServer()
        mm = ModelManager()
        self.assertEqual(mm.get_status(), "No model loaded")

        # Test metric math on empty vs non-empty
        m1 = np.ones((10, 10), dtype=np.uint8)
        m2 = np.ones((10, 10), dtype=np.uint8)
        self.assertAlmostEqual(compute_dice_coefficient(m1, m2), 1.0, places=4)
        self.assertAlmostEqual(compute_iou(m1, m2), 1.0, places=4)


class TestGUIInitialization(unittest.TestCase):
    """Ensures PyQt6 applications and all subviews initialize cleanly."""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def test_sender_window_init(self):
        from ui.sender.main_window import SenderMainWindow
        window = SenderMainWindow()
        self.assertIsNotNone(window)
        self.assertEqual(window.stack.count(), 7)

    def test_receiver_window_init(self):
        from ui.receiver.main_window import ReceiverMainWindow
        window = ReceiverMainWindow()
        self.assertIsNotNone(window)
        self.assertEqual(window.stack.count(), 7)


if __name__ == "__main__":
    unittest.main()

