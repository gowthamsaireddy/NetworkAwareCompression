"""Comprehensive tests for deployment improvements and security hardening.

Covers:
1. Configuration loading and environment variable overrides.
2. Protocol safety limits (MAX_METADATA_SIZE, MAX_IMAGE_PAYLOAD_SIZE).
3. Path traversal sanitization in ImageReceiverManager.
4. Pre-write disk space safety check in ImageReceiverManager.
5. TCPTransmissionClient reachability probe and retry behavior.
6. Batch experiment orchestration controls and cancellation.
"""

import io
import os
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from common.config import (
    SenderConfig,
    ReceiverConfig,
    load_raw_config,
    get_sender_config,
    get_receiver_config,
)
from receiver.image_receiver import ImageReceiverManager
from sender.experiment_orchestrator import ExperimentOrchestrator
from sender.tcp_client import TCPTransmissionClient
from shared.constants import MAGIC_BYTES, PROTOCOL_VERSION
from shared.metadata import TransmissionMetadata
from shared.protocol import (
    IMAGE_HEADER_FORMAT,
    MAX_IMAGE_PAYLOAD_SIZE,
    MAX_METADATA_SIZE,
    recv_image_frame,
)


class MockSocket:
    """In-memory socket mock for fast, network-isolated protocol validation."""

    def __init__(self, in_bytes: bytes = b""):
        self.in_buf = io.BytesIO(in_bytes)
        self.out_buf = io.BytesIO()

    def recv(self, num_bytes: int) -> bytes:
        return self.in_buf.read(num_bytes)

    def sendall(self, data: bytes) -> None:
        self.out_buf.write(data)

    def settimeout(self, timeout: float) -> None:
        pass

    def close(self) -> None:
        pass


class TestConfigManagement(unittest.TestCase):
    """Tests configuration loading, JSON parsing, and environment variable overrides."""

    def test_default_config_loading(self):
        cfg_s = SenderConfig()
        cfg_r = ReceiverConfig()
        self.assertEqual(cfg_s.receiver_ip, "127.0.0.1")
        self.assertEqual(cfg_s.receiver_port, 5000)
        self.assertEqual(cfg_r.listen_ip, "0.0.0.0")
        self.assertEqual(cfg_r.listen_port, 5000)
        self.assertTrue(cfg_r.auto_process_enabled)

    def test_env_var_overrides(self):
        env_patches = {
            "NAC_RECEIVER_IP": "192.168.1.150",
            "NAC_RECEIVER_PORT": "9999",
            "NAC_LISTEN_PORT": "9999",
            "NAC_AUTO_PROCESS": "0",
            "NAC_LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env_patches):
            sender_cfg = get_sender_config()
            receiver_cfg = get_receiver_config()

            self.assertEqual(sender_cfg.receiver_ip, "192.168.1.150")
            self.assertEqual(sender_cfg.receiver_port, 9999)
            self.assertEqual(receiver_cfg.listen_port, 9999)
            self.assertFalse(receiver_cfg.auto_process_enabled)
            self.assertEqual(sender_cfg.log_level, "DEBUG")


class TestProtocolSecurityLimits(unittest.TestCase):
    """Tests protocol boundary defense against DoS / excessive payload lengths."""

    def test_metadata_payload_size_limit_rejection(self):
        """Simulate a header advertising an oversized metadata length (> 64KB)."""
        oversized_len = MAX_METADATA_SIZE + 1024
        header = struct.pack(IMAGE_HEADER_FORMAT, MAGIC_BYTES, PROTOCOL_VERSION, oversized_len, 100)
        mock_sock = MockSocket(header)

        with self.assertRaises(ValueError) as ctx:
            recv_image_frame(mock_sock, timeout=1.0)
        self.assertIn("safety limit", str(ctx.exception).lower())

    def test_image_payload_size_limit_rejection(self):
        """Simulate a header advertising an oversized image length (> 50MB)."""
        oversized_img_len = MAX_IMAGE_PAYLOAD_SIZE + 1024 * 1024
        header = struct.pack(IMAGE_HEADER_FORMAT, MAGIC_BYTES, PROTOCOL_VERSION, 10, oversized_img_len)
        mock_sock = MockSocket(header)

        with self.assertRaises(ValueError) as ctx:
            recv_image_frame(mock_sock, timeout=1.0)
        self.assertIn("safety limit", str(ctx.exception).lower())


class TestPathTraversalSanitization(unittest.TestCase):
    """Tests path traversal sanitization and disk safety in ImageReceiverManager."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = ImageReceiverManager(storage_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_metadata(self, image_id: str, exp_id: str = "EXP_TEST") -> TransmissionMetadata:
        return TransmissionMetadata(
            protocol_version=1,
            experiment_id=exp_id,
            image_id=image_id,
            compression_quality=70,
            image_width=256,
            image_height=256,
            image_format="JPEG",
            original_file_size=1024,
            compressed_file_size=512,
            image_byte_length=512,
            sha256_checksum="0" * 64,
        )

    def test_save_image_sanitizes_path_traversal(self):
        """Malicious filenames like ../../test.jpg should be safely flattened."""
        dummy_data = b"\xFF\xD8\xFF\xE0" + b"\x00" * 100
        meta = self._create_metadata("../../etc/shadow.jpg")

        saved_path = self.manager.save_image(meta, dummy_data)
        self.assertTrue(saved_path.exists())

        # Verify the saved file is strictly contained inside storage_dir
        storage_path = Path(self.temp_dir.name).resolve()
        resolved_saved = saved_path.resolve()
        self.assertTrue(storage_path in resolved_saved.parents)

        # File name should not contain path traversal characters
        self.assertNotIn("..", saved_path.name)
        self.assertNotIn("/", saved_path.name)
        self.assertNotIn("\\", saved_path.name)

    def test_save_image_backslashes_traversal(self):
        """Windows-style path traversal with backslashes should also be sanitized."""
        dummy_data = b"\xFF\xD8\xFF\xE0" + b"\x00" * 100
        meta = self._create_metadata("..\\..\\windows\\system32\\trojan.jpg")

        saved_path = self.manager.save_image(meta, dummy_data)
        self.assertTrue(saved_path.exists())
        storage_path = Path(self.temp_dir.name).resolve()
        self.assertTrue(storage_path in saved_path.resolve().parents)
        self.assertNotIn("..", saved_path.name)

    def test_insufficient_disk_space_handling(self):
        """ImageReceiverManager raises OSError if free disk space is under safety threshold."""
        dummy_data = b"\xFF\xD8\xFF\xE0" + b"\x00" * 100
        meta = self._create_metadata("benign (1).png")

        # Mock disk_usage to report only 1 MB free (below 50 MB safety margin)
        fake_usage = MagicMock()
        fake_usage.free = 1 * 1024 * 1024  # 1 MB

        with patch("shutil.disk_usage", return_value=fake_usage):
            with self.assertRaises(OSError) as ctx:
                self.manager.save_image(meta, dummy_data)
            self.assertIn("insufficient disk space", str(ctx.exception).lower())


class TestClientAvailabilityAndRetry(unittest.TestCase):
    """Tests reachability probe and transmission retry mechanisms."""

    def test_check_receiver_availability_offline(self):
        """Probing an offline host should return (False, 0.0, msg)."""
        client = TCPTransmissionClient(host="192.0.2.1", port=59999, timeout=0.1)
        with patch.object(client, "connect", return_value=(False, "Connection refused")):
            is_avail, rtt, msg = client.check_receiver_availability(timeout=0.1)
            self.assertFalse(is_avail)

    def test_check_receiver_availability_online(self):
        """Probing an active server should return (True, rtt_ms >= 0, msg)."""
        client = TCPTransmissionClient(host="127.0.0.1", port=5000, timeout=1.0)
        with patch("socket.socket") as mock_sock_cls:
            mock_sock = MagicMock()
            mock_sock_cls.return_value = mock_sock
            is_avail, rtt, msg = client.check_receiver_availability(timeout=1.0)
            self.assertTrue(is_avail)
            self.assertGreaterEqual(rtt, 0.0)
            self.assertIn("active", msg.lower())

    def test_retry_attempts_exhaustion_on_offline_server(self):
        """Attempting to send to an offline server executes retries and fails gracefully."""
        client = TCPTransmissionClient(host="127.0.0.1", port=59998, timeout=0.05)
        meta = TransmissionMetadata(
            protocol_version=1,
            experiment_id="EXP_RETRY_TEST",
            image_id="benign (1).png",
            compression_quality=70,
            image_width=256,
            image_height=256,
            image_format="JPEG",
            original_file_size=100,
            compressed_file_size=100,
            image_byte_length=100,
            sha256_checksum="0" * 64,
        )
        with patch.object(client, "connect", return_value=(False, "Connection refused")):
            report = client.send_image(meta, b"x" * 100, max_retries=2, retry_delay=0.01)
            self.assertFalse(report.is_success)
            self.assertIn("attempt 2", report.error_message.lower())


class TestBatchExperimentOrchestration(unittest.TestCase):
    """Tests batch experiment runner logic and cancellation."""

    def test_batch_runner_cancellation(self):
        """Ensure batch orchestrator respects stop_requested callable promptly."""
        orchestrator = ExperimentOrchestrator(
            tcp_client=MagicMock(),
            network_controller=MagicMock(),
            adaptive_controller=MagicMock(),
            results_manager=MagicMock(),
        )
        fake_items = [MagicMock(), MagicMock()]

        # Stop requested immediately
        completed, skipped, failed, records, summary = orchestrator.run_batch_experiment(
            items=fake_items,
            compression_method="NORMAL",
            stop_requested=lambda: True,
        )

        self.assertEqual(completed, 0)
        self.assertEqual(len(records), 0)
        orchestrator.network_controller.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
