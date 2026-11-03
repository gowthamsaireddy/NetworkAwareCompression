"""Comprehensive tests for shared TCP protocol, client, server, and integrity verification."""

import io
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from receiver.image_receiver import ImageReceiverManager
from receiver.integrity import IntegrityChecker
from receiver.tcp_server import TCPReceiverServer
from sender.tcp_client import TCPTransmissionClient
from shared.constants import (
    ACK_CHECKSUM_MISMATCH,
    ACK_MAGIC_BYTES,
    ACK_OK,
    MAGIC_BYTES,
    PROTOCOL_VERSION,
)
from shared.metadata import AcknowledgmentMessage, TransmissionMetadata
from shared.protocol import (
    compute_sha256,
    recv_ack_frame,
    recv_exact,
    recv_image_frame,
    send_ack_frame,
    send_image_frame,
)


class TestProtocolFraming(unittest.TestCase):
    """Tests binary framing, chunking, and partial-read handling."""

    def test_compute_sha256(self):
        sample_bytes = b"Hello Ultrasound Image Payload"
        digest = compute_sha256(sample_bytes)
        self.assertEqual(len(digest), 64)
        # Verify determinism
        self.assertEqual(digest, compute_sha256(sample_bytes))

    def test_recv_exact_with_fragmented_stream(self):
        """Simulate extreme socket fragmentation (1 byte at a time) to prove partial-read robustness."""
        raw_data = b"NETWORK_AWARE_ADAPTIVE_COMPRESSION_TEST_DATA_STREAM"

        # Create a socket pair
        server_sock, client_sock = socket.socketpair()

        def _slow_feeder():
            for byte in raw_data:
                client_sock.sendall(bytes([byte]))
                time.sleep(0.001)
            client_sock.close()

        t = threading.Thread(target=_slow_feeder, daemon=True)
        t.start()

        # Read full buffer in one recv_exact call
        received = recv_exact(server_sock, len(raw_data), timeout=2.0)
        server_sock.close()
        t.join()

        self.assertEqual(received, raw_data)

    def test_image_frame_roundtrip(self):
        """Test serializing and deserializing an image frame over socketpair."""
        server_sock, client_sock = socket.socketpair()

        test_payload = b"\xff\xd8\xff\xe0" + b"\x00" * 4096 + b"\xff\xd9"  # Synthetic dummy JPEG bytes
        test_sha = compute_sha256(test_payload)
        metadata = TransmissionMetadata(
            protocol_version=PROTOCOL_VERSION,
            experiment_id="EXP_TEST_001",
            image_id="benign/test_1",
            compression_quality=70,
            image_width=256,
            image_height=256,
            image_format="JPEG",
            original_file_size=len(test_payload) * 2,
            compressed_file_size=len(test_payload),
            image_byte_length=len(test_payload),
            sha256_checksum=test_sha,
        )

        def _send():
            send_image_frame(client_sock, metadata, test_payload)
            client_sock.close()

        t = threading.Thread(target=_send, daemon=True)
        t.start()

        rec_meta, rec_bytes = recv_image_frame(server_sock, timeout=2.0)
        server_sock.close()
        t.join()

        self.assertEqual(rec_meta.image_id, "benign/test_1")
        self.assertEqual(rec_meta.sha256_checksum, test_sha)
        self.assertEqual(rec_bytes, test_payload)

    def test_ack_frame_roundtrip(self):
        """Test serializing and deserializing ACK frame."""
        server_sock, client_sock = socket.socketpair()

        ack_sent = AcknowledgmentMessage(
            status_code=ACK_OK,
            image_id="malignant/case_42",
            sha256_verified=True,
            error_message=None,
        )

        def _send_ack():
            send_ack_frame(client_sock, ack_sent)
            client_sock.close()

        t = threading.Thread(target=_send_ack, daemon=True)
        t.start()

        ack_rec = recv_ack_frame(server_sock, timeout=2.0)
        server_sock.close()
        t.join()

        self.assertEqual(ack_rec.status_code, ACK_OK)
        self.assertEqual(ack_rec.image_id, "malignant/case_42")
        self.assertTrue(ack_rec.sha256_verified)


class TestRealTCPEthernetCommunication(unittest.TestCase):
    """Tests live TCP communication using TCPReceiverServer and TCPTransmissionClient."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.save_dir = Path(cls.temp_dir.name)

        # Find a free dynamic port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            cls.port = s.getsockname()[1]

        cls.server = TCPReceiverServer(
            host="127.0.0.1",
            port=cls.port,
            save_dir=cls.save_dir,
            client_timeout=5.0,
        )
        started, msg = cls.server.start()
        assert started, f"Could not start server on port {cls.port}: {msg}"
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()
        cls.temp_dir.cleanup()

    def test_successful_transmission_and_ack(self):
        """Verifies end-to-end transmission with checksum verification and receipt."""
        client = TCPTransmissionClient(host="127.0.0.1", port=self.port, timeout=5.0)
        connected, msg = client.connect()
        self.assertTrue(connected, msg)

        # Real test payload
        payload = b"GENUINE_ULTRASOUND_IMAGE_BYTES_Q70" * 256
        sha = compute_sha256(payload)
        metadata = TransmissionMetadata(
            protocol_version=PROTOCOL_VERSION,
            experiment_id="TEST_RUN_101",
            image_id="benign/sample_01",
            compression_quality=70,
            image_width=512,
            image_height=512,
            image_format="JPEG",
            original_file_size=len(payload) * 3,
            compressed_file_size=len(payload),
            image_byte_length=len(payload),
            sha256_checksum=sha,
        )

        progress_records = []

        def _progress(curr, total):
            progress_records.append((curr, total))

        report = client.send_image(metadata, payload, progress_callback=_progress)
        client.disconnect()

        self.assertTrue(report.is_success)
        self.assertTrue(report.checksum_verified)
        self.assertEqual(report.ack_status, ACK_OK)
        self.assertGreater(report.transmission_time_sec, 0.0)
        self.assertGreater(report.throughput_kbps, 0.0)
        self.assertTrue(len(progress_records) > 0)

        # Verify file was written to disk
        expected_saved = self.save_dir / f"benign_sample_01_Q70_TEST_RUN_101.jpg"
        self.assertTrue(expected_saved.exists())
        with open(expected_saved, "rb") as f:
            self.assertEqual(f.read(), payload)

    def test_checksum_mismatch_rejection(self):
        """Sender and receiver must reject corrupt payloads and not report success."""
        valid_payload = b"ORIGINAL_VALID_PAYLOAD" * 100
        valid_sha = compute_sha256(valid_payload)

        # Corrupt one byte in payload after computing checksum
        corrupted_payload = b"X" + valid_payload[1:]

        metadata = TransmissionMetadata(
            protocol_version=PROTOCOL_VERSION,
            experiment_id="TEST_RUN_CORRUPT",
            image_id="malignant/corrupted_case",
            compression_quality=50,
            image_width=256,
            image_height=256,
            image_format="JPEG",
            original_file_size=len(corrupted_payload),
            compressed_file_size=len(corrupted_payload),
            image_byte_length=len(corrupted_payload),
            sha256_checksum=valid_sha,  # Intentionally mismatched!
        )

        # Note: send_image_frame checks sha256 before wire transmission by default,
        # so let's test if raw frame transmission over socket with bad sha triggers receiver rejection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", self.port))

        try:
            import struct
            meta_bytes = metadata.to_json().encode("utf-8")
            header = struct.pack(
                "!4sHII",
                MAGIC_BYTES,
                metadata.protocol_version,
                len(meta_bytes),
                len(corrupted_payload),
            )
            sock.sendall(header + meta_bytes + corrupted_payload)

            # Receiver should return ACK with ACK_CHECKSUM_MISMATCH and sha256_verified=False
            ack = recv_ack_frame(sock, timeout=5.0)

            self.assertEqual(ack.status_code, ACK_CHECKSUM_MISMATCH)
            self.assertFalse(ack.sha256_verified)
            self.assertIn("Checksum mismatch", str(ack.error_message))
        finally:
            sock.close()


if __name__ == "__main__":
    unittest.main()
