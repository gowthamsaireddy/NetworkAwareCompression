"""TCP Client for sending compressed images and handling receiver acknowledgments."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import socket
import time
from typing import Callable, List, Optional, Tuple

from common.logging_config import setup_logger
from shared.constants import ACK_OK, DEFAULT_HOST, DEFAULT_PORT
from shared.metadata import AcknowledgmentMessage, TransmissionMetadata
from shared.protocol import (
    IMAGE_HEADER_SIZE,
    compute_sha256,
    recv_ack_frame,
    send_image_frame,
)

logger = setup_logger("TCPTransmissionClient")


@dataclass
class TransmissionReport:
    """Stores measured transmission statistics for a single transfer."""
    image_id: str
    bytes_sent: int
    transmission_time_sec: float
    throughput_kbps: float
    checksum_verified: bool
    ack_status: int
    timestamp: str
    error_message: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Success is strictly conditioned on receiver confirming checksum verification."""
        return self.ack_status == ACK_OK and self.checksum_verified


class TCPTransmissionClient:
    """Handles TCP socket connections, framed transmissions, and verified ACK reception."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 15.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self.transfer_history: List[TransmissionReport] = []

    def connect(self) -> Tuple[bool, str]:
        """Establishes connection to the receiver TCP server."""
        if self.sock is not None:
            return True, f"Already connected to {self.host}:{self.port}"

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            self.sock = sock
            msg = f"Connected to {self.host}:{self.port}"
            logger.info(msg)
            return True, msg
        except Exception as e:
            self.sock = None
            err = f"Connection to {self.host}:{self.port} failed: {e}"
            logger.error(err)
            return False, err

    def disconnect(self) -> None:
        """Closes connection."""
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            logger.info(f"Disconnected from {self.host}:{self.port}")

    def is_connected(self) -> bool:
        return self.sock is not None

    def send_image(
        self,
        metadata: TransmissionMetadata,
        image_bytes: bytes,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TransmissionReport:
        """Transmits image and metadata to receiver and awaits verified acknowledgement.

        Strict Zero Dummy-Data policy:
        - Timing and throughput are purely based on physical clock measurements.
        - Success is never asserted without verified acknowledgment from receiver.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Automatically connect if not already connected
        if not self.is_connected():
            connected, msg = self.connect()
            if not connected:
                report = TransmissionReport(
                    image_id=metadata.image_id,
                    bytes_sent=0,
                    transmission_time_sec=0.0,
                    throughput_kbps=0.0,
                    checksum_verified=False,
                    ack_status=-1,
                    timestamp=now_str,
                    error_message=msg,
                )
                self.transfer_history.append(report)
                return report

        assert self.sock is not None

        try:
            # Measure actual elapsed transmission duration
            t_start = time.perf_counter()

            # Transmit binary frame over TCP
            send_image_frame(
                self.sock,
                metadata,
                image_bytes,
                progress_callback=progress_callback,
                timeout=self.timeout,
            )

            # Await receiver acknowledgment confirming byte count & SHA-256
            ack = recv_ack_frame(self.sock, timeout=self.timeout)
            t_end = time.perf_counter()

            # Physical measurements
            elapsed_sec = max(t_end - t_start, 1e-6)
            total_bytes = IMAGE_HEADER_SIZE + len(metadata.to_json().encode("utf-8")) + len(image_bytes)
            throughput_kbps = (total_bytes * 8.0) / (elapsed_sec * 1000.0)

            report = TransmissionReport(
                image_id=metadata.image_id,
                bytes_sent=total_bytes,
                transmission_time_sec=elapsed_sec,
                throughput_kbps=throughput_kbps,
                checksum_verified=ack.sha256_verified,
                ack_status=ack.status_code,
                timestamp=now_str,
                error_message=ack.error_message,
            )
            self.transfer_history.append(report)

            if report.is_success:
                logger.info(
                    f"Transfer succeeded for {metadata.image_id}: {total_bytes} bytes in {elapsed_sec:.4f}s "
                    f"({throughput_kbps:.2f} Kbps). Checksum verified."
                )
            else:
                logger.warning(
                    f"Transfer rejected or unverified for {metadata.image_id}: ACK code={ack.status_code}, error={ack.error_message}"
                )

            return report

        except (socket.timeout, TimeoutError) as e:
            err = f"Transfer timed out: {e}"
            logger.error(err)
            self.disconnect()
            report = TransmissionReport(
                image_id=metadata.image_id,
                bytes_sent=0,
                transmission_time_sec=0.0,
                throughput_kbps=0.0,
                checksum_verified=False,
                ack_status=-1,
                timestamp=now_str,
                error_message=err,
            )
            self.transfer_history.append(report)
            return report

        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            err = f"Connection broken during transfer: {e}"
            logger.error(err)
            self.disconnect()
            report = TransmissionReport(
                image_id=metadata.image_id,
                bytes_sent=0,
                transmission_time_sec=0.0,
                throughput_kbps=0.0,
                checksum_verified=False,
                ack_status=-1,
                timestamp=now_str,
                error_message=err,
            )
            self.transfer_history.append(report)
            return report

        except Exception as e:
            err = f"Unexpected transmission failure: {e}"
            logger.error(err)
            self.disconnect()
            report = TransmissionReport(
                image_id=metadata.image_id,
                bytes_sent=0,
                transmission_time_sec=0.0,
                throughput_kbps=0.0,
                checksum_verified=False,
                ack_status=-1,
                timestamp=now_str,
                error_message=err,
            )
            self.transfer_history.append(report)
            return report
