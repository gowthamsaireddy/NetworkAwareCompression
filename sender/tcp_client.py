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
    wire_time_sec: float = 0.0
    round_trip_time_sec: float = 0.0
    connection_time_ms: float = 0.0
    effective_bandwidth_kbps: float = 0.0
    decoding_time_ms: Optional[float] = None
    verification_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    dice: Optional[float] = None
    iou: Optional[float] = None
    received_bytes: Optional[int] = None

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
        self.last_connection_time_ms: float = 0.0

    def connect(self) -> Tuple[bool, str]:
        """Establishes connection to the receiver TCP server."""
        if self.sock is not None:
            return True, f"Already connected to {self.host}:{self.port}"

        try:
            t_conn_start = time.perf_counter()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            self.last_connection_time_ms = (time.perf_counter() - t_conn_start) * 1000.0
            self.sock = sock
            msg = f"Connected to {self.host}:{self.port} in {self.last_connection_time_ms:.2f}ms"
            logger.info(msg)
            return True, msg
        except Exception as e:
            try:
                sock.close()
            except Exception:
                pass
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

    def set_endpoint(self, host: str, port: int) -> None:
        """Updates target endpoint. Disconnects existing socket if endpoint changed."""
        if self.host != host or self.port != port:
            self.disconnect()
            self.host = host
            self.port = port
            logger.info(f"Updated receiver endpoint to {self.host}:{self.port}")

    def check_receiver_availability(self, timeout: float = 2.0) -> Tuple[bool, float, str]:
        """Probes receiver TCP port and returns (is_available, rtt_ms, message)."""
        t_start = time.perf_counter()
        probe_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe_sock.settimeout(timeout)
        try:
            probe_sock.connect((self.host, self.port))
            rtt_ms = (time.perf_counter() - t_start) * 1000.0
            probe_sock.close()
            return True, rtt_ms, f"Receiver active at {self.host}:{self.port} (RTT: {rtt_ms:.1f}ms)"
        except Exception as e:
            try:
                probe_sock.close()
            except Exception:
                pass
            return False, 0.0, f"Cannot reach receiver at {self.host}:{self.port}: {e}"

    def send_image(
        self,
        metadata: TransmissionMetadata,
        image_bytes: bytes,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> TransmissionReport:
        """Transmits image and metadata to receiver and awaits verified acknowledgement.

        Includes controlled retry mechanism for lossy or fluctuating networks.
        Strict Zero Dummy-Data policy:
        - Timing and throughput are purely based on physical clock measurements.
        - Success is never asserted without verified acknowledgment from receiver.
        """
        last_report: Optional[TransmissionReport] = None
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for attempt in range(1, max_retries + 1):
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Automatically connect if not already connected
            if not self.is_connected():
                connected, msg = self.connect()
                if not connected:
                    last_report = TransmissionReport(
                        image_id=metadata.image_id,
                        bytes_sent=0,
                        transmission_time_sec=0.0,
                        throughput_kbps=0.0,
                        checksum_verified=False,
                        ack_status=-1,
                        timestamp=now_str,
                        error_message=f"Attempt {attempt}/{max_retries} connect failed: {msg}",
                    )
                    if attempt < max_retries:
                        time.sleep(retry_delay * (1.5 ** (attempt - 1)))
                        continue
                    self.transfer_history.append(last_report)
                    return last_report

            assert self.sock is not None

            try:
                total_bytes = IMAGE_HEADER_SIZE + len(metadata.to_json().encode("utf-8")) + len(image_bytes)

                # 1. Pure wire transmission measurement
                t_wire_start = time.perf_counter()
                send_image_frame(
                    self.sock,
                    metadata,
                    image_bytes,
                    progress_callback=progress_callback,
                    timeout=self.timeout,
                )
                t_wire_end = time.perf_counter()
                wire_time_sec = max(t_wire_end - t_wire_start, 1e-6)
                wire_throughput_kbps = (total_bytes * 8.0) / (wire_time_sec * 1000.0)

                # 2. Await receiver acknowledgment confirming byte count, SHA-256 & receiver metrics
                ack = recv_ack_frame(self.sock, timeout=self.timeout)
                t_ack_end = time.perf_counter()
                rtt_sec = max(t_ack_end - t_wire_start, 1e-6)
                effective_bw_kbps = (total_bytes * 8.0) / (rtt_sec * 1000.0)

                report = TransmissionReport(
                    image_id=metadata.image_id,
                    bytes_sent=total_bytes,
                    transmission_time_sec=wire_time_sec,
                    throughput_kbps=wire_throughput_kbps,
                    checksum_verified=ack.sha256_verified,
                    ack_status=ack.status_code,
                    timestamp=now_str,
                    error_message=ack.error_message,
                    wire_time_sec=wire_time_sec,
                    round_trip_time_sec=rtt_sec,
                    connection_time_ms=self.last_connection_time_ms,
                    effective_bandwidth_kbps=effective_bw_kbps,
                    decoding_time_ms=ack.decoding_time_ms,
                    verification_time_ms=ack.verification_time_ms,
                    inference_time_ms=ack.inference_time_ms,
                    dice=ack.dice,
                    iou=ack.iou,
                    received_bytes=ack.received_bytes,
                )

                if report.is_success:
                    logger.info(
                        f"Transfer succeeded for {metadata.image_id}: {total_bytes} bytes in {wire_time_sec:.4f}s "
                        f"(Wire: {wire_throughput_kbps:.2f} Kbps, RTT: {rtt_sec:.4f}s / {effective_bw_kbps:.2f} Kbps). "
                        f"Checksum verified."
                    )
                    self.transfer_history.append(report)
                    return report
                else:
                    logger.warning(
                        f"Transfer rejected on attempt {attempt}/{max_retries} for {metadata.image_id}: ACK={ack.status_code}, err={ack.error_message}"
                    )
                    last_report = report
                    # If checksum mismatch or corrupt payload, retry
                    self.disconnect()

            except (socket.timeout, TimeoutError) as e:
                err = f"Transfer timed out (attempt {attempt}/{max_retries}): {e}"
                logger.error(err)
                self.disconnect()
                last_report = TransmissionReport(
                    image_id=metadata.image_id,
                    bytes_sent=0,
                    transmission_time_sec=0.0,
                    throughput_kbps=0.0,
                    checksum_verified=False,
                    ack_status=-1,
                    timestamp=now_str,
                    error_message=err,
                )

            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                err = f"Connection broken during transfer (attempt {attempt}/{max_retries}): {e}"
                logger.error(err)
                self.disconnect()
                last_report = TransmissionReport(
                    image_id=metadata.image_id,
                    bytes_sent=0,
                    transmission_time_sec=0.0,
                    throughput_kbps=0.0,
                    checksum_verified=False,
                    ack_status=-1,
                    timestamp=now_str,
                    error_message=err,
                )

            except Exception as e:
                err = f"Unexpected transmission failure (attempt {attempt}/{max_retries}): {e}"
                logger.error(err)
                self.disconnect()
                last_report = TransmissionReport(
                    image_id=metadata.image_id,
                    bytes_sent=0,
                    transmission_time_sec=0.0,
                    throughput_kbps=0.0,
                    checksum_verified=False,
                    ack_status=-1,
                    timestamp=now_str,
                    error_message=err,
                )

            if attempt < max_retries:
                time.sleep(retry_delay * (1.5 ** (attempt - 1)))

        assert last_report is not None
        self.transfer_history.append(last_report)
        return last_report
