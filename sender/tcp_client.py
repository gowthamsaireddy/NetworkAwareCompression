"""TCP Client for sending compressed images to the Receiver."""

from dataclasses import dataclass
from pathlib import Path
import socket
import time
from typing import Callable, Optional, Tuple

from shared.constants import ACK_OK, DEFAULT_HOST, DEFAULT_PORT
from shared.metadata import AcknowledgmentMessage, TransmissionMetadata
from shared.protocol import compute_sha256


@dataclass
class TransmissionReport:
    """Stores measured transmission statistics for a single transfer."""
    image_id: str
    bytes_sent: int
    transmission_time_sec: float
    throughput_kbps: float
    checksum_verified: bool
    ack_status: int
    error_message: Optional[str] = None


class TCPTransmissionClient:
    """Handles TCP socket connections and transmission to the receiver node."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def connect(self) -> Tuple[bool, str]:
        """Establishes connection to the receiver TCP server."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            return True, f"Connected to {self.host}:{self.port}"
        except Exception as e:
            self.sock = None
            return False, f"Connection failed: {e}"

    def disconnect(self) -> None:
        """Closes connection."""
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def is_connected(self) -> bool:
        return self.sock is not None

    def send_image(
        self,
        metadata: TransmissionMetadata,
        image_bytes: bytes,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TransmissionReport:
        """Transmits image and metadata to receiver and awaits acknowledgement.

        Full socket framing exchange will be finalized in Phase 2.
        """
        raise NotImplementedError("TCP transmission logic scheduled for Phase 2.")

