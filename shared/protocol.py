"""Shared TCP framing protocol and socket communication primitives."""

import hashlib
import json
import socket
import struct
from typing import Optional, Tuple

from shared.constants import (
    ACK_CHECKSUM_MISMATCH,
    ACK_CORRUPT_PAYLOAD,
    ACK_OK,
    ACK_SERVER_ERROR,
    MAGIC_BYTES,
    PROTOCOL_VERSION,
)
from shared.metadata import AcknowledgmentMessage, TransmissionMetadata

# Binary Frame Layout:
# [4 bytes: Magic b"NAC1"]
# [2 bytes: Protocol version (uint16)]
# [4 bytes: Metadata JSON length N (uint32)]
# [N bytes: Metadata UTF-8 JSON]
# [4 bytes: Image binary length M (uint32)]
# [M bytes: Image raw binary payload]

HEADER_STRUCT_FORMAT = "!4sHI"
HEADER_PREFIX_SIZE = struct.calcsize(HEADER_STRUCT_FORMAT)  # 4 + 2 + 4 = 10 bytes


def recv_exact(sock: socket.socket, num_bytes: int, timeout: Optional[float] = None) -> bytes:
    """Read exactly num_bytes from a TCP socket handling partial reads."""
    if timeout is not None:
        sock.settimeout(timeout)

    buffer = bytearray()
    while len(buffer) < num_bytes:
        chunk = sock.recv(num_bytes - len(buffer))
        if not chunk:
            raise ConnectionError(
                f"Socket closed unexpectedly while reading {num_bytes} bytes "
                f"(received {len(buffer)}/{num_bytes} bytes)."
            )
        buffer.extend(chunk)
    return bytes(buffer)


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 digest for a byte sequence."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()

