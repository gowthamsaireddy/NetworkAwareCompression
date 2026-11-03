"""Shared TCP framing protocol and socket communication primitives.

Binary Frame Formats:
1. Image Frame:
   [4 bytes: MAGIC_BYTES b"NAC1"]
   [2 bytes: Protocol version (uint16)]
   [4 bytes: Metadata JSON byte length N (uint32)]
   [4 bytes: Image payload byte length M (uint32)]
   [N bytes: Metadata UTF-8 JSON]
   [M bytes: Raw image binary payload]

2. Acknowledgment Frame:
   [4 bytes: ACK_MAGIC_BYTES b"NACK"]
   [2 bytes: Protocol version (uint16)]
   [4 bytes: ACK JSON byte length K (uint32)]
   [K bytes: ACK UTF-8 JSON]

Both sender and receiver use this exact implementation to guarantee wire compatibility.
"""

import hashlib
import json
import socket
import struct
import time
from typing import Callable, Optional, Tuple

from shared.constants import (
    ACK_CHECKSUM_MISMATCH,
    ACK_CORRUPT_PAYLOAD,
    ACK_MAGIC_BYTES,
    ACK_OK,
    ACK_SERVER_ERROR,
    MAGIC_BYTES,
    PROTOCOL_VERSION,
)
from shared.metadata import AcknowledgmentMessage, TransmissionMetadata

# Struct definitions:
# Image Header: 4s (magic), H (version), I (meta_len), I (img_len) -> 14 bytes
IMAGE_HEADER_FORMAT = "!4sHII"
IMAGE_HEADER_SIZE = struct.calcsize(IMAGE_HEADER_FORMAT)  # 14 bytes

# ACK Header: 4s (magic), H (version), I (ack_len) -> 10 bytes
ACK_HEADER_FORMAT = "!4sHI"
ACK_HEADER_SIZE = struct.calcsize(ACK_HEADER_FORMAT)  # 10 bytes

CHUNK_SIZE = 65536  # 64 KB chunk size for streaming and progress updates


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 digest for a byte sequence."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def recv_exact(sock: socket.socket, num_bytes: int, timeout: Optional[float] = None) -> bytes:
    """Read exactly num_bytes from a TCP socket, correctly handling partial reads."""
    if timeout is not None:
        sock.settimeout(timeout)

    buffer = bytearray()
    while len(buffer) < num_bytes:
        try:
            chunk = sock.recv(min(CHUNK_SIZE, num_bytes - len(buffer)))
        except (socket.timeout, TimeoutError):
            raise TimeoutError(
                f"Socket timed out while reading {num_bytes} bytes (received {len(buffer)}/{num_bytes} bytes)."
            )
        except (ConnectionResetError, ConnectionAbortedError) as e:
            raise ConnectionError(f"Connection interrupted by peer: {e}")

        if not chunk:
            raise ConnectionError(
                f"Socket closed prematurely by peer (received {len(buffer)}/{num_bytes} bytes)."
            )
        buffer.extend(chunk)
    return bytes(buffer)


def send_image_frame(
    sock: socket.socket,
    metadata: TransmissionMetadata,
    image_bytes: bytes,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    timeout: Optional[float] = None,
) -> None:
    """Serializes and sends an image frame over TCP with chunked progress reporting."""
    if timeout is not None:
        sock.settimeout(timeout)

    # 1. Validation before wire transmission
    if len(image_bytes) != metadata.image_byte_length:
        raise ValueError(
            f"Image payload length mismatch: got {len(image_bytes)} bytes, metadata specified {metadata.image_byte_length}"
        )

    # Ensure checksum in metadata matches actual payload
    actual_sha = compute_sha256(image_bytes)
    if actual_sha.lower() != metadata.sha256_checksum.lower():
        raise ValueError(
            f"Checksum mismatch prior to send: computed {actual_sha}, metadata specified {metadata.sha256_checksum}"
        )

    # 2. Prepare metadata and headers
    meta_bytes = metadata.to_json().encode("utf-8")
    meta_len = len(meta_bytes)
    img_len = len(image_bytes)

    header = struct.pack(
        IMAGE_HEADER_FORMAT,
        MAGIC_BYTES,
        metadata.protocol_version,
        meta_len,
        img_len,
    )

    # 3. Transmit header and metadata
    sock.sendall(header)
    sock.sendall(meta_bytes)

    # 4. Transmit image payload in chunks with progress tracking
    total_bytes = img_len
    bytes_sent = 0

    if progress_callback:
        progress_callback(0, total_bytes)

    while bytes_sent < total_bytes:
        chunk = image_bytes[bytes_sent : bytes_sent + CHUNK_SIZE]
        sock.sendall(chunk)
        bytes_sent += len(chunk)
        if progress_callback:
            progress_callback(bytes_sent, total_bytes)


def recv_image_frame(
    sock: socket.socket,
    timeout: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[TransmissionMetadata, bytes]:
    """Receives and deserializes an image frame from a TCP socket handling partial reads."""
    # 1. Read binary header (14 bytes)
    header_bytes = recv_exact(sock, IMAGE_HEADER_SIZE, timeout=timeout)
    magic, version, meta_len, img_len = struct.unpack(IMAGE_HEADER_FORMAT, header_bytes)

    if magic != MAGIC_BYTES:
        raise ValueError(f"Invalid frame magic: received {magic!r}, expected {MAGIC_BYTES!r}")

    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {version} (expected {PROTOCOL_VERSION})")

    # 2. Read metadata JSON payload
    meta_bytes = recv_exact(sock, meta_len, timeout=timeout)
    try:
        meta_str = meta_bytes.decode("utf-8")
        metadata = TransmissionMetadata.from_json(meta_str)
    except Exception as e:
        raise ValueError(f"Corrupt metadata JSON: {e}")

    # Validate declared metadata lengths against header
    if metadata.image_byte_length != img_len:
        raise ValueError(
            f"Metadata length inconsistency: header img_len={img_len}, metadata.image_byte_length={metadata.image_byte_length}"
        )

    # 3. Read image binary payload (chunked with progress reporting)
    buffer = bytearray()
    if progress_callback:
        progress_callback(0, img_len)

    while len(buffer) < img_len:
        needed = min(CHUNK_SIZE, img_len - len(buffer))
        chunk = recv_exact(sock, needed, timeout=timeout)
        buffer.extend(chunk)
        if progress_callback:
            progress_callback(len(buffer), img_len)

    image_bytes = bytes(buffer)
    return metadata, image_bytes


def send_ack_frame(
    sock: socket.socket,
    ack: AcknowledgmentMessage,
    timeout: Optional[float] = None,
) -> None:
    """Sends an acknowledgment frame over TCP."""
    if timeout is not None:
        sock.settimeout(timeout)

    ack_bytes = ack.to_json().encode("utf-8")
    ack_len = len(ack_bytes)

    header = struct.pack(ACK_HEADER_FORMAT, ACK_MAGIC_BYTES, PROTOCOL_VERSION, ack_len)
    sock.sendall(header + ack_bytes)


def recv_ack_frame(
    sock: socket.socket,
    timeout: Optional[float] = None,
) -> AcknowledgmentMessage:
    """Receives an acknowledgment frame from TCP."""
    header_bytes = recv_exact(sock, ACK_HEADER_SIZE, timeout=timeout)
    magic, version, ack_len = struct.unpack(ACK_HEADER_FORMAT, header_bytes)

    if magic != ACK_MAGIC_BYTES:
        raise ValueError(f"Invalid ACK magic: received {magic!r}, expected {ACK_MAGIC_BYTES!r}")

    ack_bytes = recv_exact(sock, ack_len, timeout=timeout)
    try:
        ack_str = ack_bytes.decode("utf-8")
        return AcknowledgmentMessage.from_json(ack_str)
    except Exception as e:
        raise ValueError(f"Corrupt ACK message: {e}")
