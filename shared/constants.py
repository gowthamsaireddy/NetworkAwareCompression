"""Shared constants for Network-Aware Compression system."""

# Protocol configuration
PROTOCOL_VERSION: int = 1
DEFAULT_PORT: int = 5000
DEFAULT_HOST: str = "0.0.0.0"

# Header framing: 4-byte magic number, 2-byte version, 4-byte payload length
MAGIC_BYTES: bytes = b"NAC1"  # Network-Aware Compression v1

# Acknowledgement status codes
ACK_OK: int = 0
ACK_CHECKSUM_MISMATCH: int = 1
ACK_CORRUPT_PAYLOAD: int = 2
ACK_SERVER_ERROR: int = 3

# Compression levels
COMPRESSION_QUALITIES: tuple[int, ...] = (30, 50, 70, 90)

# Target image dimensions for segmentation
TARGET_IMAGE_SIZE: tuple[int, int] = (256, 256)

# UI Strings (Zero dummy-data policy compliance)
STATUS_NO_DATA: str = "No data available"
STATUS_NOT_CALCULATED: str = "Not calculated"
STATUS_NOT_IMPLEMENTED: str = "Not implemented"
STATUS_WAITING: str = "Waiting for connection"
STATUS_DISCONNECTED: str = "Disconnected"
STATUS_CONNECTED: str = "Connected"
STATUS_IDLE: str = "Idle"

