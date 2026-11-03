"""Shared metadata structures for image transmission and experimentation."""

from dataclasses import asdict, dataclass
import json
from typing import Any, Optional


@dataclass
class TransmissionMetadata:
    """Metadata transmitted alongside image binary payloads."""

    protocol_version: int
    experiment_id: str
    image_id: str
    compression_quality: int
    image_width: int
    image_height: int
    image_format: str  # e.g., 'JPEG', 'PNG'
    original_file_size: int  # bytes
    compressed_file_size: int  # bytes
    image_byte_length: int  # length of binary payload
    sha256_checksum: str  # hex digest of binary payload

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransmissionMetadata":
        return cls(
            protocol_version=int(data["protocol_version"]),
            experiment_id=str(data["experiment_id"]),
            image_id=str(data["image_id"]),
            compression_quality=int(data["compression_quality"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            image_format=str(data["image_format"]),
            original_file_size=int(data["original_file_size"]),
            compressed_file_size=int(data["compressed_file_size"]),
            image_byte_length=int(data["image_byte_length"]),
            sha256_checksum=str(data["sha256_checksum"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TransmissionMetadata":
        return cls.from_dict(json.loads(json_str))


@dataclass
class AcknowledgmentMessage:
    """Response returned by Receiver to Sender after checksum verification."""

    status_code: int
    image_id: str
    sha256_verified: bool
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcknowledgmentMessage":
        return cls(
            status_code=int(data["status_code"]),
            image_id=str(data["image_id"]),
            sha256_verified=bool(data["sha256_verified"]),
            error_message=data.get("error_message"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AcknowledgmentMessage":
        return cls.from_dict(json.loads(json_str))

