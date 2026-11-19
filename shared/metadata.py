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
    compression_mode: str = "NORMAL"  # "NORMAL" or "REGION_ADAPTIVE"
    region_adaptive_margin: int = 0  # Dilation margin in pixels (0 for normal)
    compression_model_version: Optional[str] = None
    image_category: Optional[str] = None  # benign, malignant, normal
    client_send_timestamp: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransmissionMetadata":
        return cls(
            protocol_version=int(data.get("protocol_version", 1)),
            experiment_id=str(data.get("experiment_id", "")),
            image_id=str(data.get("image_id", "")),
            compression_quality=int(data.get("compression_quality", 70)),
            image_width=int(data.get("image_width", 256)),
            image_height=int(data.get("image_height", 256)),
            image_format=str(data.get("image_format", "JPEG")),
            original_file_size=int(data.get("original_file_size", 0)),
            compressed_file_size=int(data.get("compressed_file_size", 0)),
            image_byte_length=int(data.get("image_byte_length", 0)),
            sha256_checksum=str(data.get("sha256_checksum", "")),
            compression_mode=str(data.get("compression_mode", "NORMAL")),
            region_adaptive_margin=int(data.get("region_adaptive_margin", 0)),
            compression_model_version=data.get("compression_model_version"),
            image_category=data.get("image_category"),
            client_send_timestamp=data.get("client_send_timestamp"),
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
    decoding_time_ms: Optional[float] = None
    verification_time_ms: Optional[float] = None
    inference_time_ms: Optional[float] = None
    dice: Optional[float] = None
    iou: Optional[float] = None
    received_bytes: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcknowledgmentMessage":
        return cls(
            status_code=int(data.get("status_code", 0)),
            image_id=str(data.get("image_id", "")),
            sha256_verified=bool(data.get("sha256_verified", False)),
            error_message=data.get("error_message"),
            decoding_time_ms=data.get("decoding_time_ms"),
            verification_time_ms=data.get("verification_time_ms"),
            inference_time_ms=data.get("inference_time_ms"),
            dice=data.get("dice"),
            iou=data.get("iou"),
            received_bytes=data.get("received_bytes"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AcknowledgmentMessage":
        return cls.from_dict(json.loads(json_str))

