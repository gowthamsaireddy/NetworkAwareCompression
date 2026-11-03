"""Data integrity verification for received payloads."""

import hashlib
from typing import Tuple
from shared.metadata import TransmissionMetadata


class IntegrityChecker:
    """Verifies that received binary image payloads match transmitted metadata."""

    @staticmethod
    def verify_checksum(data: bytes, expected_sha256: str) -> Tuple[bool, str]:
        """Computes SHA-256 digest of received bytes and compares against expected checksum."""
        actual_hash = hashlib.sha256(data).hexdigest()
        is_valid = (actual_hash.lower() == expected_sha256.lower())
        return is_valid, actual_hash

    @staticmethod
    def verify_payload_length(data: bytes, expected_length: int) -> bool:
        """Verifies received byte length against metadata specification."""
        return len(data) == expected_length

    @classmethod
    def verify_all(cls, data: bytes, metadata: TransmissionMetadata) -> Tuple[bool, str]:
        """Comprehensive integrity validation."""
        if not cls.verify_payload_length(data, metadata.image_byte_length):
            return False, f"Length mismatch: got {len(data)}, expected {metadata.image_byte_length}"

        valid, actual_sha = cls.verify_checksum(data, metadata.sha256_checksum)
        if not valid:
            return False, f"Checksum mismatch: got {actual_sha}, expected {metadata.sha256_checksum}"

        return True, "Integrity verified"

