"""JPEG Compression module for sender node.

Implements variable-quality JPEG compression (Q30, Q50, Q70, Q90) while preserving
original dimensions. Masks are never compressed.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


@dataclass
class CompressionResult:
    """Stores the outcome of a compression operation."""
    quality: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    size_reduction_pct: float
    compressed_bytes: bytes
    width: int
    height: int
    psnr: Optional[float] = None
    ssim: Optional[float] = None


class ImageCompressor:
    """Performs in-memory JPEG compression at configured qualities."""

    @staticmethod
    def compress_image(image_path: Path, quality: int) -> CompressionResult:
        """Loads an image, applies JPEG compression in memory, and returns result.

        Full metric calculations (PSNR/SSIM) will be hooked in Phase 3.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        original_size = image_path.stat().st_size
        with Image.open(image_path) as img:
            # Preserve original dimensions and format
            width, height = img.size
            # Convert RGBA or grayscale to RGB for JPEG compliance
            if img.mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img

            output_io = BytesIO()
            rgb_img.save(output_io, format="JPEG", quality=quality)
            compressed_bytes = output_io.getvalue()

        compressed_size = len(compressed_bytes)
        ratio = (original_size / compressed_size) if compressed_size > 0 else 0.0
        reduction = (
            ((original_size - compressed_size) / original_size * 100.0)
            if original_size > 0
            else 0.0
        )

        return CompressionResult(
            quality=quality,
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=ratio,
            size_reduction_pct=reduction,
            compressed_bytes=compressed_bytes,
            width=width,
            height=height,
        )

