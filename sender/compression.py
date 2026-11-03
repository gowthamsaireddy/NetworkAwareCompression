"""JPEG Compression module for sender node.

Implements variable-quality JPEG compression (Q30, Q50, Q70, Q90) while strictly
preserving original dimensions and calculating real PSNR and SSIM distortion metrics.
Ground-truth masks are strictly prohibited from lossy compression.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from common.logging_config import setup_logger
from sender.image_metrics import calculate_psnr, calculate_ssim
from shared.constants import COMPRESSION_QUALITIES

logger = setup_logger("ImageCompressor")


@dataclass
class CompressionResult:
    """Stores the outcome of a compression operation."""
    quality: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    size_reduction_pct: float
    compressed_bytes: bytes
    decoded_array: np.ndarray  # Decoded uint8 RGB array for display/analysis
    width: int
    height: int
    psnr: float
    ssim: float


class ImageCompressor:
    """Performs in-memory JPEG compression and evaluates PSNR / SSIM fidelity."""

    @staticmethod
    def compress_image(
        image_source: Union[Path, np.ndarray, Image.Image],
        quality: int,
    ) -> CompressionResult:
        """Loads an image, applies JPEG compression in memory, and computes distortion metrics.

        Args:
            image_source: File Path, numpy pixel array, or PIL Image.
            quality: JPEG quality factor (1-100, standard research levels: 30, 50, 70, 90).

        Returns:
            CompressionResult with genuine size metrics, decoded array, PSNR (dB), and SSIM.
        """
        if not (1 <= quality <= 100):
            raise ValueError(f"Quality must be between 1 and 100, got {quality}")

        # Safety check: Guard against mask compression
        if isinstance(image_source, Path):
            if "_mask" in image_source.stem.lower():
                raise ValueError(
                    f"Protocol violation: Attempted to JPEG-compress ground-truth mask '{image_source.name}'. "
                    f"Ground-truth masks must NEVER be lossy compressed."
                )
            if not image_source.exists():
                raise FileNotFoundError(f"Image not found at {image_source}")
            original_size = image_source.stat().st_size
            pil_img = Image.open(image_source)
        elif isinstance(image_source, np.ndarray):
            pil_img = Image.fromarray(image_source)
            original_size = image_source.nbytes
        elif isinstance(image_source, Image.Image):
            pil_img = image_source
            original_size = len(pil_img.tobytes())
        else:
            raise TypeError(f"Unsupported image source type: {type(image_source)}")

        # Ensure RGB mode and preserve original dimensions
        if pil_img.mode != "RGB":
            rgb_orig = pil_img.convert("RGB")
        else:
            rgb_orig = pil_img.copy()

        width, height = rgb_orig.size
        orig_array = np.array(rgb_orig)

        # In-memory JPEG compression
        out_stream = BytesIO()
        rgb_orig.save(out_stream, format="JPEG", quality=quality, optimize=True)
        compressed_bytes = out_stream.getvalue()
        compressed_size = len(compressed_bytes)

        # Decode compressed bytes in-memory to evaluate reconstruction distortion
        with Image.open(BytesIO(compressed_bytes)) as comp_pil:
            decoded_array = np.array(comp_pil.convert("RGB"))
            comp_width, comp_height = comp_pil.size

        # Verify dimension preservation
        if (comp_width, comp_height) != (width, height):
            raise RuntimeError(
                f"Dimension distortion error: original {(width, height)} vs compressed {(comp_width, comp_height)}"
            )

        # Physical calculations
        ratio = (original_size / compressed_size) if compressed_size > 0 else 0.0
        reduction = (
            ((original_size - compressed_size) / original_size * 100.0)
            if original_size > 0
            else 0.0
        )

        # Calculate genuine PSNR and SSIM
        psnr_val = calculate_psnr(orig_array, decoded_array)
        ssim_val = calculate_ssim(orig_array, decoded_array)

        return CompressionResult(
            quality=quality,
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=ratio,
            size_reduction_pct=reduction,
            compressed_bytes=compressed_bytes,
            decoded_array=decoded_array,
            width=width,
            height=height,
            psnr=psnr_val,
            ssim=ssim_val,
        )

    @classmethod
    def compress_all_qualities(
        cls,
        image_source: Union[Path, np.ndarray, Image.Image],
    ) -> Dict[int, CompressionResult]:
        """Compresses the image across all four standard research levels: Q30, Q50, Q70, Q90."""
        results: Dict[int, CompressionResult] = {}
        for q in COMPRESSION_QUALITIES:
            results[q] = cls.compress_image(image_source, quality=q)
        return results
