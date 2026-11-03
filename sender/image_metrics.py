"""Image fidelity metrics (PSNR and SSIM) calculation.

Operates on raw pixel arrays to objectively evaluate compression degradation.
Strict adherence to zero-dummy policy: all metrics computed directly from pixel data.
"""

from typing import Tuple
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def ensure_comparable_arrays(original: np.ndarray, compressed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensures both arrays have matching uint8 types and channel layouts for metric calculation."""
    if original.shape != compressed.shape:
        raise ValueError(f"Shape mismatch: original shape {original.shape} vs compressed shape {compressed.shape}")

    orig_arr = original.astype(np.uint8) if original.dtype != np.uint8 else original
    comp_arr = compressed.astype(np.uint8) if compressed.dtype != np.uint8 else compressed
    return orig_arr, comp_arr


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB between two images.

    Returns float('inf') if images are bitwise identical.
    """
    orig_arr, comp_arr = ensure_comparable_arrays(original, compressed)
    if np.array_equal(orig_arr, comp_arr):
        return float("inf")
    return float(peak_signal_noise_ratio(orig_arr, comp_arr, data_range=255))


def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM) between two images (range -1.0 to 1.0)."""
    orig_arr, comp_arr = ensure_comparable_arrays(original, compressed)

    if len(orig_arr.shape) == 3 and orig_arr.shape[-1] in (3, 4):
        # Multichannel RGB/RGBA image
        return float(
            structural_similarity(
                orig_arr,
                comp_arr,
                data_range=255,
                channel_axis=-1,
            )
        )

    # 2D Grayscale image
    return float(
        structural_similarity(
            orig_arr,
            comp_arr,
            data_range=255,
        )
    )
