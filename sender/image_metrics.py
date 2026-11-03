"""Image fidelity metrics (PSNR and SSIM) calculation.

Operates on raw pixel arrays to objectively evaluate compression degradation.
"""

from typing import Tuple
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB between two images."""
    if original.shape != compressed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {compressed.shape}")
    return float(peak_signal_noise_ratio(original, compressed, data_range=255))


def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM) between two images."""
    if original.shape != compressed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {compressed.shape}")

    # Determine channel axis if multichannel
    if len(original.shape) == 3:
        return float(
            structural_similarity(
                original,
                compressed,
                data_range=255,
                channel_axis=-1,
            )
        )
    return float(
        structural_similarity(
            original,
            compressed,
            data_range=255,
        )
    )

