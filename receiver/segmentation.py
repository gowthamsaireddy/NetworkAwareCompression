"""Segmentation inference pipeline for received breast ultrasound images.

Transforms received images (decode -> grayscale -> 256x256 -> normalize -> frozen inference -> binary mask -> overlay).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


@dataclass
class SegmentationResult:
    """Holds the outputs of a segmentation inference pass."""
    image_id: str
    original_shape: Tuple[int, int]
    processed_image: np.ndarray  # 256x256 grayscale or RGB
    predicted_mask: np.ndarray   # 256x256 binary mask (0 or 255)
    overlay_image: np.ndarray    # 256x256 RGB overlay with highlighted lesion
    inference_time_ms: float


class SegmentationEngine:
    """Executes frozen segmentation inference pipeline."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def run_inference(self, image_data: bytes, image_id: str) -> Tuple[bool, Optional[SegmentationResult], str]:
        """Runs segmentation on raw received image bytes.

        Scheduled for full Phase 5 execution once model architecture is connected.
        """
        if not self.model_manager.is_loaded:
            return False, None, "Cannot run inference: No frozen model checkpoint loaded."

        return False, None, "Inference pipeline scheduled for Phase 5 integration."

