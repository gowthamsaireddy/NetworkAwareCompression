"""Segmentation inference pipeline for received breast ultrasound images.

Executes standardized sequence:
1. Decode image bytes.
2. Apply correct grayscale conversion.
3. Resize to target 256 x 256 resolution.
4. Apply exact normalization (0..1 range float32).
5. Run frozen U-Net inference with torch.no_grad().
6. Generate binary lesion mask thresholded at 0.5.
7. Generate lesion contour overlay on base ultrasound image.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch

from common.logging_config import setup_logger
from shared.constants import TARGET_IMAGE_SIZE

logger = setup_logger("SegmentationEngine")


@dataclass
class SegmentationResult:
    """Holds the outputs of a segmentation inference pass."""
    image_id: str
    original_shape: Tuple[int, int]  # (Height, Width)
    processed_image: np.ndarray       # (256, 256) uint8 grayscale
    probability_map: np.ndarray       # (256, 256) float32 [0.0..1.0]
    predicted_mask: np.ndarray        # (256, 256) uint8 binary (0 or 255)
    overlay_image: np.ndarray         # (256, 256, 3) uint8 RGB
    inference_time_ms: float          # Measured inference duration in ms
    lesion_area_pixels: int           # Pixel count of segmented lesion


def create_lesion_overlay(
    base_gray_256: np.ndarray,
    binary_mask_256: np.ndarray,
    color: Tuple[int, int, int] = (34, 197, 94),  # Emerald green (R, G, B)
    alpha: float = 0.4,
) -> np.ndarray:
    """Creates a blended RGB overlay of the predicted binary lesion mask on the ultrasound frame."""
    if len(base_gray_256.shape) == 2:
        base_rgb = cv2.cvtColor(base_gray_256, cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_gray_256.copy()

    overlay = base_rgb.copy()
    mask_indices = binary_mask_256 > 0

    if np.any(mask_indices):
        # Semi-transparent lesion tint
        tint = np.array(color, dtype=np.float32)
        blended = (1.0 - alpha) * base_rgb[mask_indices] + alpha * tint
        overlay[mask_indices] = np.clip(blended, 0, 255).astype(np.uint8)

        # Draw contour boundary for crisp boundary demarcation
        contours, _ = cv2.findContours(binary_mask_256, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 1, lineType=cv2.LINE_AA)

    return overlay


class SegmentationEngine:
    """Executes frozen segmentation inference pipeline on received image streams."""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.target_size = TARGET_IMAGE_SIZE  # (256, 256)

    def run_inference(
        self,
        image_source: Union[bytes, Path, np.ndarray],
        image_id: str,
        threshold: float = 0.5,
    ) -> Tuple[bool, Optional[SegmentationResult], str]:
        """Runs segmentation on raw received image bytes, path, or array."""
        if not self.model_manager.is_loaded or self.model_manager.model is None:
            return False, None, "Inference halted: No frozen deep learning model loaded."

        # Step 1: Decode image
        try:
            if isinstance(image_source, bytes):
                nparr = np.frombuffer(image_source, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    # Fallback to Pillow if OpenCV imdecode fails
                    pil_img = Image.open(BytesIO(image_source))
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            elif isinstance(image_source, Path):
                if not image_source.exists():
                    return False, None, f"Image file not found: {image_source}"
                img = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
            elif isinstance(image_source, np.ndarray):
                img = image_source
            else:
                return False, None, f"Unsupported image source type: {type(image_source)}"

            orig_h, orig_w = img.shape[:2]
        except Exception as e:
            return False, None, f"Failed to decode image: {e}"

        # Step 2: Apply correct grayscale conversion
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Step 3: Resize to 256 x 256
        gray_256 = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Step 4: Apply normalization (0..1 range float32)
        norm_img = gray_256.astype(np.float32) / 255.0

        # Step 5: Prepare tensor & execute frozen U-Net inference
        device = self.model_manager.device
        tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 256, 256)

        # Adapt channel dimension if model expects 3-channel input
        model = self.model_manager.model
        try:
            first_layer = next(model.parameters())
            # Check model input conv channel expectations if discernible
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    if module.in_channels == 3:
                        tensor = tensor.repeat(1, 3, 1, 1)
                    break
        except Exception:
            pass

        t_start = time.perf_counter()
        with torch.no_grad():
            output = model(tensor)
        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000.0

        # Step 6: Generate probability map & binary lesion mask
        if isinstance(output, (tuple, list)):
            output = output[0]

        if output.shape[1] > 1:
            # Multi-channel output (e.g. [Background, Lesion])
            probs = torch.softmax(output, dim=1)[:, 1, :, :]
        else:
            # Single-channel logits
            probs = torch.sigmoid(output).squeeze(1)

        prob_map = probs.squeeze().cpu().numpy()
        binary_mask = (prob_map >= threshold).astype(np.uint8) * 255
        lesion_pixels = int(np.sum(binary_mask > 0))

        # Step 7 & 8: Generate RGB lesion overlay
        overlay = create_lesion_overlay(gray_256, binary_mask)

        result = SegmentationResult(
            image_id=image_id,
            original_shape=(orig_h, orig_w),
            processed_image=gray_256,
            probability_map=prob_map,
            predicted_mask=binary_mask,
            overlay_image=overlay,
            inference_time_ms=inference_ms,
            lesion_area_pixels=lesion_pixels,
        )

        logger.info(
            f"Segmentation completed for {image_id} in {inference_ms:.2f}ms. Lesion area: {lesion_pixels} px."
        )
        return True, result, "Segmentation successful"
