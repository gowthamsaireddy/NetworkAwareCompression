"""ML-Based Region-Adaptive Compression Engine.

Preserves the diagnostic lesion region and a configurable surrounding margin at high fidelity,
while aggressively compressing non-critical background tissue.
Applies Gaussian alpha-feathering along the dilation perimeter to prevent high-frequency
step discontinuities that could trigger false positive edge artifacts during U-Net segmentation.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
import torch

from common.logging_config import setup_logger
from models.roi_detector import LightweightROIDetector

logger = setup_logger("RegionAdaptiveCompressor")


@dataclass
class RegionAdaptiveResult:
    """Encapsulates the output of a region-adaptive compression operation."""

    compressed_bytes: bytes
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    size_reduction_pct: float
    psnr: float
    ssim: float
    roi_psnr: float
    width: int
    height: int
    margin_px: int
    roi_quality: int
    bg_quality: int
    roi_mask: np.ndarray        # (H, W) uint8 binary mask [0, 255]
    fused_array: np.ndarray     # (H, W, 3) uint8 RGB image array for visualization
    model_version: str
    detection_time_ms: float
    compression_time_ms: float


class RegionAdaptiveCompressor:
    """Executes ML-driven region-adaptive image compression."""

    def __init__(self, model_path: Optional[Path] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or (Path("models") / "roi_detector_best.pth")
        self.model: Optional[LightweightROIDetector] = None
        self.model_version = "roi_detector_v1"
        self._load_model()

    def _load_model(self) -> None:
        """Loads lightweight sender ROI locator if checkpoint exists."""
        if self.model_path.exists():
            try:
                ckpt = torch.load(self.model_path, map_location=self.device, weights_only=True)
                model = LightweightROIDetector()
                model.load_state_dict(ckpt["model_state_dict"])
                model.to(self.device)
                model.eval()
                self.model = model
                self.model_version = ckpt.get("model_version", "roi_detector_v1")
                logger.info(f"Loaded ROI detector model from {self.model_path} ({self.device})")
            except Exception as e:
                logger.warning(f"Could not load ROI detector model from {self.model_path}: {e}")
                self.model = None
        else:
            logger.info(f"ROI detector checkpoint not found at {self.model_path}. Will require GT mask or fallback.")

    def detect_roi(self, gray_img: np.ndarray, threshold: float = 0.35) -> Tuple[np.ndarray, float]:
        """Runs lightweight inference to detect lesion ROI on raw ultrasound frame."""
        h, w = gray_img.shape[:2]
        if self.model is None:
            # Fallback: central region box if no model loaded
            mask = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            mask[cy - h // 6 : cy + h // 6, cx - w // 6 : cx + w // 6] = 255
            return mask, 0.0

        t_start = time.perf_counter()
        img_256 = cv2.resize(gray_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_256.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        pred_256 = (probs >= threshold).astype(np.uint8) * 255
        pred_orig = cv2.resize(pred_256, (w, h), interpolation=cv2.INTER_NEAREST)
        detection_ms = (time.perf_counter() - t_start) * 1000.0
        return pred_orig, detection_ms

    @staticmethod
    def expand_margin(binary_mask: np.ndarray, margin_px: int = 15) -> np.ndarray:
        """Dilates lesion mask by a configurable pixel radius using an elliptical structuring element."""
        if margin_px <= 0 or not np.any(binary_mask > 0):
            return binary_mask.copy()
        ksize = margin_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.dilate(binary_mask, kernel)

    @staticmethod
    def create_alpha_mask(dilated_mask: np.ndarray, margin_px: int = 15) -> np.ndarray:
        """Generates continuous feathered alpha mask [0.0, 1.0] to prevent high-frequency edge artifacts."""
        if not np.any(dilated_mask > 0):
            return np.zeros(dilated_mask.shape, dtype=np.float32)

        sigma = max(margin_px / 3.0, 1.0)
        ksize = (margin_px * 2 + 1) | 1  # Ensure odd kernel size
        blurred = cv2.GaussianBlur(dilated_mask.astype(np.float32), (ksize, ksize), sigma)
        return np.clip(blurred / 255.0, 0.0, 1.0)

    def compress_image(
        self,
        image_source: Union[Path, str, np.ndarray],
        margin_px: int = 15,
        roi_quality: int = 90,
        bg_quality: int = 25,
        roi_mode: str = "ML_MODEL",  # "ML_MODEL", "GT_MASK", or "CENTER_FALLBACK"
        gt_mask: Optional[np.ndarray] = None,
    ) -> RegionAdaptiveResult:
        """Compresses ultrasound frame with adaptive quality based on lesion location and margin expansion."""
        t_comp_start = time.perf_counter()

        # Step 1: Load image into grayscale
        if isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            orig_bytes = p.read_bytes()
            orig_size = len(orig_bytes)
            orig_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_source, np.ndarray):
            if len(image_source.shape) == 3:
                orig_gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = image_source.copy()
            orig_size = int(orig_gray.nbytes)
            orig_bytes = orig_gray.tobytes()
        else:
            raise ValueError(f"Unsupported image source type: {type(image_source)}")

        h, w = orig_gray.shape[:2]

        # Step 2: Determine ROI mask
        detection_ms = 0.0
        if roi_mode == "GT_MASK" and gt_mask is not None:
            if gt_mask.shape[:2] != (h, w):
                base_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                base_mask = gt_mask.copy()
            base_mask = (base_mask > 0).astype(np.uint8) * 255
        else:
            base_mask, detection_ms = self.detect_roi(orig_gray)

        # Step 3: Expand margin
        dilated_mask = self.expand_margin(base_mask, margin_px=margin_px)

        # Step 4: Alpha feathering along border
        alpha = self.create_alpha_mask(dilated_mask, margin_px=margin_px)
        alpha_3d = alpha[:, :, None]

        # Step 5: Dual-rate encoding
        # (a) Heavily compressed background
        _, bg_enc = cv2.imencode(".jpg", orig_gray, [cv2.IMWRITE_JPEG_QUALITY, int(bg_quality)])
        bg_dec = cv2.imdecode(bg_enc, cv2.IMREAD_GRAYSCALE)[:, :, None]

        # (b) Pristine high-quality ROI
        _, roi_enc = cv2.imencode(".jpg", orig_gray, [cv2.IMWRITE_JPEG_QUALITY, int(roi_quality)])
        roi_dec = cv2.imdecode(roi_enc, cv2.IMREAD_GRAYSCALE)[:, :, None]

        # Step 6: Seamless Alpha Blending
        fused = (alpha_3d * roi_dec + (1.0 - alpha_3d) * bg_dec).astype(np.uint8).squeeze()

        # Step 7: Encapsulate into compliant JPEG bitstream at wrapper quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(roi_quality)]
        _, comp_bytes_arr = cv2.imencode(".jpg", fused, encode_params)
        comp_bytes = comp_bytes_arr.tobytes()
        comp_size = len(comp_bytes)

        # Step 8: Metric calculations
        # Decompress transmitted byte stream to verify receiver-observed fidelity
        dec_arr = cv2.imdecode(comp_bytes_arr, cv2.IMREAD_GRAYSCALE)
        psnr_val = float(cv2.PSNR(orig_gray, dec_arr))
        ssim_val = float(ssim_fn(orig_gray, dec_arr, data_range=255))

        # Measure PSNR specifically inside the preserved lesion ROI
        roi_pixels = dilated_mask > 0
        if np.any(roi_pixels):
            mse_roi = float(np.mean((orig_gray[roi_pixels].astype(np.float32) - dec_arr[roi_pixels].astype(np.float32)) ** 2))
            roi_psnr = float(10.0 * np.log10((255.0 ** 2) / max(mse_roi, 1e-10)))
        else:
            roi_psnr = psnr_val

        ratio = (orig_size / comp_size) if comp_size > 0 else 1.0
        reduction_pct = max(0.0, ((orig_size - comp_size) / orig_size) * 100.0) if orig_size > 0 else 0.0
        compression_ms = (time.perf_counter() - t_comp_start) * 1000.0

        fused_rgb = cv2.cvtColor(dec_arr, cv2.COLOR_GRAY2RGB)

        logger.info(
            f"Region-Adaptive compression finished in {compression_ms:.1f}ms: "
            f"{orig_size}B -> {comp_size}B ({ratio:.2f}:1, -{reduction_pct:.1f}%). "
            f"Full PSNR: {psnr_val:.2f}dB, ROI PSNR: {roi_psnr:.2f}dB, SSIM: {ssim_val:.4f}"
        )

        return RegionAdaptiveResult(
            compressed_bytes=comp_bytes,
            original_size_bytes=orig_size,
            compressed_size_bytes=comp_size,
            compression_ratio=ratio,
            size_reduction_pct=reduction_pct,
            psnr=psnr_val,
            ssim=ssim_val,
            roi_psnr=roi_psnr,
            width=w,
            height=h,
            margin_px=margin_px,
            roi_quality=roi_quality,
            bg_quality=bg_quality,
            roi_mask=dilated_mask,
            fused_array=fused_rgb,
            model_version=self.model_version,
            detection_time_ms=detection_ms,
            compression_time_ms=compression_ms,
        )
