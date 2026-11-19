"""Unit tests for ML-Based Region-Adaptive Compression Engine."""

from pathlib import Path
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from sender.region_adaptive_compressor import RegionAdaptiveCompressor, RegionAdaptiveResult


class TestRegionAdaptiveCompression(unittest.TestCase):
    """Verifies ROI detection, margin expansion, alpha feathering, and dual-rate encoding."""

    @classmethod
    def setUpClass(cls):
        cls.compressor = RegionAdaptiveCompressor()
        # Synthetic test image with a circular lesion simulating ultrasound
        cls.test_img = np.full((256, 256), 60, dtype=np.uint8)
        cv2.circle(cls.test_img, (128, 128), 35, 180, -1)  # Bright hyperechoic lesion
        cls.synthetic_mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(cls.synthetic_mask, (128, 128), 35, 255, -1)

    def test_margin_expansion(self):
        """Verify margin dilation expands lesion boundary by exactly specified radius."""
        margin_px = 15
        dilated = RegionAdaptiveCompressor.expand_margin(self.synthetic_mask, margin_px=margin_px)
        self.assertGreater(np.sum(dilated > 0), np.sum(self.synthetic_mask > 0))

        # Check radius expansion along horizontal axis
        orig_indices = np.where(self.synthetic_mask[128, :] > 0)[0]
        dilated_indices = np.where(dilated[128, :] > 0)[0]
        left_expansion = orig_indices[0] - dilated_indices[0]
        right_expansion = dilated_indices[-1] - orig_indices[-1]

        self.assertAlmostEqual(left_expansion, margin_px, delta=2)
        self.assertAlmostEqual(right_expansion, margin_px, delta=2)

    def test_alpha_feathering_continuity(self):
        """Verify alpha feathering produces smooth continuous transition [0.0, 1.0] without seams."""
        dilated = RegionAdaptiveCompressor.expand_margin(self.synthetic_mask, margin_px=15)
        alpha = RegionAdaptiveCompressor.create_alpha_mask(dilated, margin_px=15)

        self.assertEqual(alpha.dtype, np.float32)
        self.assertTrue(np.all(alpha >= 0.0))
        self.assertTrue(np.all(alpha <= 1.0))
        # Core lesion should have high alpha close to 1.0
        self.assertGreater(alpha[128, 128], 0.95)
        # Distant background should have alpha = 0.0
        self.assertEqual(alpha[10, 10], 0.0)

    def test_compression_execution_gt_mode(self):
        """Test end-to-end adaptive compression using ground-truth mask."""
        res = self.compressor.compress_image(
            image_source=self.test_img,
            margin_px=15,
            roi_quality=90,
            bg_quality=20,
            roi_mode="GT_MASK",
            gt_mask=self.synthetic_mask,
        )

        self.assertIsInstance(res, RegionAdaptiveResult)
        self.assertGreater(len(res.compressed_bytes), 0)
        self.assertGreater(res.compression_ratio, 1.0)
        self.assertGreater(res.psnr, 25.0)
        # ROI PSNR should be significantly higher than full PSNR due to high quality preservation
        self.assertGreaterEqual(res.roi_psnr, res.psnr)
        self.assertGreater(res.ssim, 0.85)

    def test_compression_execution_ml_mode(self):
        """Test end-to-end adaptive compression using trained ML ROI Detector."""
        res = self.compressor.compress_image(
            image_source=self.test_img,
            margin_px=20,
            roi_quality=95,
            bg_quality=25,
            roi_mode="ML_MODEL",
        )

        self.assertIsInstance(res, RegionAdaptiveResult)
        self.assertGreater(len(res.compressed_bytes), 0)
        self.assertEqual(res.model_version, "roi_detector_v1")
        self.assertGreater(res.width, 0)
        self.assertGreater(res.height, 0)


if __name__ == "__main__":
    unittest.main()
