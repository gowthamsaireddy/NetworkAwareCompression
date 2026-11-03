"""Tests for Phase 3: BUSI Dataset Manager, JPEG Compression, and Image Metrics."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from sender.compression import CompressionResult, ImageCompressor
from sender.dataset_manager import DatasetItem, DatasetManager
from sender.image_metrics import calculate_psnr, calculate_ssim


class TestDatasetManager(unittest.TestCase):
    """Tests BUSI dataset scanning, mask pair identification, multi-mask merging, and manifests."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Create realistic synthetic BUSI folder hierarchy
        self.benign_dir = self.root_path / "benign"
        self.malignant_dir = self.root_path / "malignant"
        self.normal_dir = self.root_path / "normal"

        self.benign_dir.mkdir(parents=True)
        self.malignant_dir.mkdir(parents=True)
        self.normal_dir.mkdir(parents=True)

        # Create Benign image with 1 mask
        img_b = Image.new("RGB", (256, 256), color=(100, 100, 100))
        img_b.save(self.benign_dir / "benign (1).png")
        mask_b = Image.new("L", (256, 256), color=0)
        # Add white circle lesion
        for x in range(50, 100):
            for y in range(50, 100):
                mask_b.putpixel((x, y), 255)
        mask_b.save(self.benign_dir / "benign (1)_mask.png")

        # Create Malignant image with 2 separate masks (multi-lesion case)
        img_m = Image.new("RGB", (300, 200), color=(80, 80, 80))
        img_m.save(self.malignant_dir / "malignant (1).png")
        mask_m1 = Image.new("L", (300, 200), color=0)
        for x in range(20, 60):
            for y in range(20, 60):
                mask_m1.putpixel((x, y), 255)
        mask_m1.save(self.malignant_dir / "malignant (1)_mask.png")

        mask_m2 = Image.new("L", (300, 200), color=0)
        for x in range(120, 160):
            for y in range(80, 120):
                mask_m2.putpixel((x, y), 255)
        mask_m2.save(self.malignant_dir / "malignant (1)_mask_1.png")

        # Create Normal image with NO masks
        img_n = Image.new("RGB", (256, 256), color=(50, 50, 50))
        img_n.save(self.normal_dir / "normal (1).png")

        self.dm = DatasetManager(self.root_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scan_dataset_structure(self):
        items = self.dm.scan_dataset()
        self.assertEqual(len(items), 3)

        # Check categories
        categories = {item.category for item in items}
        self.assertEqual(categories, {"benign", "malignant", "normal"})

        # Check benign item
        item_b = self.dm.get_item_by_id("benign/benign (1)")
        self.assertIsNotNone(item_b)
        self.assertEqual(item_b.width, 256)
        self.assertEqual(item_b.height, 256)
        self.assertEqual(len(item_b.mask_paths), 1)

        # Check malignant item with 2 masks
        item_m = self.dm.get_item_by_id("malignant/malignant (1)")
        self.assertIsNotNone(item_m)
        self.assertEqual(item_m.width, 300)
        self.assertEqual(item_m.height, 200)
        self.assertEqual(len(item_m.mask_paths), 2)

        # Check normal item (0 masks)
        item_n = self.dm.get_item_by_id("normal/normal (1)")
        self.assertIsNotNone(item_n)
        self.assertEqual(len(item_n.mask_paths), 0)

    def test_multi_mask_merging(self):
        self.dm.scan_dataset()
        item_m = self.dm.get_item_by_id("malignant/malignant (1)")
        merged = self.dm.load_ground_truth_mask(item_m)
        self.assertIsNotNone(merged)
        self.assertEqual(merged.shape, (200, 300))
        # Both lesions must be present (pixel values 255)
        self.assertEqual(merged[40, 40], 255)
        self.assertEqual(merged[100, 140], 255)
        self.assertEqual(merged[0, 0], 0)

        # Normal case returns None
        item_n = self.dm.get_item_by_id("normal/normal (1)")
        self.assertIsNone(self.dm.load_ground_truth_mask(item_n))

    def test_manifest_export_and_import(self):
        self.dm.scan_dataset()
        manifest_path = self.root_path / "manifest.json"
        self.dm.export_manifest(manifest_path)
        self.assertTrue(manifest_path.exists())

        # Load into fresh DatasetManager instance
        fresh_dm = DatasetManager()
        loaded = fresh_dm.load_manifest(manifest_path)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(fresh_dm.get_item_by_id("benign/benign (1)").width, 256)


class TestJPEGCompression(unittest.TestCase):
    """Tests variable rate JPEG compression, dimension preservation, and mask protection."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img_path = Path(self.temp_dir.name) / "test_us.png"

        # Create realistic synthetic ultrasound-like texture with speckle distribution
        rng = np.random.default_rng(42)
        speckle = rng.integers(30, 220, size=(256, 256), dtype=np.uint8)
        img = Image.fromarray(speckle).convert("RGB")
        img.save(self.img_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compression_qualities(self):
        results = ImageCompressor.compress_all_qualities(self.img_path)
        self.assertEqual(set(results.keys()), {30, 50, 70, 90})

        # Dimensions must be preserved across all levels
        for q, res in results.items():
            self.assertEqual(res.width, 256)
            self.assertEqual(res.height, 256)
            self.assertEqual(res.decoded_array.shape, (256, 256, 3))
            self.assertGreater(res.compression_ratio, 1.0)
            self.assertGreater(res.psnr, 15.0)
            self.assertGreater(res.ssim, 0.40)

        # Quality monotonicity: higher quality yields larger payload and higher PSNR/SSIM
        self.assertLess(results[30].compressed_size_bytes, results[90].compressed_size_bytes)
        self.assertLess(results[30].psnr, results[90].psnr)
        self.assertLess(results[30].ssim, results[90].ssim)

    def test_mask_compression_protection(self):
        """Strict rule: ground-truth masks must NEVER be JPEG compressed."""
        mask_path = Path(self.temp_dir.name) / "case_01_mask.png"
        mask = Image.new("L", (100, 100), color=255)
        mask.save(mask_path)

        with self.assertRaises(ValueError) as ctx:
            ImageCompressor.compress_image(mask_path, quality=70)
        self.assertIn("NEVER be lossy compressed", str(ctx.exception))


class TestImageMetrics(unittest.TestCase):
    """Tests PSNR and SSIM calculations with zero-dummy validation."""

    def test_identical_images(self):
        arr1 = np.full((128, 128, 3), 120, dtype=np.uint8)
        arr2 = arr1.copy()
        psnr = calculate_psnr(arr1, arr2)
        ssim = calculate_ssim(arr1, arr2)
        self.assertEqual(psnr, float("inf"))
        self.assertEqual(ssim, 1.0)

    def test_noisy_images(self):
        arr1 = np.full((128, 128, 3), 120, dtype=np.uint8)
        arr2 = arr1.copy()
        arr2[0, 0] = 200  # Slight noise
        psnr = calculate_psnr(arr1, arr2)
        ssim = calculate_ssim(arr1, arr2)
        self.assertTrue(20.0 < psnr < 100.0)
        self.assertTrue(0.90 < ssim < 1.0)

    def test_shape_mismatch(self):
        arr1 = np.zeros((100, 100), dtype=np.uint8)
        arr2 = np.zeros((120, 120), dtype=np.uint8)
        with self.assertRaises(ValueError):
            calculate_psnr(arr1, arr2)
        with self.assertRaises(ValueError):
            calculate_ssim(arr1, arr2)


if __name__ == "__main__":
    unittest.main()
