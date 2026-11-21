"""Tests for Phase 5: Deep Learning Segmentation, Model Management, and Evaluation Metrics."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from receiver.model_manager import ModelManager
from receiver.segmentation import SegmentationEngine, SegmentationResult, create_lesion_overlay
from receiver.segmentation_metrics import (
    compute_dice_coefficient,
    compute_iou,
    create_comparative_overlay,
    evaluate_segmentation,
    get_canonical_busi_id,
    locate_ground_truth_mask,
    locate_original_uncompressed,
)
from ui.receiver.results_view import calculate_pareto_score


class SimpleUNetMock(nn.Module):
    """Isolated, minimal mock module for unit testing frozen evaluation pipeline."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestModelManager(unittest.TestCase):
    """Tests model checkpoint mounting, parameter freezing, and zero-dummy states."""

    def test_initial_unloaded_state(self):
        mgr = ModelManager()
        self.assertFalse(mgr.is_loaded)
        self.assertIsNone(mgr.model)
        self.assertEqual(mgr.get_status(), "No model loaded")

    def test_load_and_freeze_model_instance(self):
        mgr = ModelManager()
        raw_model = SimpleUNetMock()

        # Ensure parameters originally require grad
        for p in raw_model.parameters():
            p.requires_grad = True

        success, msg = mgr.load_model_instance(raw_model, model_name="MockUNet")
        self.assertTrue(success)
        self.assertTrue(mgr.is_loaded)
        self.assertIn("MockUNet", mgr.get_status())
        self.assertIn("Frozen", mgr.get_status())

        # Parameters must be strictly frozen
        for p in mgr.model.parameters():
            self.assertFalse(p.requires_grad)

        # Model must be in evaluation mode
        self.assertFalse(mgr.model.training)

        # Test unload
        mgr.unload_model()
        self.assertFalse(mgr.is_loaded)
        self.assertEqual(mgr.get_status(), "No model loaded")


class TestSegmentationInference(unittest.TestCase):
    """Tests frozen segmentation pipeline: decode -> gray -> 256x256 -> norm -> infer -> mask -> overlay."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img_path = Path(self.temp_dir.name) / "case_01.png"

        # Create gradient test ultrasound image
        arr = np.linspace(50, 200, 300 * 400, dtype=np.uint8).reshape((300, 400))
        img = Image.fromarray(arr).convert("RGB")
        img.save(self.img_path)

        self.mgr = ModelManager()
        self.mgr.load_model_instance(SimpleUNetMock(), model_name="MockUNet")
        self.engine = SegmentationEngine(self.mgr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_run_inference_pipeline(self):
        success, result, msg = self.engine.run_inference(self.img_path, "benign/case_01")
        self.assertTrue(success, msg)
        self.assertIsNotNone(result)

        # Verify shapes and dimensions
        self.assertEqual(result.original_shape, (300, 400))
        self.assertEqual(result.processed_image.shape, (256, 256))
        self.assertEqual(result.probability_map.shape, (256, 256))
        self.assertEqual(result.predicted_mask.shape, (256, 256))
        self.assertEqual(result.overlay_image.shape, (256, 256, 3))

        # Binary mask values must be strictly 0 or 255
        unique_vals = set(np.unique(result.predicted_mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

        # Measured execution time
        self.assertGreater(result.inference_time_ms, 0.0)

    def test_inference_refusal_when_unloaded(self):
        self.mgr.unload_model()
        success, result, msg = self.engine.run_inference(self.img_path, "benign/case_01")
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertIn("No frozen deep learning model loaded", msg)


class TestSegmentationMetrics(unittest.TestCase):
    """Tests genuine Dice, IoU, comparative overlays, and ground-truth search."""

    def test_dice_and_iou_exact_match(self):
        mask1 = np.zeros((256, 256), dtype=np.uint8)
        mask1[50:150, 50:150] = 255
        mask2 = mask1.copy()

        dice = compute_dice_coefficient(mask1, mask2)
        iou = compute_iou(mask1, mask2)
        self.assertAlmostEqual(dice, 1.0, places=4)
        self.assertAlmostEqual(iou, 1.0, places=4)

    def test_dice_and_iou_disjoint(self):
        mask1 = np.zeros((256, 256), dtype=np.uint8)
        mask1[10:50, 10:50] = 255

        mask2 = np.zeros((256, 256), dtype=np.uint8)
        mask2[100:150, 100:150] = 255

        dice = compute_dice_coefficient(mask1, mask2)
        iou = compute_iou(mask1, mask2)
        self.assertAlmostEqual(dice, 0.0, places=3)
        self.assertAlmostEqual(iou, 0.0, places=3)

    def test_dice_and_iou_partial_overlap(self):
        # A: [0:10, 0:10] -> area 100
        # B: [0:10, 5:15] -> area 100
        # Intersection: [0:10, 5:10] -> area 50
        mask_a = np.zeros((20, 20), dtype=np.uint8)
        mask_a[0:10, 0:10] = 255

        mask_b = np.zeros((20, 20), dtype=np.uint8)
        mask_b[0:10, 5:15] = 255

        # Expected Dice: (2 * 50) / (100 + 100) = 0.5000
        # Expected IoU: 50 / (100 + 100 - 50) = 50 / 150 = 0.3333
        dice = compute_dice_coefficient(mask_a, mask_b)
        iou = compute_iou(mask_a, mask_b)
        self.assertAlmostEqual(dice, 0.5000, places=3)
        self.assertAlmostEqual(iou, 0.3333, places=3)

    def test_zero_dummy_policy_when_gt_missing(self):
        pred_mask = np.ones((256, 256), dtype=np.uint8) * 255
        dice, iou, msg = evaluate_segmentation(None, pred_mask)
        self.assertIsNone(dice)
        self.assertIsNone(iou)
        self.assertIn("unavailable", msg)

    def test_comparative_overlay_generation(self):
        base = np.full((256, 256), 100, dtype=np.uint8)
        gt = np.zeros((256, 256), dtype=np.uint8)
        gt[50:100, 50:100] = 255
        pred = np.zeros((256, 256), dtype=np.uint8)
        pred[80:120, 80:120] = 255

        overlay = create_comparative_overlay(base, gt, pred)
        self.assertEqual(overlay.shape, (256, 256, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_canonical_busi_id_parsing(self):
        cases = [
            ("benign/benign (1)", "benign/benign (1)", "benign (1)", "benign", "benign (1).png"),
            ("benign (1)", "benign/benign (1)", "benign (1)", "benign", "benign (1).png"),
            ("benign_benign (148)_Q90_RUN_20261121_123652_ROI", "benign/benign (148)", "benign (148)", "benign", "benign (148).png"),
            ("BG70_Satellite_benign_benign__148_", "benign/benign (148)", "benign (148)", "benign", "benign (148).png"),
            ("malignant/malignant (23)", "malignant/malignant (23)", "malignant (23)", "malignant", "malignant (23).png"),
            ("normal/normal (5)", "normal/normal (5)", "normal (5)", "normal", "normal (5).png"),
        ]
        for raw, exp_id, exp_stem, exp_cat, exp_fn in cases:
            orig_id, stem, cat, fn = get_canonical_busi_id(raw)
            self.assertEqual(orig_id, exp_id)
            self.assertEqual(stem, exp_stem)
            self.assertEqual(cat, exp_cat)
            self.assertEqual(fn, exp_fn)

    def test_ground_truth_and_orig_locating(self):
        from common.config import DATA_DIR
        search_roots = [DATA_DIR / "BUSI"]

        # 1. Single mask image
        gt_mask, mask_id, paths = locate_ground_truth_mask("benign/benign (1)", search_roots)
        self.assertIsNotNone(gt_mask)
        self.assertEqual(gt_mask.shape, (256, 256))
        self.assertEqual(len(paths), 1)
        self.assertIn("benign (1)_mask.png", mask_id)

        orig_p = locate_original_uncompressed("benign/benign (1)", search_roots)
        self.assertIsNotNone(orig_p)
        self.assertEqual(orig_p.name, "benign (1).png")

        # 2. Multi-mask image (benign (100) has _mask and _mask_1)
        gt_mask_multi, mask_id_m, paths_m = locate_ground_truth_mask("benign/benign (100)", search_roots)
        self.assertIsNotNone(gt_mask_multi)
        self.assertEqual(len(paths_m), 2)
        self.assertIn("benign (100)_mask.png", mask_id_m)
        self.assertIn("benign (100)_mask_1.png", mask_id_m)

        # 3. Nonexistent image must return None per zero-dummy data policy
        gt_none, _, paths_none = locate_ground_truth_mask("nonexistent_phantom_img_9999", search_roots)
        self.assertIsNone(gt_none)
        self.assertEqual(len(paths_none), 0)

    def test_pareto_tradeoff_score_calculation(self):
        # High quality, moderate compression
        score_high = calculate_pareto_score(
            dice=0.92, iou=0.85, ssim=0.98, psnr=42.0, comp_ratio=8.5, tx_time=0.15
        )
        self.assertGreater(score_high, 80.0)
        self.assertLessEqual(score_high, 100.0)

        # Poor segmentation should penalize the score heavily
        score_low = calculate_pareto_score(
            dice=0.20, iou=0.10, ssim=0.60, psnr=25.0, comp_ratio=25.0, tx_time=0.01
        )
        self.assertLess(score_low, 60.0)
        self.assertGreater(score_high, score_low)


if __name__ == "__main__":
    unittest.main()

