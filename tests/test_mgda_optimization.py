"""Unit tests for Multiple Gradient Descent Algorithm (MGDA) loss optimization."""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.train_unet import compute_mgda_lambda, train_epoch_mgda, DiceLoss
from models.unet import UNet


class TestMGDAOptimization(unittest.TestCase):
    """Verifies mathematical correctness of the closed-form MGDA solver and PyTorch gradient blending."""

    def test_orthogonal_gradients_equal_norm(self):
        # g1 = [1, 0], g2 = [0, 1]
        # ||g2||^2 = 1, <g1, g2> = 0, ||g1 - g2||^2 = 2
        # hat_lambda = (1 - 0) / 2 = 0.5
        g1 = [torch.tensor([1.0, 0.0])]
        g2 = [torch.tensor([0.0, 1.0])]
        lam = compute_mgda_lambda(g1, g2)
        self.assertAlmostEqual(lam, 0.5, places=5)

    def test_orthogonal_gradients_unequal_norm(self):
        # g1 = [2, 0] (norm^2 = 4), g2 = [0, 1] (norm^2 = 1)
        # ||g2||^2 = 1, <g1, g2> = 0, ||g1 - g2||^2 = 5
        # hat_lambda = 1 / 5 = 0.2
        g1 = [torch.tensor([2.0, 0.0])]
        g2 = [torch.tensor([0.0, 1.0])]
        lam = compute_mgda_lambda(g1, g2)
        self.assertAlmostEqual(lam, 0.2, places=5)

    def test_boundary_clipping_lower(self):
        # g1 and g2 such that projection is < 0 -> should clip to 0.0
        # g1 = [1.0], g2 = [2.0]
        # ||g2||^2 = 4, <g1, g2> = 2, ||g1 - g2||^2 = 1
        # hat_lambda = (4 - 2) / 1 = 2 -> clips to 1.0
        # Conversely, g1 = [2.0], g2 = [1.0]:
        # ||g2||^2 = 1, <g1, g2> = 2, ||g1 - g2||^2 = 1
        # hat_lambda = (1 - 2) / 1 = -1.0 -> clips to 0.0
        g1 = [torch.tensor([2.0])]
        g2 = [torch.tensor([1.0])]
        lam = compute_mgda_lambda(g1, g2)
        self.assertEqual(lam, 0.0)

    def test_boundary_clipping_upper(self):
        g1 = [torch.tensor([1.0])]
        g2 = [torch.tensor([2.0])]
        lam = compute_mgda_lambda(g1, g2)
        self.assertEqual(lam, 1.0)

    def test_identical_gradients(self):
        # When g1 == g2, ||g1 - g2||^2 = 0 -> fallback to 0.5
        g1 = [torch.tensor([1.5, -2.0])]
        g2 = [torch.tensor([1.5, -2.0])]
        lam = compute_mgda_lambda(g1, g2)
        self.assertAlmostEqual(lam, 0.5, places=5)

    def test_invalid_gradients_nan_fallback(self):
        # When g1 has NaN, fallback to 0.0 (BCE)
        g1 = [torch.tensor([float("nan"), 1.0])]
        g2 = [torch.tensor([1.0, 2.0])]
        lam = compute_mgda_lambda(g1, g2)
        self.assertEqual(lam, 0.0)

        # When g2 has Inf, fallback to 1.0 (Dice)
        g1 = [torch.tensor([1.0, 1.0])]
        g2 = [torch.tensor([float("inf"), 2.0])]
        lam = compute_mgda_lambda(g1, g2)
        self.assertEqual(lam, 1.0)

    def test_mgda_train_step_execution(self):
        # Mini U-Net with synthetic data
        device = torch.device("cpu")
        model = UNet(in_channels=1, out_channels=1, base_filters=8).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        dice_criterion = DiceLoss(smooth=1.0)
        bce_criterion = nn.BCEWithLogitsLoss()

        # Create dummy loader with 2 batches of (1, 1, 32, 32)
        x = torch.randn(4, 1, 32, 32)
        y = torch.randint(0, 2, (4, 1, 32, 32)).float()
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)

        mean_loss, mean_bce, mean_dice, mean_lambda = train_epoch_mgda(
            model, loader, optimizer, dice_criterion, bce_criterion, device
        )

        self.assertGreater(mean_loss, 0.0)
        self.assertGreater(mean_bce, 0.0)
        self.assertGreater(mean_dice, 0.0)
        self.assertGreaterEqual(mean_lambda, 0.0)
        self.assertLessEqual(mean_lambda, 1.0)


if __name__ == "__main__":
    unittest.main()
