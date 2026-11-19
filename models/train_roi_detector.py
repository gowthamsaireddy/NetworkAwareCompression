"""Training script for Lightweight Sender ROI Detector.

Trains strictly on the training partition of the BUSI dataset.
Validates on the validation partition.
Evaluates final generalization on the held-out test partition.
Saves weights separately to models/roi_detector_best.pth.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.logging_config import setup_logger
from models.roi_detector import LightweightROIDetector
from receiver.segmentation_metrics import compute_dice_coefficient, compute_iou

logger = setup_logger("TrainROIDetector")


class BUSIROISet(Dataset):
    """Dataset for training and evaluating the Sender ROI Detector."""

    def __init__(self, items: List[Dict], target_size: Tuple[int, int] = (256, 256)):
        self.items = items
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        img_path = Path(item["image_path"])

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.target_size, dtype=np.uint8)
        img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0)

        # Load mask(s)
        mask_paths = [Path(p) for p in item.get("mask_paths", [])]
        combined_mask = np.zeros(self.target_size, dtype=np.uint8)

        for mp in mask_paths:
            if mp.exists():
                m_img = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m_img is not None:
                    m_resized = cv2.resize(m_img, self.target_size, interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.maximum(combined_mask, (m_resized > 0).astype(np.uint8) * 255)

        mask_tensor = torch.from_numpy((combined_mask > 0).astype(np.float32)).unsqueeze(0)
        return img_tensor, mask_tensor


class DiceLoss(nn.Module):
    """Soft Dice Loss for region localization."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        total = probs_flat.sum() + targets_flat.sum()
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Evaluates ROI detector on a dataset partition."""
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_bce = 0.0
    bce_fn = nn.BCEWithLogitsLoss()
    n = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss_bce = bce_fn(logits, masks)
            total_bce += loss_bce.item() * len(imgs)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.35).float()  # Conservative threshold

            for p, m in zip(preds, masks):
                p_np = p.squeeze().cpu().numpy().astype(np.uint8) * 255
                m_np = m.squeeze().cpu().numpy().astype(np.uint8) * 255
                d = compute_dice_coefficient(m_np, p_np)
                j = compute_iou(m_np, p_np)
                total_dice += d
                total_iou += j
                n += 1

    mean_dice = total_dice / max(n, 1)
    mean_iou = total_iou / max(n, 1)
    mean_bce = total_bce / max(n, 1)
    return mean_dice, mean_iou, mean_bce


def train_roi_detector(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    splits_path: str = "data/BUSI/splits.json",
    save_path: str = "models/roi_detector_best.pth",
    history_path: str = "models/roi_detector_history.json",
):
    """Trains the lightweight Sender ROI detector on the BUSI training partition."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for training: {device}")

    # Load BUSI splits
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_items = splits["train"]
    val_items = splits["validation"]
    test_items = splits["test"]

    logger.info(
        f"Loaded splits: {len(train_items)} train, {len(val_items)} validation, {len(test_items)} held-out test"
    )

    train_loader = DataLoader(BUSIROISet(train_items), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(BUSIROISet(val_items), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(BUSIROISet(test_items), batch_size=batch_size, shuffle=False)

    model = LightweightROIDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()

    best_val_dice = 0.0
    history = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "history": [],
        "best_epoch": 0,
        "best_val_dice": 0.0,
        "test_dice": 0.0,
        "test_iou": 0.0,
    }

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            loss_bce = bce_criterion(logits, masks)
            loss_dice = dice_criterion(logits, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)

        scheduler.step()
        train_loss /= len(train_items)

        # Validation
        val_dice, val_iou, val_bce = evaluate(model, val_loader, device)
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f} | Val BCE: {val_bce:.4f}"
        )

        history["history"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "val_bce": val_bce,
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            history["best_epoch"] = epoch
            history["best_val_dice"] = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "model_version": "roi_detector_v1",
                    "parameters_count": sum(p.numel() for p in model.parameters()),
                },
                save_path,
            )
            logger.info(f"[*] Checkpoint saved: {save_path} (Val Dice: {val_dice:.4f})")

    total_training_sec = time.perf_counter() - start_time
    logger.info(f"Training completed in {total_training_sec:.1f}s. Evaluating best checkpoint on held-out test split...")

    # Load best checkpoint and evaluate on held-out test split
    best_ckpt = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_dice, test_iou, test_bce = evaluate(model, test_loader, device)
    history["test_dice"] = test_dice
    history["test_iou"] = test_iou
    history["test_bce"] = test_bce
    history["training_duration_sec"] = total_training_sec

    logger.info(
        f"[TEST RESULTS] Held-Out Test Dice: {test_dice:.4f} | "
        f"Test IoU: {test_iou:.4f} | Test BCE: {test_bce:.4f}"
    )

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Saved training history to {history_path}")
    return history


if __name__ == "__main__":
    train_roi_detector(epochs=20, batch_size=16)
