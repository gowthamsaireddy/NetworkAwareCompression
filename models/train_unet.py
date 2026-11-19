"""Reproducible U-Net Training Pipeline for Breast Ultrasound Lesion Segmentation.

Strictly adheres to:
1. Training only on splits['train'] (546 samples).
2. Validation only on splits['validation'] (117 samples) for checkpoint selection.
3. Held-out splits['test'] is NEVER accessed during training/validation.
4. Paired sample alignment: normal images have empty (all-zeros) target masks,
   and multi-lesion images combine masks via np.maximum.
5. Saves best validation checkpoint as models/best_unet.pth and exports training_history.json.
6. Zero fabrication: all metrics, losses, and timings are measured directly.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNet


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BUSIDataset(Dataset):
    """PyTorch Dataset for paired BUSI ultrasound scans and ground-truth lesion masks."""

    def __init__(self, items: List[Dict], is_train: bool = False, target_size: Tuple[int, int] = (256, 256)):
        self.items = items
        self.is_train = is_train
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        img_path = PROJECT_ROOT / item["image_path"]

        # Step 1: Load and preprocess image (identical to receiver/segmentation.py)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback to PIL
            pil_img = Image.open(img_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_256 = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_norm = gray_256.astype(np.float32) / 255.0

        # Step 2: Load and combine all ground-truth masks
        mask_paths = [PROJECT_ROOT / p for p in item["mask_paths"]]
        if not mask_paths or item["category"] == "normal":
            # Normal scans have no lesions -> all zeros mask
            mask_256 = np.zeros(self.target_size, dtype=np.float32)
        else:
            combined_mask = None
            for mp in mask_paths:
                m_raw = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m_raw is None:
                    # Fallback to PIL with grayscale conversion
                    m_raw = np.array(Image.open(mp).convert("L"))
                if combined_mask is None:
                    combined_mask = m_raw
                else:
                    combined_mask = np.maximum(combined_mask, m_raw)

            # Resize with nearest-neighbor to avoid fractional mask interpolation artifacts
            mask_resized = cv2.resize(combined_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask_256 = (mask_resized > 127).astype(np.float32)

        # Step 3: Data augmentation (training only)
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_norm = np.fliplr(img_norm)
                mask_256 = np.fliplr(mask_256)
            # Random Vertical Flip
            if random.random() > 0.8:
                img_norm = np.flipud(img_norm)
                mask_256 = np.flipud(mask_256)

        # Step 4: Convert to PyTorch Tensors (1, 256, 256)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_norm)).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_256)).unsqueeze(0)

        return img_tensor, mask_tensor


class DiceLoss(nn.Module):
    """Soft Dice Loss with Laplace smoothing."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """Compound BCEWithLogitsLoss + DiceLoss for robust medical lesion boundary segmentation."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float]:
    """Computes exact mathematical Dice Similarity Coefficient and IoU."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) >= threshold).float()
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum().item()
        total_pred = preds_flat.sum().item()
        total_target = targets_flat.sum().item()

        # Handle empty mask edge-cases (normal healthy scans)
        if total_target == 0 and total_pred == 0:
            return 1.0, 1.0  # Perfect true negative
        elif total_target == 0 or total_pred == 0:
            return 0.0, 0.0

        dice = (2.0 * intersection) / (total_pred + total_target)
        union = total_pred + total_target - intersection
        iou = intersection / union if union > 0 else 0.0

        return dice, iou


def compute_mgda_lambda(
    g_dice: List[torch.Tensor],
    g_bce: List[torch.Tensor],
    eps: float = 1e-8,
) -> float:
    """Computes optimal lambda* for combining Dice (g_dice) and BCE (g_bce) gradients via MGDA (Sener & Koltun 2018).

    Solves: min_{0 <= lambda <= 1} ||lambda*g_dice + (1-lambda)*g_bce||^2
    Analytical closed-form:
      hat_lambda = (||g_bce||^2 - <g_dice, g_bce>) / ||g_dice - g_bce||^2
      lambda* = clip(hat_lambda, 0.0, 1.0)
    Combined descent vector:
      g* = lambda* * g_dice + (1 - lambda*) * g_bce
    """
    # Safety Check 1: Check for invalid gradients (NaN or Inf)
    g_dice_valid = all(torch.isfinite(g).all() for g in g_dice if g is not None)
    g_bce_valid = all(torch.isfinite(g).all() for g in g_bce if g is not None)

    if not g_dice_valid and not g_bce_valid:
        return 0.5
    elif not g_dice_valid:
        return 0.0  # Fallback entirely to BCE
    elif not g_bce_valid:
        return 1.0  # Fallback entirely to Dice

    g1_dot_g1 = sum((g * g).sum() for g in g_dice if g is not None)
    g2_dot_g2 = sum((g * g).sum() for g in g_bce if g is not None)
    g1_dot_g2 = sum((g_a * g_b).sum() for g_a, g_b in zip(g_dice, g_bce) if g_a is not None and g_b is not None)

    # ||g_dice - g_bce||^2 = ||g_dice||^2 + ||g_bce||^2 - 2 * <g_dice, g_bce>
    denom = g1_dot_g1 + g2_dot_g2 - 2.0 * g1_dot_g2

    # Safety Check 2: Near-zero denominator or degenerate zero gradient distance
    if torch.isnan(denom) or torch.isinf(denom) or denom.item() < eps:
        return 0.5

    hat_lambda = (g2_dot_g2 - g1_dot_g2) / denom

    # Safety Check 3: Ensure finite scalar before clamping
    if torch.isnan(hat_lambda) or torch.isinf(hat_lambda):
        return 0.5

    lambda_star = torch.clamp(hat_lambda, 0.0, 1.0).item()
    return float(lambda_star)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Trains for one epoch over the training split using fixed scalar weighting."""
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0

    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss(smooth=1.0)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        bce_val = bce_fn(logits, masks).item()
        dice_val = dice_fn(logits, masks).item()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_bce += bce_val * bs
        running_dice += dice_val * bs

    n = len(loader.dataset)
    return running_loss / n, running_bce / n, running_dice / n


def train_epoch_mgda(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dice_criterion: nn.Module,
    bce_criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Trains for one epoch using Multiple Gradient Descent Algorithm (MGDA).

    Dynamically calculates lambda* for every mini-batch to find the minimum-norm
    common descent vector in the convex hull of Dice and BCE gradients.

    Returns:
        (mean_loss, mean_bce, mean_dice, mean_lambda)
    """
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_dice = 0.0
    running_lambda = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        # Step 1: Forward pass
        logits = model(images)
        loss_dice = dice_criterion(logits, masks)
        loss_bce = bce_criterion(logits, masks)

        # Step 2: Backward pass for Dice Loss (retain computational graph for BCE)
        optimizer.zero_grad()
        loss_dice.backward(retain_graph=True)
        g_dice = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

        # Step 3: Backward pass for BCE Loss
        optimizer.zero_grad()
        loss_bce.backward()
        g_bce = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

        # Step 4: Solve analytical min-norm problem for optimal lambda*
        lambda_star = compute_mgda_lambda(g_dice, g_bce)

        # Step 5: Construct Pareto-optimal gradient: g* = lambda* * g_dice + (1 - lambda*) * g_bce
        for p, grad_dice, grad_bce in zip(model.parameters(), g_dice, g_bce):
            if p.grad is not None and grad_dice is not None and grad_bce is not None:
                p.grad.copy_(lambda_star * grad_dice + (1.0 - lambda_star) * grad_bce)

        # Step 6: Parameter update
        optimizer.step()

        batch_loss = lambda_star * loss_dice.item() + (1.0 - lambda_star) * loss_bce.item()
        bs = images.size(0)
        running_loss += batch_loss * bs
        running_bce += loss_bce.item() * bs
        running_dice += loss_dice.item() * bs
        running_lambda += lambda_star

    dataset_size = len(loader.dataset)
    num_batches = max(len(loader), 1)
    mean_loss = running_loss / dataset_size
    mean_bce = running_bce / dataset_size
    mean_dice = running_dice / dataset_size
    mean_lambda = running_lambda / num_batches
    return mean_loss, mean_bce, mean_dice, mean_lambda


def validate(
    model: nn.Module,
    loader: DataLoader,
    dice_criterion: nn.Module,
    bce_criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    """Evaluates model strictly on the validation split.

    Returns:
        (val_loss, val_bce, val_dice_loss, val_dice, val_iou)
    """
    model.eval()
    running_bce = 0.0
    running_dice = 0.0
    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            bce = bce_criterion(logits, masks)
            dice_l = dice_criterion(logits, masks)

            bs = images.size(0)
            running_bce += bce.item() * bs
            running_dice += dice_l.item() * bs

            # Evaluate each sample in batch individually for precise metric averaging
            for i in range(images.size(0)):
                d, j = calculate_metrics(logits[i : i + 1], masks[i : i + 1])
                dice_scores.append(d)
                iou_scores.append(j)

    dataset_size = len(loader.dataset)
    val_bce = running_bce / dataset_size
    val_dice_loss = running_dice / dataset_size
    val_loss = 0.5 * val_bce + 0.5 * val_dice_loss
    val_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    val_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return val_loss, val_bce, val_dice_loss, val_dice, val_iou


def run_training(
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 7,
    seed: int = 42,
    device_str: Optional[str] = None,
    use_mgda: bool = True,
) -> Dict:
    """Executes U-Net training on the training split with validation checkpoint selection."""
    set_seed(seed)

    # 1. Device selection
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[+] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 2. Load splits
    splits_path = PROJECT_ROOT / "data" / "BUSI" / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found at {splits_path}")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_items = splits["train"]
    val_items = splits["validation"]
    print(f"[+] Loaded {len(train_items)} training samples, {len(val_items)} validation samples.")
    print(f"[+] Held-out test set: {len(splits['test'])} samples (STRICTLY UNTOUCHED).")

    # 3. Create datasets and loaders
    train_dataset = BUSIDataset(train_items, is_train=True)
    val_dataset = BUSIDataset(val_items, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Initialize model, optimizer, scheduler, and loss
    model = UNet(in_channels=1, out_channels=1, base_filters=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Initialized UNet (Trainable Parameters: {total_params:,})")

    criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    dice_criterion = DiceLoss(smooth=1.0)
    bce_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 5. Training loop tracking
    best_val_dice = -1.0
    best_val_iou = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_bce_loss": [],
        "train_dice_loss": [],
        "mean_lambda": [],
        "val_loss": [],
        "val_bce_loss": [],
        "val_dice_loss": [],
        "val_dice": [],
        "val_iou": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "best_unet_mgda.pth" if use_mgda else "best_unet.pth"
    best_checkpoint_path = models_dir / checkpoint_name

    start_total_time = time.perf_counter()
    print("\n" + "=" * 105)
    if use_mgda:
        print("[+] Optimization Mode: MGDA (Multiple Gradient Descent Algorithm - Sener & Koltun 2018)")
    else:
        print("[+] Optimization Mode: Standard Fixed BCE+Dice (50/50)")
    print(f"{'Epoch':<6} | {'Tr Loss':<9} | {'Tr BCE':<9} | {'Tr Dice':<9} | {'Val Loss':<9} | {'Val Dice':<9} | {'Val IoU':<9} | {'Mean λ*':<9} | {'Time (s)':<8} | {'Status'}")
    print("=" * 105)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        if use_mgda:
            train_loss, tr_bce, tr_dice, mean_lambda = train_epoch_mgda(
                model, train_loader, optimizer, dice_criterion, bce_criterion, device
            )
        else:
            train_loss, tr_bce, tr_dice = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            mean_lambda = 0.5

        val_loss, val_bce, val_dice_l, val_dice, val_iou = validate(
            model, val_loader, dice_criterion, bce_criterion, device
        )
        scheduler.step()

        elapsed = time.perf_counter() - t0
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_bce_loss"].append(tr_bce)
        history["train_dice_loss"].append(tr_dice)
        history["mean_lambda"].append(mean_lambda)
        history["val_loss"].append(val_loss)
        history["val_bce_loss"].append(val_bce)
        history["val_dice_loss"].append(val_dice_l)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["lr"].append(curr_lr)
        history["epoch_time_sec"].append(elapsed)

        # Checkpoint selection based on Validation Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_val_iou = val_iou
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint dictionary
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "val_iou": best_val_iou,
                "val_loss": best_val_loss,
                "train_loss": train_loss,
                "config": {
                    "architecture": "UNet",
                    "in_channels": 1,
                    "out_channels": 1,
                    "base_filters": 32,
                    "input_size": [256, 256],
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "seed": seed,
                    "device": str(device),
                    "optimization_mode": "MGDA" if use_mgda else "Standard_BCE_Dice_0.5",
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            status = "(*) Best Saved"
        else:
            patience_counter += 1
            status = f"Patience: {patience_counter}/{patience}"

        print(
            f"{epoch:<6} | {train_loss:<9.4f} | {tr_bce:<9.4f} | {tr_dice:<9.4f} | {val_loss:<9.4f} | {val_dice:<9.4f} | {val_iou:<9.4f} | {mean_lambda:<9.4f} | {elapsed:<8.2f} | {status}"
        )

        if patience_counter >= patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch} (No validation improvement for {patience} epochs).")
            break

    total_training_sec = time.perf_counter() - start_total_time
    print("=" * 105)
    print(f"[+] Training completed in {total_training_sec:.2f} seconds.")
    print(f"[+] Best Validation Checkpoint: Epoch {best_epoch} with Dice={best_val_dice:.4f}, IoU={best_val_iou:.4f}, Loss={best_val_loss:.4f}")
    print(f"[+] Checkpoint saved to: {best_checkpoint_path}")

    # 6. Save training history JSON
    history_name = "training_history_mgda.json" if use_mgda else "training_history.json"
    history_path = models_dir / history_name
    history_payload = {
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_iou": best_val_iou,
        "best_val_loss": best_val_loss,
        "total_training_time_sec": total_training_sec,
        "device": str(device),
        "history": history,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"[+] Training history exported to: {history_path}")

    return history_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on BUSI dataset using MGDA or fixed loss weighting.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--use-fixed", action="store_true", help="Use legacy fixed lambda=0.5 weighting instead of MGDA")
    parser.add_argument("--mgda", action="store_true", default=True, help="Train using MGDA (default: True)")
    args = parser.parse_args()

    use_mgda = not args.use_fixed

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        device_str=args.device,
        use_mgda=use_mgda,
    )

