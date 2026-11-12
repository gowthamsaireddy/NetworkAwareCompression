"""PyTorch U-Net architecture for breast ultrasound lesion segmentation.

Designed for 256x256 single-channel grayscale breast ultrasound images.
Produces 1-channel raw logits matching the Receiver's SegmentationEngine:
- Input shape:  (B, 1, 256, 256) normalized float32 in [0.0, 1.0]
- Output shape: (B, 1, 256, 256) unnormalized logits (evaluated with sigmoid(logits) >= 0.5)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection concatenation"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # in_channels is the total number of channels entering the up block
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Input is CHW. Handle possible padding differences if spatial dims differ slightly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX != 0 or diffY != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net for binary lesion segmentation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)        # 32 -> 64
        self.down2 = Down(base_filters * 2, base_filters * 4)    # 64 -> 128
        self.down3 = Down(base_filters * 4, base_filters * 8)    # 128 -> 256
        self.down4 = Down(base_filters * 8, base_filters * 16)   # 256 -> 512 (Bottleneck)

        # Decoder
        self.up1 = Up(base_filters * 16, base_filters * 8)       # 512 -> 256
        self.up2 = Up(base_filters * 8, base_filters * 4)        # 256 -> 128
        self.up3 = Up(base_filters * 4, base_filters * 2)        # 128 -> 64
        self.up4 = Up(base_filters * 2, base_filters)            # 64 -> 32
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

