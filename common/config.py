"""Application configuration for sender and receiver nodes."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
LOGS_DIR = BASE_DIR / "logs"


@dataclass
class SenderConfig:
    """Sender node configuration."""
    receiver_ip: str = "127.0.0.1"
    receiver_port: int = 5000
    dataset_path: str = str(DATA_DIR / "BUSI")
    compression_qualities: List[int] = field(default_factory=lambda: [30, 50, 70, 90])
    default_quality: int = 70
    network_interface: str = "eth0"
    socket_timeout: float = 10.0


@dataclass
class ReceiverConfig:
    """Receiver node configuration."""
    listen_ip: str = "0.0.0.0"
    listen_port: int = 5000
    save_directory: str = str(DATA_DIR / "received")
    model_checkpoint_path: str = str(MODELS_DIR / "unet_busi.pth")
    socket_timeout: float = 15.0
    device: str = "cpu"  # Or "cuda" if available


def ensure_directories():
    """Ensure runtime folders exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "received").mkdir(parents=True, exist_ok=True)

