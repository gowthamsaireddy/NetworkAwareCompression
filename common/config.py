"""Application configuration for sender and receiver nodes.

Loads dynamic settings from config.json with fallback to robust dataclass defaults
and environment variable overrides.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
LOGS_DIR = BASE_DIR / "logs"
CONFIG_FILE = BASE_DIR / "config.json"


@dataclass
class SenderConfig:
    """Sender node configuration."""
    receiver_ip: str = "127.0.0.1"
    receiver_port: int = 5000
    dataset_path: str = str(DATA_DIR / "BUSI")
    compression_qualities: List[int] = field(default_factory=lambda: [30, 50, 70, 90])
    default_quality: int = 70
    network_interface: str = "eth0"
    socket_timeout: float = 15.0
    max_retries: int = 3
    retry_delay_sec: float = 1.5
    default_network_profile: str = "unshaped"
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReceiverConfig:
    """Receiver node configuration."""
    listen_ip: str = "0.0.0.0"
    listen_port: int = 5000
    save_directory: str = str(DATA_DIR / "received")
    model_checkpoint_path: str = str(MODELS_DIR / "best_unet.pt")
    socket_timeout: float = 30.0
    device: str = "auto"  # "auto", "cuda", or "cpu"
    auto_process_enabled: bool = True
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_raw_config() -> Dict[str, Any]:
    """Loads raw config dict from config.json or returns defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"sender": {}, "receiver": {}}


def save_config(config_dict: Dict[str, Any]) -> None:
    """Persists configuration dictionary to config.json."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)


def get_sender_config() -> SenderConfig:
    """Constructs SenderConfig with config.json and environment variable overrides."""
    raw = load_raw_config().get("sender", {})
    return SenderConfig(
        receiver_ip=os.environ.get("NAC_RECEIVER_IP", raw.get("receiver_ip", "127.0.0.1")),
        receiver_port=int(os.environ.get("NAC_RECEIVER_PORT", raw.get("receiver_port", 5000))),
        dataset_path=os.environ.get("NAC_DATASET_PATH", raw.get("dataset_path", str(DATA_DIR / "BUSI"))),
        compression_qualities=raw.get("compression_qualities", [30, 50, 70, 90]),
        default_quality=int(os.environ.get("NAC_DEFAULT_QUALITY", raw.get("default_quality", 70))),
        network_interface=os.environ.get("NAC_NET_IFACE", raw.get("network_interface", "eth0")),
        socket_timeout=float(raw.get("connection_timeout", raw.get("socket_timeout", 15.0))),
        max_retries=int(raw.get("max_retries", 3)),
        retry_delay_sec=float(raw.get("retry_delay_sec", 1.5)),
        default_network_profile=raw.get("default_network_profile", "unshaped"),
        log_level=os.environ.get("NAC_LOG_LEVEL", raw.get("log_level", "INFO")),
    )


def get_receiver_config() -> ReceiverConfig:
    """Constructs ReceiverConfig with config.json and environment variable overrides."""
    raw = load_raw_config().get("receiver", {})
    default_model = str(MODELS_DIR / "best_unet.pt")
    if not (MODELS_DIR / "best_unet.pt").exists() and (MODELS_DIR / "best_unet.pth").exists():
        default_model = str(MODELS_DIR / "best_unet.pth")

    env_ap = os.environ.get("NAC_AUTO_PROCESS")
    if env_ap is not None:
        auto_process = env_ap.lower() in ("1", "true", "yes")
    else:
        auto_process = bool(raw.get("auto_process_enabled", True))

    return ReceiverConfig(
        listen_ip=os.environ.get("NAC_LISTEN_IP", raw.get("listen_ip", "0.0.0.0")),
        listen_port=int(os.environ.get("NAC_LISTEN_PORT", raw.get("listen_port", 5000))),
        save_directory=os.environ.get("NAC_SAVE_DIR", raw.get("save_directory", str(DATA_DIR / "received"))),
        model_checkpoint_path=os.environ.get("NAC_MODEL_PATH", raw.get("model_checkpoint_path", default_model)),
        socket_timeout=float(raw.get("connection_timeout", raw.get("socket_timeout", 30.0))),
        device=os.environ.get("NAC_DEVICE", raw.get("device", "auto")),
        auto_process_enabled=auto_process,
        log_level=os.environ.get("NAC_LOG_LEVEL", raw.get("log_level", "INFO")),
    )


def ensure_directories():
    """Ensure runtime folders exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "received").mkdir(parents=True, exist_ok=True)
