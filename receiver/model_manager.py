"""Model checkpoint manager for receiver node.

Maintains strict adherence to zero-dummy policy:
- Does NOT fabricate a model architecture or fake weights.
- Expects an external pre-trained checkpoint to be loaded in Phase 5.
- Enforces frozen evaluation mode (torch.no_grad, eval()).
"""

from pathlib import Path
from typing import Optional, Tuple
import torch


class ModelManager:
    """Manages frozen deep learning model checkpoint loading and state."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.checkpoint_path: Optional[Path] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded: bool = False

    def load_checkpoint(self, checkpoint_path: Path) -> Tuple[bool, str]:
        """Loads a pre-trained checkpoint into memory and freezes weights.

        Actual architecture definition and loading logic will be supplied in Phase 5.
        """
        if not checkpoint_path.exists():
            self.is_loaded = False
            return False, f"Checkpoint file not found: {checkpoint_path}"

        # Real checkpoint loading scheduled for Phase 5 when architecture is supplied
        return False, "Model architecture and checkpoint loading will be integrated in Phase 5."

    def unload_model(self) -> None:
        """Unloads current model from memory."""
        self.model = None
        self.checkpoint_path = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_status(self) -> str:
        """Returns verified human-readable model status."""
        if not self.is_loaded or self.model is None:
            return "No model loaded"
        return f"Loaded: {self.checkpoint_path.name} on {self.device} (Frozen)"

