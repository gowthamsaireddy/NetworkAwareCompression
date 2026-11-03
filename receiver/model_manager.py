"""Model checkpoint manager for receiver node.

Maintains strict adherence to zero-dummy policy:
- Does NOT fabricate a model architecture or fake weights.
- Loads pre-trained checkpoints (TorchScript, full nn.Module, or state_dict).
- Enforces frozen evaluation mode (eval(), requires_grad=False).
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn

from common.logging_config import setup_logger

logger = setup_logger("ModelManager")


class ModelManager:
    """Manages pre-trained deep learning checkpoints in frozen evaluation mode."""

    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.checkpoint_path: Optional[Path] = None
        self.model_name: str = "None"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded: bool = False
        self._model_factory: Optional[Callable[[], nn.Module]] = None

    def register_architecture(self, factory: Callable[[], nn.Module]) -> None:
        """Registers the external U-Net architecture constructor for state_dict checkpoints."""
        self._model_factory = factory
        logger.info("Registered external U-Net architecture factory.")

    def load_model_instance(self, model: nn.Module, model_name: str = "Frozen-UNet") -> Tuple[bool, str]:
        """Directly mounts an instantiated PyTorch model and freezes all weights."""
        try:
            self.model = model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

            self.model_name = model_name
            self.checkpoint_path = None
            self.is_loaded = True
            msg = f"Model '{model_name}' successfully mounted on {self.device} (Frozen mode)."
            logger.info(msg)
            return True, msg
        except Exception as e:
            self.unload_model()
            err = f"Failed to mount model instance: {e}"
            logger.error(err)
            return False, err

    def load_checkpoint(self, checkpoint_path: Path) -> Tuple[bool, str]:
        """Loads a pre-trained PyTorch checkpoint file and enforces frozen evaluation mode."""
        if not checkpoint_path.exists():
            self.unload_model()
            return False, f"Checkpoint file not found: {checkpoint_path}"

        try:
            # 1. Attempt loading as TorchScript module first
            try:
                scripted = torch.jit.load(str(checkpoint_path), map_location=self.device)
                scripted.eval()
                for p in scripted.parameters():
                    p.requires_grad = False
                self.model = scripted
                self.checkpoint_path = checkpoint_path
                self.model_name = checkpoint_path.name
                self.is_loaded = True
                msg = f"Loaded TorchScript checkpoint '{checkpoint_path.name}' on {self.device} (Frozen)."
                logger.info(msg)
                return True, msg
            except Exception:
                pass  # Fall back to standard PyTorch load

            # 2. Standard torch.load
            loaded_obj = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if isinstance(loaded_obj, nn.Module):
                # Full serialized model object
                self.model = loaded_obj.to(self.device)
            elif isinstance(loaded_obj, dict):
                # Checkpoint is a state_dict or training dict containing 'model_state_dict' / 'state_dict'
                state_dict = loaded_obj.get("state_dict") or loaded_obj.get("model_state_dict") or loaded_obj

                if self._model_factory is not None:
                    instance = self._model_factory()
                    instance.load_state_dict(state_dict)
                    self.model = instance.to(self.device)
                else:
                    return False, (
                        f"Checkpoint '{checkpoint_path.name}' contains a state_dict, but no U-Net architecture "
                        "has been registered. Register architecture or provide a full model/TorchScript file."
                    )
            else:
                return False, f"Unsupported checkpoint format in {checkpoint_path.name}: {type(loaded_obj)}"

            # 3. Freeze all parameters strictly
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False

            self.checkpoint_path = checkpoint_path
            self.model_name = checkpoint_path.name
            self.is_loaded = True
            msg = f"Loaded checkpoint '{checkpoint_path.name}' on {self.device} (Frozen)."
            logger.info(msg)
            return True, msg

        except Exception as e:
            self.unload_model()
            err = f"Failed to load checkpoint: {e}"
            logger.error(err)
            return False, err

    def unload_model(self) -> None:
        """Safely unloads the active model and releases accelerator memory."""
        self.model = None
        self.checkpoint_path = None
        self.model_name = "None"
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    def get_status(self) -> str:
        """Returns verified human-readable model status."""
        if not self.is_loaded or self.model is None:
            return "No model loaded"
        source = self.checkpoint_path.name if self.checkpoint_path else self.model_name
        return f"Loaded: {source} ({self.device}) [Frozen]"
