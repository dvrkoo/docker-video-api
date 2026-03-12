from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch
import torchvision

logger = logging.getLogger(__name__)


class TorchRGBModel:
    def __init__(self, name: str, model: torch.nn.Module, device: torch.device) -> None:
        self.name = name
        self.model = model
        self.device = device
        self.post = torch.nn.Softmax(dim=1)

    def predict_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out = self.model(batch_tensor)
            probs = self.post(out)
        return probs[:, 1]


class TorchScriptRGBModel:
    def __init__(self, name: str, model: torch.jit.ScriptModule, device: torch.device) -> None:
        self.name = name
        self.model = model
        self.device = device
        self.post = torch.nn.Softmax(dim=1)

    def predict_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            out = self.model(batch_tensor)
            probs = self.post(out)
        if probs.ndim == 1:
            return probs.unsqueeze(0)[..., -1]
        return probs[:, -1]


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        while normalized.startswith("module.") or normalized.startswith("model."):
            if normalized.startswith("module."):
                normalized = normalized[len("module.") :]
                continue
            if normalized.startswith("model."):
                normalized = normalized[len("model.") :]
                continue
        cleaned[normalized] = value
    return cleaned


def _load_resnet50_binary(path: Path, device: torch.device) -> torch.nn.Module:
    model = torchvision.models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is not a valid state dict")

    state_dict = _clean_state_dict_keys(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _try_load_model(path: Path, device: torch.device):
    try:
        ts_model = torch.jit.load(path, map_location=device)
        ts_model = ts_model.to(device)
        ts_model.eval()
        logger.info("Loaded TorchScript model: %s", path.name)
        return TorchScriptRGBModel(path.stem, ts_model, device)
    except Exception:
        pass

    model = _load_resnet50_binary(path, device)
    logger.info("Loaded ResNet50 state_dict model: %s", path.name)
    return TorchRGBModel(path.stem, model, device)


def load_models(models_dir: str, device: torch.device) -> List:
    root = Path(models_dir)
    if not root.exists():
        raise FileNotFoundError(f"Models directory does not exist: {root}")

    files = sorted(root.glob("*.pt")) + sorted(root.glob("*.pth"))
    loaded = []
    failures = []
    for path in files:
        try:
            loaded.append(_try_load_model(path, device))
        except Exception as exc:
            failures.append((path.name, str(exc)))
            logger.error("Could not load model %s: %s", path.name, exc)

    if not loaded:
        details = "; ".join(f"{name}: {error}" for name, error in failures) or "no .pt/.pth files found"
        raise RuntimeError(f"No valid RGB models loaded from {root}. Details: {details}")

    return loaded
