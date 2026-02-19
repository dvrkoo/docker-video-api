from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchvision
from torchvision import transforms

from .dummy import AlwaysRealModel

logger = logging.getLogger(__name__)


class TorchRGBModel:
    def __init__(self, name: str, model: torch.nn.Module, device: torch.device) -> None:
        self.name = name
        self.model = model
        self.device = device
        self.post = torch.nn.Softmax(dim=1)
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def predict(self, face_rgb: np.ndarray) -> float:
        tensor = self.preprocess(face_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = self.post(out)
        return float(probs[0, 1].item())


class TorchScriptRGBModel:
    def __init__(self, name: str, model: torch.jit.ScriptModule, device: torch.device) -> None:
        self.name = name
        self.model = model
        self.device = device
        self.post = torch.nn.Softmax(dim=1)
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def predict(self, face_rgb: np.ndarray) -> float:
        tensor = self.preprocess(face_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = self.post(out)
        if probs.ndim == 1:
            return float(probs[-1].item())
        return float(probs[0, -1].item())


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
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
    model.load_state_dict(state_dict, strict=False)
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
        logger.warning("Models directory does not exist: %s", root)
        return [AlwaysRealModel()]

    files = sorted(root.glob("*.pt")) + sorted(root.glob("*.pth"))
    loaded = []
    for path in files:
        try:
            loaded.append(_try_load_model(path, device))
        except Exception as exc:
            logger.error("Could not load model %s: %s", path.name, exc)

    if not loaded:
        logger.warning("No valid RGB models found, using AlwaysRealModel fallback")
        loaded = [AlwaysRealModel()]

    return loaded
