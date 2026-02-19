from __future__ import annotations

import torch


class AlwaysRealModel:
    """Fallback model used until real RGB weights are provided."""

    def __init__(self, name: str = "always_real") -> None:
        self.name = name

    def predict_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch_tensor.shape[0], device=batch_tensor.device)
