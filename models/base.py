from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RGBModel(Protocol):
    name: str

    def predict(self, face_rgb: np.ndarray) -> float:
        """Return fake probability in [0, 1]."""


@dataclass
class ModelPrediction:
    model_name: str
    score: float
