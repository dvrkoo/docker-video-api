from __future__ import annotations

import numpy as np


class AlwaysRealModel:
    """Fallback model used until real RGB weights are provided."""

    def __init__(self, name: str = "always_real") -> None:
        self.name = name

    def predict(self, face_rgb: np.ndarray) -> float:
        _ = face_rgb
        return 0.0
