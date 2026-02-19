from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import dlib
import torch

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, float]


def _expand_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_shape,
    scale: float,
) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale

    nx1 = max(0, int(cx - bw / 2.0))
    ny1 = max(0, int(cy - bh / 2.0))
    nx2 = min(w, int(cx + bw / 2.0))
    ny2 = min(h, int(cy + bh / 2.0))
    return nx1, ny1, nx2, ny2


class DlibDetector:
    def __init__(self, upscale: int = 1):
        self.detector = dlib.get_frontal_face_detector()
        self.upscale = upscale

    def detect(self, frame_bgr) -> List[BBox]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, self.upscale)
        out: List[BBox] = []
        for face in faces:
            out.append((face.left(), face.top(), face.right(), face.bottom(), 1.0))
        return out


class RetinaFaceDetector:
    def __init__(self, device: torch.device, det_size: int = 640, box_scale: float = 1.25):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        self.box_scale = box_scale
        available = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda":
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        elif device.type == "mps":
            if "CoreMLExecutionProvider" in available:
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            ctx_id = -1
        else:
            ctx_id = -1

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        logger.info("RetinaFace initialized with providers: %s", providers)

    def detect(self, frame_bgr) -> List[BBox]:
        faces = self.app.get(frame_bgr)
        out: List[BBox] = []
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, y1, x2, y2 = _expand_bbox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_shape=frame_bgr.shape,
                scale=self.box_scale,
            )
            score = float(getattr(face, "det_score", 1.0))
            out.append((x1, y1, x2, y2, score))
        return out


def build_detector(
    device: torch.device,
    backend: str = "auto",
    retinaface_det_size: int = 640,
    retinaface_box_scale: float = 1.25,
):
    normalized = backend.lower().strip()
    wants_retina = normalized == "retinaface" or (
        normalized == "auto" and device.type in {"cuda", "mps"}
    )

    if wants_retina:
        try:
            return RetinaFaceDetector(
                device=device,
                det_size=retinaface_det_size,
                box_scale=retinaface_box_scale,
            )
        except Exception as exc:
            logger.warning("RetinaFace unavailable, falling back to dlib: %s", exc)

    return DlibDetector(upscale=1)
