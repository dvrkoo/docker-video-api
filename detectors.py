from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
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
    def __init__(self, upscale: int = 1, scale_factor: float = 1.0):
        import dlib

        self.detector = dlib.get_frontal_face_detector()
        self.upscale = upscale
        self.scale_factor = scale_factor

    def make_worker(self) -> "DlibDetector":
        return DlibDetector(upscale=self.upscale, scale_factor=self.scale_factor)

    def detect(self, frame_bgr) -> List[BBox]:
        if self.scale_factor < 1.0:
            small = cv2.resize(
                frame_bgr, (0, 0), fx=self.scale_factor, fy=self.scale_factor
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray, self.upscale)
        inv = 1.0 / self.scale_factor if self.scale_factor < 1.0 else 1.0
        out: List[BBox] = []
        for face in faces:
            out.append(
                (
                    int(face.left() * inv),
                    int(face.top() * inv),
                    int(face.right() * inv),
                    int(face.bottom() * inv),
                    1.0,
                )
            )
        return out


class MTCNNDetector:
    def __init__(self, device: torch.device, box_scale: float = 1.0):
        from facenet_pytorch import MTCNN

        self.box_scale = box_scale
        self.mtcnn = MTCNN(keep_all=True, device=device)
        logger.info("MTCNN initialized on device=%s, box_scale=%.2f", device, box_scale)

    def make_worker(self) -> "MTCNNDetector":
        return self  # GPU model; shared instance is safe

    def detect(self, frame_bgr) -> List[BBox]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(frame_rgb)
        out: List[BBox] = []
        if boxes is None:
            return out

        if probs is None:
            probs = [1.0] * len(boxes)

        for box, score in zip(boxes, probs):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1, x2, y2 = _expand_bbox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_shape=frame_bgr.shape,
                scale=self.box_scale,
            )
            score_val = 1.0 if score is None else float(score)
            out.append((x1, y1, x2, y2, score_val))
        return out


class RetinaFaceDetector:
    def __init__(
        self, device: torch.device, det_size: int = 640, box_scale: float = 1.25
    ):
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

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        logger.info(
            "RetinaFace initialized with providers: %s, allowed_modules=['detection'], det_size=%d",
            providers,
            det_size,
        )

    def make_worker(self) -> "RetinaFaceDetector":
        return self  # GPU model; shared instance is safe

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
    mtcnn_box_scale: float = 1.0,
    retinaface_det_size: int = 640,
    retinaface_box_scale: float = 1.25,
    dlib_scale_factor: float = 1.0,
):
    normalized = backend.lower().strip()
    wants_mtcnn = normalized in {"mtcnn", "auto"}
    wants_retina = normalized == "retinaface" or (
        normalized == "auto" and device.type in {"cuda", "mps"}
    )

    if wants_mtcnn:
        try:
            return MTCNNDetector(device=device, box_scale=mtcnn_box_scale)
        except Exception as exc:
            logger.warning(
                "MTCNN unavailable, trying RetinaFace/dlib fallback: %s", exc
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

    return DlibDetector(upscale=1, scale_factor=dlib_scale_factor)
