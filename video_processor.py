from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.registry import load_models

logger = logging.getLogger(__name__)

_tls = threading.local()


def _worker_init(detector_factory):
    _tls.detector = detector_factory()


def _worker_detect(frame_bgr):
    return _tls.detector.detect(frame_bgr)


@dataclass
class VideoStats:
    total_frames: int
    face_frames: int
    fake_face_frames: int
    real_face_frames: int
    fake_percent: float
    real_percent: float
    verdict: str
    model_flagged_frames: Dict[str, int]


def summarize_counts(
    total_frames: int,
    face_frames: int,
    fake_face_frames: int,
    video_fake_threshold: float,
) -> VideoStats:
    real_face_frames = max(0, face_frames - fake_face_frames)

    if face_frames == 0:
        fake_percent = 0.0
        real_percent = 0.0
        verdict = "NO_FACES_DETECTED"
    else:
        fake_percent = 100.0 * (fake_face_frames / face_frames)
        real_percent = 100.0 - fake_percent
        verdict = (
            "FAKE"
            if (fake_face_frames / face_frames) >= video_fake_threshold
            else "REAL"
        )

    return VideoStats(
        total_frames=total_frames,
        face_frames=face_frames,
        fake_face_frames=fake_face_frames,
        real_face_frames=real_face_frames,
        fake_percent=fake_percent,
        real_percent=real_percent,
        verdict=verdict,
        model_flagged_frames={},
    )


_PLAYER_RESNET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


def _select_primary_face(faces):
    return faces[0]


def _player_style_bbox(face, frame_shape, scale: float = 1.3):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = face

    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)

    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, x1 + size_bb, y1 + size_bb


def _preprocess_face_to_tensor(face_rgb):
    return _PLAYER_RESNET_TRANSFORM(Image.fromarray(face_rgb))


def _preprocess_faces_gpu(face_crops: List, device: torch.device) -> torch.Tensor:
    """Preprocess a list of HWC uint8 numpy face crops entirely on GPU.

    Returns a (N, 3, 224, 224) float32 tensor already on *device*.
    """
    tensors = []
    for crop in face_crops:
        t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float().to(device)
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        t = t.div_(255.0).sub_(0.5).div_(0.5)
        tensors.append(t)
    return torch.cat(tensors, dim=0)


def _infer_fake_flags(
    face_batch_cpu: torch.Tensor,
    models: List,
    device: torch.device,
    frame_fake_threshold: float,
    model_flagged_frames: Dict[str, int],
    batch_already_on_device: bool = False,
) -> List[bool]:
    if face_batch_cpu.shape[0] == 0:
        return []

    batch = face_batch_cpu if batch_already_on_device else face_batch_cpu.to(device)
    frame_flags = torch.zeros(batch.shape[0], dtype=torch.bool, device=batch.device)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode(), amp_ctx:
        for model in models:
            scores = model.predict_batch(batch)
            scores = scores.float()
            model_flags = scores > frame_fake_threshold
            frame_flags |= model_flags
            model_flagged_frames[model.name] += int(model_flags.sum().item())

    return frame_flags.cpu().tolist()


def _read_frame_chunk(cap, n: int) -> List:
    """Read up to *n* frames from *cap*. Returns fewer than *n* at end of stream."""
    chunk = []
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        chunk.append(frame)
    return chunk


def _write_report(
    report_path: Path,
    input_video: Path,
    output_video: Path,
    stats: VideoStats,
) -> None:
    if stats.verdict == "FAKE":
        confidence = f"{stats.fake_percent:.2f}%"
    elif stats.verdict == "REAL":
        confidence = f"{stats.real_percent:.2f}%"
    else:
        confidence = "0.00%"

    lines: List[str] = [
        "Video Deepfake Report",
        f"timestamp={datetime.utcnow().isoformat()}Z",
        f"input_video={input_video.name}",
        f"output_video={output_video.name}",
        f"total_frames={stats.total_frames}",
        f"face_frames_considered={stats.face_frames}",
        f"fake_frames={stats.fake_face_frames}",
        f"result_confidence={confidence}",
        f"result={stats.verdict}",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_video_file(
    input_path: str,
    output_folder: str,
    device: torch.device,
    models_dir: str,
    frame_fake_threshold: float,
    video_fake_threshold: float,
    models: Optional[List] = None,
    detector=None,
    inference_batch_size: int = 32,
    gpu_preprocess: bool = False,
    detection_num_workers: int = 1,
) -> Dict[str, str]:
    input_video = Path(input_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"{input_video.stem}_processed.mp4"
    report_path = output_dir / f"{input_video.stem}_report.txt"

    loaded_models = (
        models
        if models is not None
        else load_models(models_dir=models_dir, device=device)
    )
    if not loaded_models:
        raise RuntimeError(
            f"No models available to process {input_path}. "
            "Place .pt/.pth files in the models directory."
        )
    model_flagged_frames: Dict[str, int] = {m.name: 0 for m in loaded_models}
    if detector is None:
        from detectors import build_detector

        face_detector = build_detector(device=device, backend="auto")
    else:
        face_detector = detector

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    total_frames = 0
    face_frames = 0
    fake_face_frames = 0

    pending_frames: List[dict] = []
    # CPU path: list of torch.Tensor; GPU path: list of np.ndarray face crops
    pending_face_data: List = []

    def flush_pending() -> None:
        nonlocal fake_face_frames
        if not pending_frames:
            return

        face_flags: List[bool] = []
        if pending_face_data:
            if gpu_preprocess:
                batch_gpu = _preprocess_faces_gpu(pending_face_data, device)
                face_flags = _infer_fake_flags(
                    face_batch_cpu=batch_gpu,
                    models=loaded_models,
                    device=device,
                    frame_fake_threshold=frame_fake_threshold,
                    model_flagged_frames=model_flagged_frames,
                    batch_already_on_device=True,
                )
            else:
                batch_cpu = torch.stack(pending_face_data, dim=0)
                face_flags = _infer_fake_flags(
                    face_batch_cpu=batch_cpu,
                    models=loaded_models,
                    device=device,
                    frame_fake_threshold=frame_fake_threshold,
                    model_flagged_frames=model_flagged_frames,
                )

        for item in pending_frames:
            frame_bgr = item["frame"]
            if item["face_idx"] is not None:
                is_fake = face_flags[item["face_idx"]]
                if is_fake:
                    fake_face_frames += 1
                    label = "FAKE"
                    color = (0, 0, 255)
                else:
                    label = "REAL"
                    color = (0, 255, 0)

                x1, y1, x2, y2 = item["bbox"]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            writer.write(frame_bgr)

        pending_frames.clear()
        pending_face_data.clear()

    logger.info(
        "Processing video: %s (detection_workers=%d)",
        input_video.name,
        detection_num_workers,
    )
    with ThreadPoolExecutor(
        max_workers=detection_num_workers,
        initializer=_worker_init,
        initargs=(face_detector.make_worker,),
    ) as executor:
        while True:
            chunk = _read_frame_chunk(cap, detection_num_workers)
            if not chunk:
                break

            face_results = list(executor.map(_worker_detect, chunk))

            for frame_bgr, faces in zip(chunk, face_results):
                total_frames += 1

                entry = {"frame": frame_bgr, "bbox": None, "face_idx": None}
                if faces:
                    face_frames += 1
                    face = _select_primary_face(faces)
                    x1, y1, x2, y2, _ = face
                    x1, y1, x2, y2 = _player_style_bbox(
                        (x1, y1, x2, y2), frame_bgr.shape
                    )
                    if x2 > x1 and y2 > y1:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        face_rgb = frame_rgb[y1:y2, x1:x2]
                        entry["bbox"] = (x1, y1, x2, y2)
                        entry["face_idx"] = len(pending_face_data)
                        if gpu_preprocess:
                            pending_face_data.append(face_rgb)
                        else:
                            pending_face_data.append(
                                _preprocess_face_to_tensor(face_rgb)
                            )

                pending_frames.append(entry)
                if len(pending_frames) >= max(1, inference_batch_size):
                    flush_pending()

    flush_pending()

    cap.release()
    writer.release()

    stats = summarize_counts(
        total_frames=total_frames,
        face_frames=face_frames,
        fake_face_frames=fake_face_frames,
        video_fake_threshold=video_fake_threshold,
    )
    stats.model_flagged_frames = model_flagged_frames

    _write_report(
        report_path=report_path,
        input_video=input_video,
        output_video=output_video,
        stats=stats,
    )

    logger.info(
        "Finished %s - face_frames=%d fake=%d (%.2f%%)",
        input_video.name,
        stats.face_frames,
        stats.fake_face_frames,
        stats.fake_percent,
    )

    return {
        "output_video": str(output_video),
        "report_file": str(report_path),
        "verdict": stats.verdict,
    }
