from __future__ import annotations

import logging
import json
import queue
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


@dataclass
class FramePrediction:
    frame_index: int
    timestamp_ms: float
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: Optional[float]
    is_fake: bool


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
) -> Tuple[List[bool], List[float]]:
    if face_batch_cpu.shape[0] == 0:
        return [], []

    batch = face_batch_cpu if batch_already_on_device else face_batch_cpu.to(device)
    max_scores = torch.zeros(batch.shape[0], dtype=torch.float32, device=batch.device)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode(), amp_ctx:
        for model in models:
            scores = model.predict_batch(batch)
            scores = scores.float()
            max_scores = torch.maximum(max_scores, scores)
            model_flags = scores > frame_fake_threshold
            model_flagged_frames[model.name] += int(model_flags.sum().item())

    frame_flags = max_scores > frame_fake_threshold
    return frame_flags.cpu().tolist(), max_scores.cpu().tolist()


_FRAME_SENTINEL = object()


def _frame_reader_thread(cap, q: queue.Queue) -> None:
    """Background thread: decode frames from *cap* and push them onto *q*.

    Sends ``_FRAME_SENTINEL`` as the last item to signal end-of-stream.
    Runs as a daemon so it is silently killed if the main thread exits early.
    """
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            q.put(frame)
    finally:
        q.put(_FRAME_SENTINEL)


def _overlay_style(frame_shape) -> Tuple[int, float, int, int]:
    min_dim = min(frame_shape[:2])
    box_thickness = max(2, min(6, int(round(min_dim / 360.0))))
    font_scale = max(0.5, min(1.4, min_dim / 900.0))
    text_thickness = max(1, min(4, int(round(min_dim / 480.0))))
    text_y_offset = max(8, min(36, int(round(min_dim / 64.0))))
    return box_thickness, font_scale, text_thickness, text_y_offset


def _normalized_bbox(
    bbox: Optional[Tuple[int, int, int, int]], width: int, height: int
) -> Optional[Dict[str, float]]:
    if bbox is None or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = bbox
    return {
        "x1": round(x1 / width, 6),
        "y1": round(y1 / height, 6),
        "x2": round(x2 / width, 6),
        "y2": round(y2 / height, 6),
    }


def _encode_output_video(
    source_path: Path,
    output_path: Path,
    codec: str,
    crf: int,
    preset: str,
    pixel_format: str,
) -> Path:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        pixel_format,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        source_path.unlink(missing_ok=True)
        return output_path
    except Exception as exc:
        logger.warning(
            "ffmpeg re-encode failed (%s); using raw writer output instead", exc
        )
        if output_path.exists():
            output_path.unlink()
        shutil.move(str(source_path), str(output_path))
        return output_path


def _write_report_json(
    report_path: Path,
    input_video: Path,
    output_video: Optional[Path],
    stats: VideoStats,
    frame_predictions: List[FramePrediction],
    width: int,
    height: int,
    fps: float,
    frame_fake_threshold: float,
    video_fake_threshold: float,
) -> None:
    if stats.verdict == "FAKE":
        verdict_confidence = stats.fake_percent / 100.0
    elif stats.verdict == "REAL":
        verdict_confidence = stats.real_percent / 100.0
    else:
        verdict_confidence = 0.0

    payload = {
        "schema_version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_video": input_video.name,
        "output_video": output_video.name if output_video is not None else None,
        "video": {
            "width": width,
            "height": height,
            "fps": round(fps, 6),
            "total_frames": stats.total_frames,
        },
        "thresholds": {
            "frame_fake_threshold": frame_fake_threshold,
            "video_fake_threshold": video_fake_threshold,
        },
        "summary": {
            "face_frames_considered": stats.face_frames,
            "fake_frames": stats.fake_face_frames,
            "real_frames": stats.real_face_frames,
            "fake_percent": round(stats.fake_percent, 6),
            "real_percent": round(stats.real_percent, 6),
            "result": stats.verdict,
            "result_confidence": round(verdict_confidence, 6),
        },
        "frames": [
            {
                "frame_index": item.frame_index,
                "timestamp_ms": item.timestamp_ms,
                "bbox_px": (
                    {
                        "x1": item.bbox[0],
                        "y1": item.bbox[1],
                        "x2": item.bbox[2],
                        "y2": item.bbox[3],
                    }
                    if item.bbox is not None
                    else None
                ),
                "bbox_norm": _normalized_bbox(item.bbox, width, height),
                "confidence": item.confidence,
                "is_fake": item.is_fake,
            }
            for item in frame_predictions
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    output_codec: str = "libx264",
    output_crf: int = 23,
    output_preset: str = "medium",
    output_pixel_format: str = "yuv420p",
    output_video_enabled: bool = False,
) -> Dict[str, Optional[str]]:
    input_video = Path(input_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"{input_video.stem}_processed.mp4"
    raw_video = output_dir / f"{input_video.stem}_processed_raw.mp4"
    final_output_video: Optional[Path] = None
    report_path = output_dir / f"{input_video.stem}_report.json"

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
    writer = None
    if output_video_enabled:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_video), fourcc, fps, (width, height))

    total_frames = 0
    face_frames = 0
    fake_face_frames = 0

    pending_frames: List[dict] = []
    # CPU path: list of torch.Tensor; GPU path: list of np.ndarray face crops
    pending_face_data: List = []
    frame_predictions: List[FramePrediction] = []
    box_thickness, font_scale, text_thickness, text_y_offset = _overlay_style(
        (height, width, 3)
    )

    def flush_pending() -> None:
        nonlocal fake_face_frames
        if not pending_frames:
            return

        face_flags: List[bool] = []
        face_confidences: List[float] = []
        if pending_face_data:
            if gpu_preprocess:
                batch_gpu = _preprocess_faces_gpu(pending_face_data, device)
                face_flags, face_confidences = _infer_fake_flags(
                    face_batch_cpu=batch_gpu,
                    models=loaded_models,
                    device=device,
                    frame_fake_threshold=frame_fake_threshold,
                    model_flagged_frames=model_flagged_frames,
                    batch_already_on_device=True,
                )
            else:
                batch_cpu = torch.stack(pending_face_data, dim=0)
                face_flags, face_confidences = _infer_fake_flags(
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
                confidence = face_confidences[item["face_idx"]]
                if is_fake:
                    fake_face_frames += 1
                    label = "FAKE"
                    color = (0, 0, 255)
                else:
                    label = "REAL"
                    color = (0, 255, 0)

                x1, y1, x2, y2 = item["bbox"]
                if output_video_enabled:
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, box_thickness)
                    cv2.putText(
                        frame_bgr,
                        f"{label} {confidence:.2f}",
                        (x1, max(0, y1 - text_y_offset)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        text_thickness,
                    )
                frame_predictions.append(
                    FramePrediction(
                        frame_index=item["frame_index"],
                        timestamp_ms=item["timestamp_ms"],
                        bbox=(x1, y1, x2, y2),
                        confidence=round(float(confidence), 6),
                        is_fake=bool(is_fake),
                    )
                )
            else:
                frame_predictions.append(
                    FramePrediction(
                        frame_index=item["frame_index"],
                        timestamp_ms=item["timestamp_ms"],
                        bbox=None,
                        confidence=None,
                        is_fake=False,
                    )
                )

            if writer is not None:
                writer.write(frame_bgr)

        pending_frames.clear()
        pending_face_data.clear()

    logger.info(
        "Processing video: %s (detection_workers=%d)",
        input_video.name,
        detection_num_workers,
    )

    # Queue depth: keep enough frames buffered so workers never starve.
    frame_queue: queue.Queue = queue.Queue(maxsize=detection_num_workers * 8)
    reader = threading.Thread(
        target=_frame_reader_thread, args=(cap, frame_queue), daemon=True
    )
    reader.start()

    chunk_size = detection_num_workers * 4

    with ThreadPoolExecutor(
        max_workers=detection_num_workers,
        initializer=_worker_init,
        initargs=(face_detector.make_worker,),
    ) as executor:
        done = False
        while not done:
            chunk = []
            for _ in range(chunk_size):
                item = frame_queue.get()
                if item is _FRAME_SENTINEL:
                    done = True
                    break
                chunk.append(item)

            if not chunk:
                break

            face_results = list(executor.map(_worker_detect, chunk))

            for frame_bgr, faces in zip(chunk, face_results):
                frame_index = total_frames
                total_frames += 1

                entry = {
                    "frame": frame_bgr,
                    "bbox": None,
                    "face_idx": None,
                    "frame_index": frame_index,
                    "timestamp_ms": round((frame_index / fps) * 1000.0, 3),
                }
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
    if writer is not None:
        writer.release()
        final_output_video = _encode_output_video(
            source_path=raw_video,
            output_path=output_video,
            codec=output_codec,
            crf=output_crf,
            preset=output_preset,
            pixel_format=output_pixel_format,
        )

    stats = summarize_counts(
        total_frames=total_frames,
        face_frames=face_frames,
        fake_face_frames=fake_face_frames,
        video_fake_threshold=video_fake_threshold,
    )
    stats.model_flagged_frames = model_flagged_frames

    _write_report_json(
        report_path=report_path,
        input_video=input_video,
        output_video=final_output_video,
        stats=stats,
        frame_predictions=frame_predictions,
        width=width,
        height=height,
        fps=fps,
        frame_fake_threshold=frame_fake_threshold,
        video_fake_threshold=video_fake_threshold,
    )

    logger.info(
        "Finished %s - face_frames=%d fake=%d (%.2f%%)",
        input_video.name,
        stats.face_frames,
        stats.fake_face_frames,
        stats.fake_percent,
    )

    return {
        "output_video": str(final_output_video)
        if final_output_video is not None
        else None,
        "report_file": str(report_path),
        "verdict": stats.verdict,
    }
