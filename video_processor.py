from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import dlib
import torch

from models.registry import load_models

logger = logging.getLogger(__name__)


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
            "VIDEO_FAKE_NOT_FALSE_POSITIVE"
            if (fake_face_frames / face_frames) >= video_fake_threshold
            else "MOSTLY_REAL"
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


def _select_largest_face(faces):
    return max(faces, key=lambda face: face.width() * face.height())


def _clamp_face_bbox(face, frame_shape):
    h, w = frame_shape[:2]
    x1 = max(0, face.left())
    y1 = max(0, face.top())
    x2 = min(w, face.right())
    y2 = min(h, face.bottom())
    return x1, y1, x2, y2


def _write_report(
    report_path: Path,
    input_video: Path,
    output_video: Path,
    device: torch.device,
    frame_fake_threshold: float,
    video_fake_threshold: float,
    stats: VideoStats,
) -> None:
    lines: List[str] = [
        "Video Deepfake Report",
        f"timestamp={datetime.utcnow().isoformat()}Z",
        f"input_video={input_video.name}",
        f"output_video={output_video.name}",
        f"device={device}",
        f"frame_fake_threshold={frame_fake_threshold:.4f}",
        f"video_fake_threshold={video_fake_threshold:.4f}",
        f"total_frames={stats.total_frames}",
        f"face_frames={stats.face_frames}",
        f"fake_face_frames={stats.fake_face_frames}",
        f"real_face_frames={stats.real_face_frames}",
        f"fake_percent_over_face_frames={stats.fake_percent:.2f}",
        f"real_percent_over_face_frames={stats.real_percent:.2f}",
        f"verdict={stats.verdict}",
        "model_flagged_frames:",
    ]

    for name, count in sorted(stats.model_flagged_frames.items()):
        lines.append(f"  - {name}: {count}")

    if stats.face_frames == 0:
        lines.append("note=No faces were detected in this video")
    elif stats.fake_percent >= (video_fake_threshold * 100.0):
        lines.append(
            "note=Fake percentage is above threshold; result is considered not a false positive"
        )
    else:
        lines.append("note=Fake percentage is below threshold")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_video_file(
    input_path: str,
    output_folder: str,
    device: torch.device,
    models_dir: str,
    frame_fake_threshold: float,
    video_fake_threshold: float,
) -> Dict[str, str]:
    input_video = Path(input_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / f"{input_video.stem}_processed.mp4"
    report_path = output_dir / f"{input_video.stem}_report.txt"

    models = load_models(models_dir=models_dir, device=device)
    model_flagged_frames: Dict[str, int] = {m.name: 0 for m in models}
    detector = dlib.get_frontal_face_detector()

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

    logger.info("Processing video: %s", input_video.name)
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        total_frames += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector(frame_gray, 1)

        if faces:
            face_frames += 1
            face = _select_largest_face(faces)
            x1, y1, x2, y2 = _clamp_face_bbox(face, frame_bgr.shape)
            if x2 > x1 and y2 > y1:
                face_rgb = frame_rgb[y1:y2, x1:x2]
                frame_is_fake = False

                for model in models:
                    score = model.predict(face_rgb)
                    if score > frame_fake_threshold:
                        frame_is_fake = True
                        model_flagged_frames[model.name] += 1

                if frame_is_fake:
                    fake_face_frames += 1
                    label = "FAKE"
                    color = (0, 0, 255)
                else:
                    label = "REAL"
                    color = (0, 255, 0)

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
        device=device,
        frame_fake_threshold=frame_fake_threshold,
        video_fake_threshold=video_fake_threshold,
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
