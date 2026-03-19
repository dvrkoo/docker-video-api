import json

import pytest
import torch

from video_processor import (
    FramePrediction,
    VideoStats,
    _infer_fake_flags,
    _write_report_json,
    summarize_counts,
)


def test_summarize_counts_fake_video_over_threshold():
    stats = summarize_counts(
        total_frames=100,
        face_frames=50,
        fake_face_frames=25,
        video_fake_threshold=0.4,
    )

    assert stats.fake_percent == 50.0
    assert stats.real_percent == 50.0
    assert stats.verdict == "FAKE"


def test_summarize_counts_mostly_real_below_threshold():
    stats = summarize_counts(
        total_frames=100,
        face_frames=50,
        fake_face_frames=10,
        video_fake_threshold=0.4,
    )

    assert stats.fake_percent == 20.0
    assert stats.real_percent == 80.0
    assert stats.verdict == "REAL"


def test_summarize_counts_no_faces():
    stats = summarize_counts(
        total_frames=120,
        face_frames=0,
        fake_face_frames=0,
        video_fake_threshold=0.4,
    )

    assert stats.fake_percent == 0.0
    assert stats.real_percent == 0.0
    assert stats.verdict == "NO_FACES_DETECTED"


class _DummyModel:
    def __init__(self, name, scores):
        self.name = name
        self._scores = torch.tensor(scores, dtype=torch.float32)

    def predict_batch(self, batch_tensor):
        return self._scores[: batch_tensor.shape[0]]


def test_infer_fake_flags_returns_any_model_flag_and_max_confidence():
    batch = torch.zeros((3, 3, 224, 224), dtype=torch.float32)
    models = [
        _DummyModel("m1", [0.1, 0.9, 0.6]),
        _DummyModel("m2", [0.7, 0.2, 0.4]),
    ]
    flagged_counts = {"m1": 0, "m2": 0}

    flags, confidences = _infer_fake_flags(
        face_batch_cpu=batch,
        models=models,
        device=torch.device("cpu"),
        frame_fake_threshold=0.5,
        model_flagged_frames=flagged_counts,
    )

    assert flags == [True, True, True]
    assert confidences == pytest.approx([0.7, 0.9, 0.6], abs=1e-6)
    assert flagged_counts == {"m1": 2, "m2": 1}


def test_write_report_json_with_per_frame_records(tmp_path):
    report_path = tmp_path / "sample_report.json"
    stats = VideoStats(
        total_frames=2,
        face_frames=1,
        fake_face_frames=1,
        real_face_frames=0,
        fake_percent=100.0,
        real_percent=0.0,
        verdict="FAKE",
        model_flagged_frames={},
    )
    frames = [
        FramePrediction(
            frame_index=0,
            timestamp_ms=0.0,
            bbox=(10, 20, 30, 40),
            confidence=0.83,
            is_fake=True,
        ),
        FramePrediction(
            frame_index=1,
            timestamp_ms=33.333,
            bbox=None,
            confidence=None,
            is_fake=False,
        ),
    ]

    _write_report_json(
        report_path=report_path,
        input_video=tmp_path / "input.mp4",
        output_video=tmp_path / "output.mp4",
        stats=stats,
        frame_predictions=frames,
        width=100,
        height=50,
        fps=30.0,
        frame_fake_threshold=0.5,
        video_fake_threshold=0.4,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["result"] == "FAKE"
    assert payload["summary"]["result_confidence"] == 1.0
    assert payload["video"] == {
        "width": 100,
        "height": 50,
        "fps": 30.0,
        "total_frames": 2,
    }
    assert payload["thresholds"] == {
        "frame_fake_threshold": 0.5,
        "video_fake_threshold": 0.4,
    }
    assert payload["frames"][0]["bbox_px"] == {
        "x1": 10,
        "y1": 20,
        "x2": 30,
        "y2": 40,
    }
    assert payload["frames"][0]["bbox_norm"] == {
        "x1": 0.1,
        "y1": 0.4,
        "x2": 0.3,
        "y2": 0.8,
    }
    assert payload["frames"][0]["confidence"] == 0.83
    assert payload["frames"][1]["bbox_px"] is None
    assert payload["frames"][1]["bbox_norm"] is None
    assert payload["frames"][1]["confidence"] is None


def test_write_report_json_allows_missing_output_video(tmp_path):
    report_path = tmp_path / "sample_report_no_video.json"
    stats = VideoStats(
        total_frames=1,
        face_frames=0,
        fake_face_frames=0,
        real_face_frames=0,
        fake_percent=0.0,
        real_percent=0.0,
        verdict="NO_FACES_DETECTED",
        model_flagged_frames={},
    )

    _write_report_json(
        report_path=report_path,
        input_video=tmp_path / "input.mp4",
        output_video=None,
        stats=stats,
        frame_predictions=[
            FramePrediction(
                frame_index=0,
                timestamp_ms=0.0,
                bbox=None,
                confidence=None,
                is_fake=False,
            )
        ],
        width=1920,
        height=1080,
        fps=30.0,
        frame_fake_threshold=0.5,
        video_fake_threshold=0.4,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["output_video"] is None
