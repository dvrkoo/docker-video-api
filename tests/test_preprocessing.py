"""Tests that GPU-path preprocessing produces results close to CPU/PIL preprocessing.

These tests run on CPU so no GPU is required in CI.
"""

import numpy as np
import torch

from video_processor import _preprocess_face_to_tensor, _preprocess_faces_gpu


def _random_face_crop(h: int = 167, w: int = 167) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_gpu_preprocess_close_to_cpu_preprocess():
    """GPU (bilinear + tensor ops) and CPU (PIL resize + transforms) should agree within atol=1e-3."""
    face_crop = _random_face_crop()
    device = torch.device("cpu")

    cpu_tensor = _preprocess_face_to_tensor(face_crop)  # (3, 224, 224)
    gpu_batch = _preprocess_faces_gpu([face_crop], device)  # (1, 3, 224, 224)

    assert gpu_batch.shape == (1, 3, 224, 224)
    # PIL LANCZOS and torch bilinear differ slightly in sub-pixel sampling;
    # allow up to 0.01 (out of a [-1, 1] range) which is ~1.25 pixel brightness units.
    assert torch.allclose(cpu_tensor, gpu_batch[0], atol=0.01), (
        f"Max abs diff: {(cpu_tensor - gpu_batch[0]).abs().max().item():.6f}"
    )


def test_gpu_preprocess_multiple_crops():
    """Batching N crops should produce the same result as processing them individually."""
    crops = [_random_face_crop(h=100 + i * 20, w=100 + i * 20) for i in range(4)]
    device = torch.device("cpu")

    batch = _preprocess_faces_gpu(crops, device)
    assert batch.shape == (4, 3, 224, 224)

    for i, crop in enumerate(crops):
        single = _preprocess_faces_gpu([crop], device)
        assert torch.allclose(batch[i], single[0], atol=1e-6), f"Mismatch at index {i}"


def test_gpu_preprocess_output_range():
    """After normalizing with mean=0.5 std=0.5, output should be in roughly [-1, 1]."""
    face_crop = _random_face_crop()
    device = torch.device("cpu")
    batch = _preprocess_faces_gpu([face_crop], device)
    assert batch.min() >= -1.1
    assert batch.max() <= 1.1
