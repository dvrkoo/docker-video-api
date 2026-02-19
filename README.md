# Docker Video API (RGB Models)

This repository is a video-focused baseline module modeled after `docker-api`.

It watches an input folder, processes new videos, and writes:
- an annotated output video (`*_processed.mp4`)
- a text report (`*_report.txt`)

Only **RGB models** are used. Wavelet/frequency models are not part of this module.

## Detection logic

- The pipeline detects faces per frame and processes **only the biggest detected face**.
- If at least one enabled model predicts fake above threshold (default `0.5`), the frame is marked fake.
- Fake frames are drawn with a red box and label `FAKE`.
- Real frames are drawn with a green box and label `REAL`.
- Final percentages are computed over **frames with at least one detected face**.

Video-level rule:
- If `fake_face_frames / face_frames >= 0.40`, verdict is `VIDEO_FAKE_NOT_FALSE_POSITIVE`.
- Otherwise verdict is `MOSTLY_REAL`.

## Directory structure

```text
docker-video-api/
├── app.py
├── video_processor.py
├── models/
├── trained_models/
├── Dockerfile
├── Dockerfile.cuda
├── Dockerfile.mps
├── docker-compose.yml
└── requirements.txt
```

## Build and run

Create directories:

```bash
mkdir -p input output logs trained_models
```

CPU:

```bash
docker-compose up video-detector-cpu
```

CUDA:

```bash
docker-compose --profile cuda up video-detector-cuda
```

MPS (Docker on Apple Silicon may still run CPU-only in practice):

```bash
docker-compose --profile mps up video-detector-mps
```

## Inputs and outputs

Supported input extensions: `.mp4`, `.avi`, `.mov`, `.mkv`

Drop a video in `./input`:

```bash
cp /path/to/video.mp4 input/
```

Outputs in `./output`:
- `video_processed.mp4`
- `video_report.txt`

## Model loading

Place model files in `./trained_models` (mounted as `/data/models` in Docker).

Supported loading attempts:
1. TorchScript (`torch.jit.load`)
2. ResNet50 binary classifier state_dict (`2` classes, fake class index `1`)

If no valid model is present, fallback model `always_real` is used so the pipeline remains runnable.

## Environment variables

- `WATCH_FOLDER` (default `/data/input` in Docker, `./input` native)
- `OUTPUT_FOLDER` (default `/data/output` in Docker, `./output` native)
- `MODELS_FOLDER` (default `/data/models` in Docker, `./trained_models` native)
- `FORCE_CPU` (`true/false`)
- `FRAME_FAKE_THRESHOLD` (default `0.5`)
- `VIDEO_FAKE_THRESHOLD` (default `0.4`)
