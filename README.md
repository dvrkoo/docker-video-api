# Docker Video API (RGB Models)

Video deepfake detection module based on the `docker-api` baseline, adapted for **video-in / video-out**.

It watches an input folder, processes incoming videos, and writes:
- JSON report (`*_report.json`)
- optional annotated output video (`*_processed.mp4`) when enabled

Only **RGB models** are used (no wavelet/frequency models).

## What the pipeline does

- Detects faces per frame and processes the **first detected face** (player-compatible).
- Face crop follows player-style square policy with scale `1.3`.
- Marks frame as `FAKE` if **any enabled model** predicts fake above threshold.
- Draws red bbox for fake, green bbox for real.
- Computes percentages over **face frames only**:
  - `fake_face_frames / face_frames`
- Video verdict:
  - `FAKE` if fake ratio >= `VIDEO_FAKE_THRESHOLD` (default `0.40`)
  - `REAL` otherwise
  - `NO_FACES_DETECTED` if no faces were found in any frame

## Core processing pipeline

`video_processor.py` is the main processing unit. It runs one video end-to-end and returns a result dict.

### Parallel face detection

Frames are read in chunks of `DLIB_NUM_WORKERS` at a time. A `ThreadPoolExecutor` with that many worker threads runs face detection in parallel across the chunk:

```
chunk (N frames) → ThreadPoolExecutor (N workers) → face_results (N lists of bboxes)
```

Each worker thread owns a **private detector instance** created at pool startup via the `initializer` parameter. This is required for dlib: `dlib.get_frontal_face_detector()` is not thread-safe — calling `.detect()` on the same instance from multiple threads simultaneously causes a segfault in dlib's internal C++ state. Thread-local instances eliminate the race condition without any locking overhead.

- `DlibDetector.make_worker()` — constructs a new independent instance per thread
- `MTCNNDetector.make_worker()` / `RetinaFaceDetector.make_worker()` — return `self` (GPU models are safe to share)

### Inference batching

After detection, face crops accumulate in a pending buffer. When the buffer reaches `INFERENCE_BATCH_SIZE`, all crops are batched into a single tensor and run through every loaded model in one forward pass. This keeps GPU utilisation high and avoids per-frame inference overhead.

### GPU preprocessing

When `GPU_PREPROCESS=true`, raw face crop arrays (HWC uint8) are sent to the GPU and resized/normalised there with `torch.nn.functional.interpolate`, skipping PIL and CPU torchvision transforms entirely.

### Output per video

- `<name>_report.json` — summary + per-frame predictions (`bbox_px`, `bbox_norm`, `confidence`, `is_fake`)
- `<name>_processed.mp4` — optional annotated video with bbox + confidence overlay

## Get model weights

Model files are **not** committed in this repository.

Download them from:
- https://drive.google.com/drive/folders/1AB1vPWF9dEv4l-aeKp8tbjXHumzJ0bLO?usp=sharing

Place all `.pt`/`.pth` files into:
- `./trained_models` (mounted in container as `/data/models`)

## Prebuilt container images (GHCR)

Images are published by GitHub Actions on pushes to `main`.

Registry path:
- `ghcr.io/dvrkoo/docker-video-api/video-deepfake-detector`

Main tags:
- CPU: `latest`
- CUDA: `latest-cuda`
- Apple Silicon profile image: `latest-mps`

Example pulls:

```bash
docker pull ghcr.io/dvrkoo/docker-video-api/video-deepfake-detector:latest
docker pull ghcr.io/dvrkoo/docker-video-api/video-deepfake-detector:latest-cuda
docker pull ghcr.io/dvrkoo/docker-video-api/video-deepfake-detector:latest-mps
```

## Run with docker-compose

Create folders:

```bash
mkdir -p input output logs trained_models
```

### CPU

```bash
docker compose up video-detector-cpu
```

Default behavior is JSON-only output (`OUTPUT_VIDEO_ENABLED=false`).

### CUDA (NVIDIA)

```bash
docker compose --profile cuda up video-detector-cuda
```

Default behavior is JSON-only output (`OUTPUT_VIDEO_ENABLED=false`).

### MPS profile (Apple Silicon Docker)

```bash
docker compose --profile mps up video-detector-mps
```

Default behavior is JSON-only output (`OUTPUT_VIDEO_ENABLED=false`).

### Debug profile (video + JSON, live code edits)

```bash
docker compose --profile debug up --force-recreate video-detector-debug
```

`env.debug` sets `OUTPUT_VIDEO_ENABLED=true`, so debug writes both JSON and annotated video.

### Toggle video output per run

JSON-only (default in cpu/cuda/mps):

```bash
OUTPUT_VIDEO_ENABLED=false docker compose --profile cuda up -d video-detector-cuda
```

JSON + annotated video:

```bash
OUTPUT_VIDEO_ENABLED=true docker compose --profile cuda up -d video-detector-cuda
```

Notes:
- `video-detector-mps` is built as `linux/arm64` for M1/M2/M3 machines.
- Inside Docker on macOS, runtime may still use CPU depending on backend/provider availability.

## Run prebuilt image directly

### CPU

```bash
docker run -d \
  --name video-detector-cpu \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -v $(pwd)/logs:/data/logs \
  -v $(pwd)/trained_models:/data/models \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  -e MODELS_FOLDER=/data/models \
  -e FORCE_CPU=true \
  ghcr.io/dvrkoo/docker-video-api/video-deepfake-detector:latest
```

### CUDA

```bash
docker run -d \
  --name video-detector-cuda \
  --gpus all \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -v $(pwd)/logs:/data/logs \
  -v $(pwd)/trained_models:/data/models \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  -e MODELS_FOLDER=/data/models \
  ghcr.io/dvrkoo/docker-video-api/video-deepfake-detector:latest-cuda
```

### MPS profile image

```bash
docker run -d \
  --name video-detector-mps \
  --platform linux/arm64 \
  -v $(pwd)/input:/data/input \
  -v $(pwd)/output:/data/output \
  -v $(pwd)/logs:/data/logs \
  -v $(pwd)/trained_models:/data/models \
  -e WATCH_FOLDER=/data/input \
  -e OUTPUT_FOLDER=/data/output \
  -e MODELS_FOLDER=/data/models \
  ghcr.io/dvrkoo/docker-video-api/video-deepfake-detector:latest-mps
```

## Input/output behavior

- Supported input: `.mp4`, `.avi`, `.mov`, `.mkv`
- Default detector is dlib (player-compatible behavior).
- RetinaFace and MTCNN remain available as optional override backends.
- Drop video into `./input`
- Output files in `./output`:
  - `<video>_report.json` (always)
  - `<video>_processed.mp4` (only when `OUTPUT_VIDEO_ENABLED=true`)

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `WATCH_FOLDER` | `/data/input` | Folder monitored for incoming videos |
| `OUTPUT_FOLDER` | `/data/output` | Where processed videos and reports are written |
| `MODELS_FOLDER` | `/data/models` | Path to `.pt`/`.pth` model weight files |
| `FORCE_CPU` | `false` | Force CPU even when GPU is available |
| `AUTO_FALLBACK_CPU_ON_UNSUPPORTED_CUDA` | `true` | Fall back to CPU if GPU arch is not in PyTorch build |
| `FRAME_FAKE_THRESHOLD` | `0.5` | Per-frame model score above which a frame is flagged fake |
| `VIDEO_FAKE_THRESHOLD` | `0.4` | Ratio of fake face frames above which the video is `FAKE` |
| `INFERENCE_BATCH_SIZE` | `32` | Number of face crops batched per inference pass |
| `GPU_PREPROCESS` | `false` | Resize and normalise face crops on GPU instead of CPU |
| `OUTPUT_VIDEO_ENABLED` | `false` | Write annotated output video in addition to JSON report |
| `OUTPUT_CODEC` | `libx264` | ffmpeg codec for annotated video encoding |
| `OUTPUT_CRF` | `23` | ffmpeg CRF quality target (lower = higher quality, larger file) |
| `OUTPUT_PRESET` | `medium` | ffmpeg encoding speed/compression preset |
| `OUTPUT_PIXEL_FORMAT` | `yuv420p` | ffmpeg output pixel format |
| `DETECTOR_BACKEND` | `dlib` | Face detector: `dlib`, `mtcnn`, `retinaface`, `auto` |
| `DLIB_NUM_WORKERS` | `1` | Worker threads for parallel dlib face detection |
| `DLIB_SCALE_FACTOR` | `1.0` | Pre-detection resize factor for dlib (1.0 = full resolution) |
| `MTCNN_BOX_SCALE` | `1.0` | Bbox expansion multiplier for MTCNN |
| `RETINAFACE_DET_SIZE` | `640` | Input resolution for RetinaFace detector |
| `RETINAFACE_BOX_SCALE` | `1.25` | Bbox expansion multiplier for RetinaFace |

## Faster rebuilds

BuildKit cache mounts are already enabled in Dockerfiles.

Use BuildKit locally:

```bash
DOCKER_BUILDKIT=1 docker compose build
```

## CI/CD

On each push/PR:
- unit tests
- CPU docker build + smoke test

On pushes to `main`:
- publish GHCR images for CPU, CUDA, and MPS profile.
