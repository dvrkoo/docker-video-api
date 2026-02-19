# Docker Video API (RGB Models)

Video deepfake detection module based on the `docker-api` baseline, adapted for **video-in / video-out**.

It watches an input folder, processes incoming videos, and writes:
- annotated output video (`*_processed.mp4`)
- report file (`*_report.txt`)

Only **RGB models** are used (no wavelet/frequency models).

## What the pipeline does

- Detects faces per frame and processes **only the largest detected face**.
- Marks frame as `FAKE` if **any enabled model** predicts fake above threshold.
- Draws red bbox for fake, green bbox for real.
- Computes percentages over **face frames only**:
  - `fake_face_frames / face_frames`
- Video verdict:
  - `VIDEO_FAKE_NOT_FALSE_POSITIVE` if fake ratio >= `VIDEO_FAKE_THRESHOLD` (default `0.40`)
  - otherwise `MOSTLY_REAL`

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

### CUDA (NVIDIA)

```bash
docker compose --profile cuda up video-detector-cuda
```

### MPS profile (Apple Silicon Docker)

```bash
docker compose --profile mps up video-detector-mps
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
- Drop video into `./input`
- Output files in `./output`:
  - `<video>_processed.mp4`
  - `<video>_report.txt`

## Environment variables

- `WATCH_FOLDER` (default `/data/input` in Docker, `./input` native)
- `OUTPUT_FOLDER` (default `/data/output` in Docker, `./output` native)
- `MODELS_FOLDER` (default `/data/models` in Docker, `./trained_models` native)
- `FORCE_CPU` (`true/false`)
- `AUTO_FALLBACK_CPU_ON_UNSUPPORTED_CUDA` (`true/false`, default `true`)
- `FRAME_FAKE_THRESHOLD` (default `0.5`)
- `VIDEO_FAKE_THRESHOLD` (default `0.4`)
- `INFERENCE_BATCH_SIZE` (default `32`)
- `DETECTOR_BACKEND` (`auto`, `retinaface`, `dlib`)
- `RETINAFACE_DET_SIZE` (default `640`)
- `RETINAFACE_BOX_SCALE` (default `1.25`)

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
