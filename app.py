import logging
import os
import queue
import threading
import time
from pathlib import Path

import torch
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import logger_config
from video_processor import process_video_file

logger = logging.getLogger(__name__)

WATCH_FOLDER = os.getenv("WATCH_FOLDER", "./input" if not os.path.exists("/data") else "/data/input")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./output" if not os.path.exists("/data") else "/data/output")
MODELS_FOLDER = os.getenv("MODELS_FOLDER", "./trained_models")
FRAME_FAKE_THRESHOLD = float(os.getenv("FRAME_FAKE_THRESHOLD", "0.5"))
VIDEO_FAKE_THRESHOLD = float(os.getenv("VIDEO_FAKE_THRESHOLD", "0.4"))
FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

if FORCE_CPU:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

logger.info("Starting video deepfake docker module")
logger.info("Watch folder: %s", WATCH_FOLDER)
logger.info("Output folder: %s", OUTPUT_FOLDER)
logger.info("Models folder: %s", MODELS_FOLDER)
logger.info("Using device: %s", DEVICE)
logger.info("Frame fake threshold: %.2f", FRAME_FAKE_THRESHOLD)
logger.info("Video fake threshold: %.2f", VIDEO_FAKE_THRESHOLD)

file_queue: queue.Queue[str] = queue.Queue()


def _is_supported_video(path: str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def _wait_for_file_stable(path: str, checks: int = 3, delay: float = 1.0) -> bool:
    stable_count = 0
    previous_size = -1

    for _ in range(30):
        if not os.path.exists(path):
            time.sleep(delay)
            continue

        current_size = os.path.getsize(path)
        if current_size > 0 and current_size == previous_size:
            stable_count += 1
            if stable_count >= checks:
                return True
        else:
            stable_count = 0
            previous_size = current_size

        time.sleep(delay)

    return False


def _cleanup_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def process_file(file_path: str) -> None:
    if not _is_supported_video(file_path):
        logger.info("Skipping non-video file: %s", file_path)
        return

    if not _wait_for_file_stable(file_path):
        logger.error("File did not stabilize in time: %s", file_path)
        return

    logger.info("Processing file: %s", file_path)
    result = process_video_file(
        input_path=file_path,
        output_folder=OUTPUT_FOLDER,
        device=DEVICE,
        models_dir=MODELS_FOLDER,
        frame_fake_threshold=FRAME_FAKE_THRESHOLD,
        video_fake_threshold=VIDEO_FAKE_THRESHOLD,
    )
    logger.info("Processing completed: %s", result)
    _cleanup_accelerator_cache()


class FileCreatedHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        if _is_supported_video(path):
            logger.info("Queued new file: %s", path)
            file_queue.put(path)

    def on_moved(self, event):
        if event.is_directory:
            return
        path = event.dest_path
        if _is_supported_video(path):
            logger.info("Queued moved file: %s", path)
            file_queue.put(path)


def worker() -> None:
    while True:
        file_path = file_queue.get()
        try:
            process_file(file_path)
        except Exception:
            logger.exception("Failed to process file: %s", file_path)
        finally:
            file_queue.task_done()


def main() -> None:
    event_handler = FileCreatedHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_FOLDER, recursive=False)
    observer.start()

    threading.Thread(target=worker, daemon=True).start()

    logger.info("Monitoring folder: %s", WATCH_FOLDER)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping observer")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
