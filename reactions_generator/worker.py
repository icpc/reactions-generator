"""Continuous submission rendering and deterministic worker partitioning."""

import hashlib
import logging
import math
import os
from pathlib import Path
from threading import Event, Thread

import ffmpeg
from tqdm import tqdm

from reactions_generator.defaults import Defaults
from reactions_generator.overlayer import build_submission, fetch_runs
from reactions_generator.scheduler import SubmissionQueue

logger = logging.getLogger(__name__)


def log_error(error_string: str, id: str, output_directory: str) -> None:
    logger.error("Error in %s: %s", id, error_string)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{id}.err").write_text(error_string, encoding="utf-8")


def stable_hash(input_string: str) -> int:
    return int.from_bytes(hashlib.sha256(input_string.encode("utf-8")).digest())


def poll_submissions(
    url: str,
    queue: SubmissionQueue,
    stop: Event,
    output_directory: str,
    total_workers: int,
    worker_id: int,
    poll_interval: float,
) -> None:
    """Refresh pending work independently of the foreground encoder."""
    directory = Path(output_directory)
    while not stop.is_set():
        try:
            runs = fetch_runs(url)
            queue.update(
                [
                    run
                    for run in runs
                    if stable_hash(run.id) % total_workers == worker_id
                    and not (directory / f"{run.id}.mp4").exists()
                ]
            )
        except Exception:
            logger.exception("Failed to refresh submissions from %s", url)
        if stop.wait(poll_interval):
            return


def continuous_build_submission(
    url: str,
    success_audio_path: str = Defaults.success_audio_path,
    fail_audio_path: str = Defaults.fail_audio_path,
    output_directory: str = Defaults.output_directory,
    vcodec: str = Defaults.vcodec,
    acodec: str = Defaults.acodec,
    vertical: bool = True,
    total_workers: int = 1,
    worker_id: int = 0,
    sound: bool = Defaults.sound,
    hwaccel: str | None = Defaults.hwaccel,
    poll_interval: float = 2.0,
) -> None:
    """Poll during renders; prioritize first-to-solve, accepted, then other runs."""
    if total_workers <= 0 or not 0 <= worker_id < total_workers:
        raise ValueError(
            "Workers require total_workers > 0 and 0 <= worker_id < total_workers"
        )
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("Poll interval must be a positive finite number")
    os.makedirs(output_directory, exist_ok=True)

    queue = SubmissionQueue()
    stop = Event()
    poller = Thread(
        target=poll_submissions,
        args=(
            url,
            queue,
            stop,
            output_directory,
            total_workers,
            worker_id,
            poll_interval,
        ),
        name="submission-poller",
        daemon=True,
    )
    poller.start()
    try:
        with tqdm(desc="Rendering submissions", unit="submission") as progress:
            while not stop.is_set():
                run = queue.take(timeout=0.5)
                if run is None:
                    continue
                try:
                    build_submission(
                        id=run.id,
                        url=url,
                        success_audio_path=success_audio_path,
                        fail_audio_path=fail_audio_path,
                        output_directory=output_directory,
                        overwrite=False,
                        print_progress=False,
                        vcodec=vcodec,
                        acodec=acodec,
                        vertical=vertical,
                        sound=sound,
                        hwaccel=hwaccel,
                    )
                    progress.update(1)
                except ffmpeg.FFMpegExecuteError as e:
                    log_error(
                        os.linesep.join(str(x) for x in (e, e.stdout, e.stderr)),
                        id=run.id,
                        output_directory=output_directory,
                    )
                except Exception as e:  # noqa: BLE001 - One failed run must not stop the worker.
                    log_error(str(e), id=run.id, output_directory=output_directory)
                finally:
                    queue.complete(run.id)
    finally:
        stop.set()
        queue.close()
        poller.join(timeout=1)
