"""Continuous submission rendering and deterministic worker partitioning."""

import hashlib
import logging
import math
import os
import signal
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock, current_thread, main_thread
from types import FrameType

import ffmpeg

from reactions_generator.defaults import Defaults
from reactions_generator.overlayer import RunSummary, build_submission, fetch_runs
from reactions_generator.progress import RenderProgress
from reactions_generator.scheduler import SubmissionQueue

logger = logging.getLogger(__name__)


@contextmanager
def shutdown_signals(stop: Event) -> Generator[None, None, None]:
    """Mark cancellation before interrupted encoders report their errors."""
    if current_thread() is not main_thread():
        yield
        return

    def interrupt(_signum: int, _frame: FrameType | None) -> None:
        stop.set()

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


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
    output_directory: str,
    total_workers: int,
    worker_id: int | None,
) -> None:
    """Refresh one pending snapshot; retain the previous queue on API failure."""
    directory = Path(output_directory)
    try:
        runs = fetch_runs(url)
        queue.update(
            [
                run
                for run in runs
                if (
                    worker_id is None
                    or stable_hash(run.id) % total_workers == worker_id
                )
                and not (directory / f"{run.id}.mp4").exists()
                and not (directory / f"{run.id}.err").exists()
            ]
        )
    except Exception:
        logger.exception("Failed to refresh submissions from %s", url)


def continuous_build_submission(
    url: str,
    success_audio_path: str = Defaults.success_audio_path,
    fail_audio_path: str = Defaults.fail_audio_path,
    output_directory: str = Defaults.output_directory,
    vcodec: str | None = Defaults.vcodec,
    acodec: str = Defaults.acodec,
    hardware: str = Defaults.hardware,
    quality: str = Defaults.quality,
    preset: str | None = Defaults.preset,
    vaapi_device: str | None = Defaults.vaapi_device,
    vertical: bool = True,
    total_workers: int = 1,
    worker_id: int | None = None,
    sound: bool = Defaults.sound,
    hwaccel: str | None = Defaults.hwaccel,
    poll_interval: float = 2.0,
) -> None:
    """Poll during renders; prioritize first-to-solve, accepted, then other runs."""
    if total_workers <= 0 or (
        worker_id is not None and not 0 <= worker_id < total_workers
    ):
        raise ValueError(
            "Workers require total_workers > 0 and 0 <= worker_id < total_workers"
        )
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("Poll interval must be a positive finite number")
    print(
        f"Starting worker: hardware={hardware}, "
        f"total_workers={total_workers}, worker_id={worker_id}",
        flush=True,
    )
    os.makedirs(output_directory, exist_ok=True)

    queue = SubmissionQueue()
    stop = Event()

    def render_one(run: RunSummary) -> bool:
        try:
            print(f"Starting render: {run.id}", flush=True)
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
                hardware=hardware,
                quality=quality,
                preset=preset,
                vaapi_device=vaapi_device,
            )
            print(f"Finished render: {run.id}", flush=True)
            return True
        except ffmpeg.FFMpegExecuteError as e:
            if stop.is_set():
                return False
            log_error(
                os.linesep.join(str(x) for x in (e, e.stdout, e.stderr)),
                id=run.id,
                output_directory=output_directory,
            )
        except Exception as e:  # noqa: BLE001 - Isolate failed submissions.
            if not stop.is_set():
                log_error(str(e), id=run.id, output_directory=output_directory)
        return False

    # An explicit worker ID owns one distributed partition and renders serially.
    concurrency = total_workers if worker_id is None else 1
    with (
        shutdown_signals(stop),
        RenderProgress() as progress,
        ThreadPoolExecutor(
            max_workers=concurrency + 1, thread_name_prefix="reaction"
        ) as executor,
    ):
        progress_lock = Lock()

        def render_loop() -> None:
            try:
                while not stop.is_set() and (run := queue.take()) is not None:
                    succeeded = None
                    try:
                        succeeded = render_one(run)
                    finally:
                        with progress_lock:
                            queue.complete(run.id)
                            if succeeded is not None and not stop.is_set():
                                progress.report(queue.remaining, succeeded=succeeded)
            finally:
                stop.set()

        def poll_loop() -> None:
            try:
                while not stop.is_set():
                    poll_submissions(
                        url, queue, output_directory, total_workers, worker_id
                    )
                    with progress_lock:
                        progress.report(queue.remaining)
                    stop.wait(poll_interval)
            finally:
                stop.set()

        poller = executor.submit(poll_loop)
        renderers = [executor.submit(render_loop) for _ in range(concurrency)]
        try:
            stop.wait()
        finally:
            stop.set()
            queue.close()
        poller.result()
        for renderer in renderers:
            renderer.result()
