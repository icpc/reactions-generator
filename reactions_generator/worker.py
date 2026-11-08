"""Continuous submission rendering and deterministic worker partitioning."""

import hashlib
import logging
import os
from pathlib import Path

import ffmpeg
from tqdm import tqdm

from reactions_generator.defaults import Defaults
from reactions_generator.overlayer import build_submission, fetch_submission_ids

logger = logging.getLogger(__name__)


def log_error(error_string: str, id: str, output_directory: str) -> None:
    logger.error("Error in %s: %s", id, error_string)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{id}.err").write_text(error_string, encoding="utf-8")


def stable_hash(input_string: str) -> int:
    return int.from_bytes(hashlib.sha256(input_string.encode("utf-8")).digest())


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
) -> None:
    """Continuously render this worker's deterministic share of submissions."""
    if total_workers <= 0 or not 0 <= worker_id < total_workers:
        raise ValueError(
            "Workers require total_workers > 0 and 0 <= worker_id < total_workers"
        )
    os.makedirs(output_directory, exist_ok=True)

    while True:
        ids = fetch_submission_ids(url)

        filtered = [id for id in ids if stable_hash(id) % total_workers == worker_id]

        for id in tqdm(filtered, desc="Rendering submissions"):
            try:
                build_submission(
                    id=id,
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
            except ffmpeg.exceptions.FFMpegExecuteError as e:
                log_error(
                    os.linesep.join([str(x) for x in [e, e.stdout, e.stderr]]),
                    id=id,
                    output_directory=output_directory,
                )
            except Exception as e:  # noqa: BLE001 - One failed submission must not stop the worker.
                log_error(str(e), id=id, output_directory=output_directory)
