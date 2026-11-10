"""Benchmark complete reaction renders and local worker concurrency."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Required, TypedDict

import ffmpeg

from reactions_generator.defaults import Defaults
from reactions_generator.encoding import (
    CODECS,
    EncoderConfig,
    make_encoder,
    probe_encoder,
)
from reactions_generator.nvidia_full import probe_nvidia_full, settings_from_cli
from reactions_generator.rendering import render_reaction


@dataclass(frozen=True)
class RenderSample:
    ok: bool
    seconds: float
    bytes: int = 0
    video_seconds: float = 0.0
    error: str = ""


class BenchmarkResult(TypedDict, total=False):
    hardware: Required[str]
    codec: Required[str]
    status: Required[str]
    reason: str
    preset: str | int | None
    quality: str
    encoder_options: dict[str, str | int]
    workers: int
    repetition: int
    jobs: int
    successful: int
    failed: int
    wall_seconds: float
    videos_per_second: float
    video_seconds_per_second: float
    mean_latency_seconds: float | None
    median_latency_seconds: float | None
    total_bytes: int
    mean_file_bytes: int | None
    errors: list[str]


def parse_worker_counts(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(
            "--worker-counts must be comma-separated positive integers"
        ) from exc
    if not values or any(v <= 0 for v in values) or len(values) != len(set(values)):
        raise ValueError("--worker-counts must contain unique positive integers")
    return values


def inspect_output(path: Path, expect_audio: bool = True) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    data = json.loads(result.stdout)
    streams = data["streams"]
    videos = [item for item in streams if item.get("codec_type") == "video"]
    if len(videos) != 1 or videos[0].get("codec_name") != "h264":
        raise ValueError(f"Unexpected video codec in {path}")
    if int(videos[0]["width"]) != 1080 or int(videos[0]["height"]) != 1920:
        raise ValueError(f"Unexpected dimensions in {path}")
    if expect_audio and not any(s.get("codec_type") == "audio" for s in streams):
        raise ValueError(f"Missing audio stream in {path}")
    duration = float(data["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Invalid duration in {path}")
    return duration


def prepare_clip(source: str, destination: Path, duration: float) -> str:
    """Copy a short sample without changing the input codecs being benchmarked."""
    if Path(source).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return source
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            source,
            "-t",
            str(duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return str(destination)


def benchmark(
    hardware: str = "all",
    worker_counts: str = "1,2,4",
    quality: str = "balanced",
    repetitions: int = 1,
    output_directory: str = "out/benchmarks",
    webcam_source: str = Defaults.webcam_source,
    screen_source: str = Defaults.screen_source,
    background_source: str = Defaults.background_source,
    preset: str | None = None,
    vaapi_device: str | None = None,
    duration_seconds: float = 5.0,
    jobs_per_worker: int = 1,
) -> None:
    """Measure whole-video throughput at fixed nominal quality."""
    if hardware not in {*CODECS, "all"}:
        raise ValueError(f"Unknown hardware {hardware!r}")
    if repetitions < 1:
        raise ValueError("--repetitions must be >= 1")
    if hardware == "all" and preset is not None:
        raise ValueError("--preset with --hardware all is ambiguous")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("--duration-seconds must be finite and positive")
    if jobs_per_worker < 1:
        raise ValueError("--jobs-per-worker must be >= 1")
    counts = parse_worker_counts(worker_counts)
    for filename in [
        webcam_source,
        screen_source,
        background_source,
        Defaults.logo_source,
        Defaults.success_audio_path,
        Defaults.fail_audio_path,
    ]:
        if not Path(filename).is_file():
            raise ValueError(f"Missing benchmark fixture: {filename}")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    candidates = list(CODECS) if hardware == "all" else [hardware]
    jobs = jobs_per_worker * max(counts)
    results: list[BenchmarkResult] = []

    def render_one(backend: str, job_path: Path) -> RenderSample:
        started = time.perf_counter()
        try:
            render_reaction(
                hardware=backend,
                quality=quality,
                preset=preset,
                vaapi_device=vaapi_device,
                webcam_source=webcam_source,
                screen_source=screen_source,
                background_source=background_source,
                output_path=str(job_path),
                print_progress=False,
            )
            video_duration = inspect_output(job_path)
            return RenderSample(
                ok=True,
                seconds=time.perf_counter() - started,
                bytes=job_path.stat().st_size,
                video_seconds=video_duration,
            )
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            ffmpeg.FFMpegExecuteError,
        ) as exc:
            return RenderSample(
                ok=False,
                seconds=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        finally:
            job_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="renders-", dir=destination) as tmp:
        temporary = Path(tmp)
        # Preparation is performed once and excluded from all timed batches.
        webcam_source = prepare_clip(
            webcam_source, temporary / "webcam.mkv", duration_seconds
        )
        screen_source = prepare_clip(
            screen_source, temporary / "screen.mkv", duration_seconds
        )
        for backend in candidates:
            config: EncoderConfig | None = None
            try:
                config = make_encoder(
                    backend, quality, preset, vaapi_device=vaapi_device
                )
                if backend == "nvidia_full":
                    available, reason = probe_nvidia_full(
                        webcam_source,
                        settings_from_cli(quality=quality, preset=preset, vcodec=None),
                    )
                else:
                    available, reason = probe_encoder(config)
            except (ValueError, OSError) as exc:
                available, reason = False, str(exc)
            if not available:
                if hardware == "nvidia_full":
                    raise RuntimeError(f"nvidia_full unavailable: {reason}")
                results.append(
                    {
                        "hardware": backend,
                        "codec": CODECS[backend],
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                print(f"{backend:8} SKIP {reason}")
                continue
            assert config is not None
            warmup = render_one(backend, temporary / f"warmup-{backend}.mp4")
            if not warmup.ok:
                if hardware == "nvidia_full":
                    raise RuntimeError(
                        f"nvidia_full full-render warm-up failed: {warmup.error}"
                    )
                results.append(
                    {
                        "hardware": backend,
                        "codec": config.codec,
                        "status": "skipped",
                        "reason": f"Full render warm-up failed: {warmup.error}",
                    }
                )
                continue
            for workers in counts:
                for repeat in range(repetitions):
                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        samples = list(
                            executor.map(
                                partial(render_one, backend),
                                [
                                    temporary
                                    / f"{backend}-{workers}-{repeat}-{index}.mp4"
                                    for index in range(jobs)
                                ],
                            )
                        )
                    elapsed = time.perf_counter() - started
                    successful = [s for s in samples if s.ok]
                    failed = [s for s in samples if not s.ok]
                    durations = [s.seconds for s in successful]
                    entry: BenchmarkResult = {
                        "hardware": backend,
                        "codec": config.codec,
                        "preset": config.options.get(
                            "preset", config.options.get("quality")
                        ),
                        "quality": quality,
                        "encoder_options": config.options,
                        "workers": workers,
                        "repetition": repeat + 1,
                        "jobs": jobs,
                        "successful": len(successful),
                        "failed": len(failed),
                        "wall_seconds": round(elapsed, 3),
                        "videos_per_second": round(len(successful) / elapsed, 4),
                        "video_seconds_per_second": round(
                            sum(s.video_seconds for s in successful) / elapsed,
                            4,
                        ),
                        "mean_latency_seconds": round(statistics.mean(durations), 3)
                        if durations
                        else None,
                        "median_latency_seconds": round(statistics.median(durations), 3)
                        if durations
                        else None,
                        "total_bytes": sum(s.bytes for s in successful),
                        "mean_file_bytes": round(
                            statistics.mean(s.bytes for s in successful)
                        )
                        if successful
                        else None,
                        "status": "ok" if not failed else "failed",
                        "errors": [s.error for s in failed[:3]],
                    }
                    results.append(entry)
                    print(
                        f"{backend:8} workers={workers:<3} rep={repeat + 1} "
                        f"ok={len(successful)}/{jobs} elapsed={elapsed:.2f}s "
                        f"throughput={entry['videos_per_second']} videos/s"
                    )

    winners = [
        result
        for result in results
        if result.get("status") == "ok" and "videos_per_second" in result
    ]
    best = (
        max(winners, key=lambda result: result["videos_per_second"])
        if winners
        else None
    )
    version = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=False
    )
    report = {
        "platform": " ".join(
            part
            for part in (platform.system(), platform.release(), platform.machine())
            if part
        ),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "ffmpeg_version": version.stdout.splitlines()[0]
        if version.stdout
        else "unknown",
        "jobs_per_run": jobs,
        "requested_duration_seconds": duration_seconds,
        "jobs_per_worker": jobs_per_worker,
        "results": results,
        "recommended": best,
        "warning": "Nominal quality settings are not equal perceptual quality across encoders.",
    }
    report_path = destination / "results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Results: {report_path}")
    if best:
        assert "workers" in best
        print(
            f"Best: {best['hardware']}, {best['workers']} workers, {best['videos_per_second']} videos/s"
        )
        print(
            f"Production: main all OVERLAYER_URL --hardware {best['hardware']} "
            f"--quality {quality} --total-workers {best['workers']}"
        )
    else:
        print("No configuration completed all jobs successfully.")
