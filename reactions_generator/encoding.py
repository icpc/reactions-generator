"""Explicit, portable H.264 encoder selection (no implicit GPU detection)."""

from __future__ import annotations

import glob
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)

CODECS = {
    "cpu": "libx264",
    "nvidia": "h264_nvenc",
    "nvidia_full": "h264_nvenc",
    "intel": "h264_qsv",
    "amd": "h264_amf",
    "apple": "h264_videotoolbox",
    "vaapi": "h264_vaapi",
}
BACKENDS = {
    codec: backend for backend, codec in CODECS.items() if backend != "nvidia_full"
}
# These values are arbitrary starting points on each encoder's own quality scale.
# They favor small reaction videos; they do not imply equivalent quality or size.
QUALITY = {
    "cpu": {"low": 29, "balanced": 26, "high": 21},
    "nvidia": {"low": 32, "balanced": 29, "high": 23},
    "intel": {"low": 32, "balanced": 29, "high": 23},
    "amd": {"low": 32, "balanced": 29, "high": 23},
    "apple": {"low": 5, "balanced": 10, "high": 30},
    "vaapi": {"low": 32, "balanced": 29, "high": 23},
}
PRESETS = {
    "cpu": (
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ),
    "nvidia": ("p1", "p2", "p3", "p4", "p5", "p6", "p7"),
    "intel": ("veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"),
    "amd": ("speed", "balanced", "quality"),
    "apple": (),
    "vaapi": (),
}
DEFAULT_PRESETS = {
    "cpu": "veryfast",
    "nvidia": "p3",
    "intel": "veryfast",
    "amd": "speed",
    "apple": None,
    "vaapi": None,
}


@dataclass(frozen=True)
class EncoderConfig:
    hardware: str
    codec: str
    options: dict[str, str | int]
    pixel_format: str | None
    vaapi_device: str | None = None

    def output_options(self, acodec: str | None) -> dict[str, str | int]:
        return {
            **self.options,
            **({"b:a": "96k"} if acodec == "aac" else {}),
            "movflags": "+faststart",
        }

    def device_args(self) -> list[str]:
        if self.hardware == "nvidia_full":
            return ["-init_hw_device", "cuda=gpu:0", "-filter_hw_device", "gpu"]
        if self.codec != "h264_vaapi":
            return []
        assert self.vaapi_device is not None
        return [
            "-init_hw_device",
            f"vaapi=rg_vaapi:{self.vaapi_device}",
            "-filter_hw_device",
            "rg_vaapi",
        ]


def make_encoder(
    hardware: str = "cpu",
    quality: str = "balanced",
    preset: str | None = None,
    vcodec: str | None = None,
    vaapi_device: str | None = None,
) -> EncoderConfig:
    """Resolve an explicitly chosen encoder and its quality/speed parameters."""
    if hardware not in CODECS:
        raise ValueError(f"Unknown --hardware {hardware!r}; choose {', '.join(CODECS)}")
    if hardware == "nvidia_full":
        if vaapi_device is not None:
            raise ValueError("--vaapi-device requires --hardware vaapi")
        if vcodec not in {None, "h264_nvenc"}:
            raise ValueError("--hardware nvidia_full requires --vcodec h264_nvenc")
        if quality == "medium":
            quality = "balanced"
    if quality not in {"low", "balanced", "high"}:
        raise ValueError("--quality must be low, balanced, or high")
    if vcodec is not None and hardware not in {"cpu", "nvidia_full"}:
        raise ValueError("--vcodec cannot be combined with a non-CPU --hardware")

    codec = vcodec or CODECS[hardware]
    backend = BACKENDS.get(codec)
    if backend is None:
        if vaapi_device is not None:
            raise ValueError("--vaapi-device requires h264_vaapi")
        logger.warning("Custom codec %s: --quality is not automatically mapped", codec)
        return EncoderConfig(
            "custom", codec, {"preset": preset} if preset else {}, "yuv420p"
        )

    if preset is None:
        preset = "p5" if hardware == "nvidia_full" else DEFAULT_PRESETS[backend]
    elif preset not in PRESETS[backend]:
        raise ValueError(
            f"--preset for {backend} must be one of {', '.join(PRESETS[backend]) or '(none)'}"
        )

    if vaapi_device and backend != "vaapi":
        raise ValueError("--vaapi-device requires --hardware vaapi")

    opts: dict[str, str | int]
    if backend == "cpu":
        opts = {"crf": QUALITY[backend][quality]}
    elif backend == "nvidia":
        opts = {"rc": "vbr", "cq": QUALITY[backend][quality], "b:v": 0}
        if hardware == "nvidia_full":
            opts.update(tune="hq", bf=2)
    elif backend == "intel":
        opts = {"global_quality": QUALITY[backend][quality], "look_ahead": 0}
    elif backend == "amd":
        value = QUALITY[backend][quality]
        opts = {"quality": preset or "speed", "rc": "cqp", "qp_i": value, "qp_p": value}
    elif backend == "apple":
        opts = {"q:v": QUALITY[backend][quality]}
    else:
        opts = {"rc_mode": "CQP", "qp": QUALITY[backend][quality]}

    if preset is not None and backend not in {"amd"}:
        opts["preset"] = preset

    if backend == "vaapi":
        if vaapi_device is None:
            vaapi_device = next(
                (
                    path
                    for path in sorted(glob.glob("/dev/dri/renderD*"))
                    if Path(path).exists()
                ),
                None,
            )
        if vaapi_device is None:
            raise ValueError(
                "No VAAPI render device found; pass --vaapi-device /dev/dri/renderDXXX"
            )
    return EncoderConfig(
        hardware if hardware == "nvidia_full" else backend,
        codec,
        opts,
        None
        if backend == "vaapi"
        else ("nv12" if backend in {"intel", "amd"} else "yuv420p"),
        vaapi_device,
    )


def prepare_video(
    video: ffmpeg.VideoStream, config: EncoderConfig
) -> ffmpeg.VideoStream:
    """Keep software composition; convert/upload only for the selected encoder."""
    if config.hardware in {"intel", "amd", "vaapi"}:
        video = video.format(pix_fmts="nv12")
    if config.hardware == "vaapi":
        video = video.hwupload()
    return video


def probe_encoder(config: EncoderConfig, timeout: float = 15.0) -> tuple[bool, str]:
    """Perform a real short encode, not just `ffmpeg -encoders` inspection."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        *config.device_args(),
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=128x128:rate=15",
        "-frames:v",
        "8",
    ]
    if config.hardware == "vaapi":
        command += ["-vf", "format=nv12,hwupload"]
    elif config.hardware in {"intel", "amd"}:
        command += ["-vf", "format=nv12"]
    command += ["-c:v", config.codec]
    for key, value in config.options.items():
        command += [f"-{key}", str(value)]
    if config.pixel_format:
        command += ["-pix_fmt", config.pixel_format]
    command += ["-f", "null", "-"]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, (result.stderr.strip() or f"ffmpeg returned {result.returncode}")[
            -800:
        ]
    return True, "ok"
