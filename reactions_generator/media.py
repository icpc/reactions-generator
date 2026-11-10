"""Image sources and media metadata, independent of CLI commands."""

import logging
import re
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import NamedTuple, TypedDict, cast
from urllib.parse import unquote, urlsplit

import ffmpeg
import requests
from PIL import Image, ImageColor

logger = logging.getLogger(__name__)


def load_image_or_color(source: str, dimensions: tuple[int, int]) -> Image.Image:
    if source.startswith("#"):
        return Image.new("RGBA", dimensions, color=ImageColor.getrgb(source))

    if source.lower().startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as image:
            return image.convert("RGBA")

    with Image.open(source) as image:
        return image.convert("RGBA")


class Metadata(NamedTuple):
    fps: Fraction
    duration: float
    audio: bool


class ProbeStream(TypedDict, total=False):
    codec_type: str
    r_frame_rate: str
    avg_frame_rate: str
    duration: str


class ProbeFormat(TypedDict, total=False):
    duration: str


class ProbeResult(TypedDict):
    streams: list[ProbeStream]
    format: ProbeFormat


def direct_ffprobe(video_source: str) -> Metadata:
    # FFprobe owns this schema; keep its untyped JSON at this boundary.
    result = cast(ProbeResult, ffmpeg.probe(video_source))
    streams = result["streams"]
    probe = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    if probe is None:
        raise ValueError(f"No video stream found in {video_source}")

    try:
        fps = Fraction(probe.get("r_frame_rate", "0/0"))
        if fps <= 0:
            raise ValueError("Frame rate must be positive")
    except (KeyError, ValueError, ZeroDivisionError):
        fps = Fraction(probe.get("avg_frame_rate", "0/0"))

    duration = probe.get("duration") or result.get("format", {}).get("duration")
    if fps <= 0 or duration is None:
        raise ValueError(f"Missing frame rate or duration in {video_source}")
    return Metadata(
        fps=fps,
        duration=float(duration),
        audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def stage_media(source: str, directory: Path, name: str) -> str:
    """Download an HTTP media source into the render's temporary directory."""
    url = urlsplit(source)
    if url.scheme not in {"http", "https"}:
        return source

    suffix = Path(unquote(url.path)).suffix
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,12}", suffix):
        suffix = ""
    destination = directory / f"{name}{suffix}"
    with requests.get(source, stream=True, timeout=30) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                output.write(chunk)
    return str(destination)


def metadata_via_ffmpeg(video_source: str) -> Metadata:
    _, error_output = (
        ffmpeg.input(video_source)
        .output(filename="null", f="null")
        .run(capture_stderr=True)
    )

    diagnostic = error_output.decode("utf-8", errors="replace")

    pattern = r"frame=\s*(\d+)[^\r\n]*?time=(\d+):(\d{2}):(\d{2}\.\d+)"
    matches = re.findall(pattern, diagnostic)
    if not matches:
        raise ValueError(f"No frame timing found for {video_source}")
    frames, hours, minutes, seconds = matches[-1]

    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if duration <= 0 or int(frames) <= 0:
        raise ValueError(f"Invalid frame timing for {video_source}")

    return Metadata(
        fps=Fraction(frames) / Fraction(str(duration)),
        duration=duration,
        audio="Audio:" in diagnostic,
    )


def get_metadata(video_source: str, expect_audio: bool = False) -> Metadata:
    for metadata_method in (direct_ffprobe, metadata_via_ffmpeg):
        try:
            metadata = metadata_method(video_source)
            if not metadata.audio and expect_audio:
                logger.warning("Audio missing in %s", video_source)
            return metadata
        except Exception as e:  # noqa: BLE001 - Try the next independent metadata probe.
            logger.debug("Metadata probe failed for %s: %s", video_source, e)
    raise ValueError(f"No metadata found for {video_source}")
