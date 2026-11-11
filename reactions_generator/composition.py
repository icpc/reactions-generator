"""Shared video composition with software and CUDA filter operations."""

from __future__ import annotations

import math

import ffmpeg

from reactions_generator.encoding import EncoderConfig


def _even_dimension(value: float) -> int:
    """Scale 4:2:0 video to the nearest positive, even dimension."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Video dimensions must be finite and positive")
    result = round(value / 2) * 2
    if result < 2:
        raise ValueError("Video dimensions must be at least 2 pixels")
    return result


def _even_offset(value: float) -> int:
    """Match 4:2:0 chroma alignment, including fractional layout positions."""
    if not math.isfinite(value):
        raise ValueError("Video offsets must be finite")
    return math.floor(value / 2) * 2


def _aligned_position(
    rectangle: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x, y, width, height = rectangle
    return (
        _even_offset(x),
        _even_offset(y),
        _even_dimension(width),
        _even_dimension(height),
    )


def _upload_static(video: ffmpeg.VideoStream, fps: float) -> ffmpeg.VideoStream:
    """Upload a single image and repeat its GPU frame reference."""
    return (
        video.format(pix_fmts="yuv420p")
        .hwupload()
        .loop(loop=-1, size=1, start=0)
        .settb(expr="AVTB")
        .setpts(expr=f"N/({fps:.12g}*TB)")
    )


def video_input(
    source: str,
    encoding: EncoderConfig,
    *,
    hwaccel: str | None = None,
    static: bool = False,
) -> ffmpeg.AVStream:
    """Decode moving CUDA inputs on the GPU and keep static uploads explicit."""
    if encoding.hardware == "nvidia_full":
        return ffmpeg.input(
            source,
            hwaccel=None if static else "cuda",
            extra_options=None
            if static
            else {"hwaccel_device": "gpu", "hwaccel_output_format": "cuda"},
        )
    return ffmpeg.input(source, hwaccel=hwaccel)


def scale_video(
    video: ffmpeg.VideoStream,
    width: float,
    height: float,
    *,
    cuda: bool,
    fps: float,
    static: bool = False,
) -> ffmpeg.VideoStream:
    """Scale and reset timestamps, uploading/repeating static CUDA frames."""
    width = _even_dimension(width)
    height = _even_dimension(height)
    if not cuda:
        return video.scale(w=width, h=height).setpts(expr="PTS-STARTPTS")
    if static:
        """Don't know if this is really necessary, but we scale once on the CPU, and only then upload to the GPU"""
        scaled = video.scale(w=width, h=height, flags="bicubic")
        return _upload_static(scaled, fps)
    return video.scale_cuda(w=width, h=height, format="yuv420p").setpts(
        expr="PTS-STARTPTS"
    )


def prepare_card_video(card: ffmpeg.VideoStream, *, cuda: bool) -> ffmpeg.VideoStream:
    """Keep Pillow's alpha channel when uploading card frames to CUDA."""
    if cuda:
        return card.format(pix_fmts="yuva420p").hwupload().setpts(expr="PTS-STARTPTS")
    return card


def overlay_video(
    video: ffmpeg.VideoStream,
    layer: ffmpeg.VideoStream,
    x: float,
    y: float,
    *,
    cuda: bool,
) -> ffmpeg.VideoStream:
    """Overlay a layer and repeat its final frame on the selected backend."""
    x = _even_offset(x)
    y = _even_offset(y)
    if cuda:
        return video.overlay_cuda(layer, x=x, y=y, eof_action="repeat")
    return video.overlay(layer, x=x, y=y, eof_action="repeat")


def _validate_cuda_layout(
    background: tuple[float, float, float, float],
    webcam: tuple[float, float, float, float],
    screen: tuple[float, float, float, float],
    card: tuple[int, int, int, int],
) -> None:
    bx, by, bw, bh = _aligned_position(background)
    cx, cy, cw, ch = _aligned_position(card)
    if bx != 0 or by != 0:
        raise ValueError("The NVIDIA renderer requires a background at (0, 0)")
    if (cw, ch) != (card[2], card[3]):
        raise ValueError("RGBA card dimensions must be even")
    for x, y, width, height in (
        _aligned_position(webcam),
        _aligned_position(screen),
        (cx, cy, cw, ch),
    ):
        if x < 0 or y < 0 or x + width > bw or y + height > bh:
            raise ValueError("An overlay lies outside the NVIDIA canvas")


def compose_video(
    *,
    webcam: ffmpeg.VideoStream,
    screen: ffmpeg.VideoStream,
    background: ffmpeg.VideoStream,
    card: ffmpeg.VideoStream,
    screen_is_video: bool,
    fps: float,
    background_position: tuple[float, float, float, float],
    webcam_position: tuple[float, float, float, float],
    screen_position: tuple[float, float, float, float],
    card_position: tuple[int, int, int, int],
    encoding: EncoderConfig,
) -> ffmpeg.VideoStream:
    """Compose the same layers through backend-aware scale/overlay helpers."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be finite and positive")
    cuda = encoding.hardware == "nvidia_full"
    if cuda:
        _validate_cuda_layout(
            background_position, webcam_position, screen_position, card_position
        )

    background = scale_video(
        background,
        background_position[2],
        background_position[3],
        cuda=cuda,
        fps=fps,
        static=True,
    )
    webcam = scale_video(
        webcam,
        webcam_position[2],
        webcam_position[3],
        cuda=cuda,
        fps=fps,
    )
    screen = scale_video(
        screen,
        screen_position[2],
        screen_position[3],
        cuda=cuda,
        fps=fps,
        static=not screen_is_video,
    )
    card = prepare_card_video(card, cuda=cuda)

    video = background
    for layer, position in (
        (webcam, webcam_position),
        (screen, screen_position),
        (card, card_position),
    ):
        video = overlay_video(video, layer, position[0], position[1], cuda=cuda)
    return video
