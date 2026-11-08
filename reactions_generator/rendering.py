"""Card video encoding and reaction composition with FFmpeg."""

import atexit
import logging
import math
import os
import stat
import subprocess
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import ffmpeg
import numpy as np
from PIL import Image

from reactions_generator.card import Card
from reactions_generator.defaults import Defaults
from reactions_generator.media import get_metadata, load_image_or_color

logger = logging.getLogger(__name__)


def pipe_card_input(width: int, height: int, fps: float) -> ffmpeg.VideoStream:
    return ffmpeg.input(
        "pipe:",
        f="rawvideo",
        pix_fmt="rgba",
        s=f"{width}x{height}",
        r=fps,
    ).video


def to_ffmpeg_frame(image: Image.Image) -> bytes:
    return np.asarray(image, np.uint8).tobytes()


def render(
    ffmpeg_input: Sequence[ffmpeg.AudioStream | ffmpeg.VideoStream],
    card: Card,
    last_frame: int,
    output_path: str,
    fps: float,
    print_progress: bool,
    vcodec: str,
    acodec: str | None,
) -> None:
    if fps <= 0 or last_frame < 0:
        raise ValueError("Frame rate must be positive and last frame nonnegative")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # FFmpeg creates the file with normal permissions; the directory keeps the
    # temporary output isolated and on the same filesystem for atomic replacement.
    temporary_directory = tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.stem}-"
    )
    tmp_output = Path(temporary_directory.name) / output.name
    try:
        graph = (
            ffmpeg.output(
                *ffmpeg_input,
                filename=str(tmp_output),
                vcodec=vcodec,
                acodec=acodec,
                r=fps,
                pix_fmt="yuv420p",
            )
            .global_args(loglevel="info" if print_progress else "error")
            .overwrite_output()
        )
        command = graph.compile(auto_fix=False)
        # A file can capture arbitrary diagnostics without filling a stderr pipe.
        with tempfile.TemporaryFile() as errors:
            process: subprocess.Popen[bytes] = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=None if print_progress else errors,
            )

            def clean_up() -> None:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                if process.stdin is not None:
                    with suppress(BrokenPipeError):
                        process.stdin.close()

            def encoder_error() -> ffmpeg.FFMpegExecuteError:
                errors.seek(0)
                return ffmpeg.FFMpegExecuteError(
                    process.returncode, " ".join(command), b"", errors.read()
                )

            atexit.register(clean_up)
            try:
                if process.stdin is None:
                    raise RuntimeError("FFmpeg input pipe was not created")
                try:
                    for frame in range(last_frame + 1):
                        process.stdin.write(to_ffmpeg_frame(card.render_frame(frame)))
                    process.stdin.close()
                except BrokenPipeError:
                    process.wait()
                    raise encoder_error() from None
                if process.wait() != 0:
                    raise encoder_error()
                with suppress(FileNotFoundError):
                    tmp_output.chmod(stat.S_IMODE(output.stat().st_mode))
                os.replace(tmp_output, output)
            finally:
                atexit.unregister(clean_up)
                clean_up()
    finally:
        temporary_directory.cleanup()


def render_card(
    title: str = Defaults.title,
    subtitle: str = Defaults.subtitle,
    hashtag: str = Defaults.hashtag,
    task: str = Defaults.task,
    time: float = Defaults.time,
    outcome: str = Defaults.outcome,
    success: bool = Defaults.success,
    rank_before: int = Defaults.rank_before,
    rank_after: int = Defaults.rank_after,
    logo_source: str = Defaults.logo_source,
    fps: float = Defaults.fps,
    duration_seconds: float = Defaults.duration_seconds,
    output_path: str = Defaults.output_path,
    vcodec: str = Defaults.vcodec,
    print_progress: bool = True,
) -> None:
    """Render card as a video file."""
    last_frame = math.floor(duration_seconds * fps)
    animation_start = max(0, round(last_frame - 30 * fps))
    logo = load_image_or_color(logo_source, dimensions=(152, 152))

    card_creator = Card(
        title=title,
        subtitle=subtitle,
        hashtag=hashtag,
        task=task,
        time=time,
        outcome=outcome,
        success=success,
        rank_before=rank_before,
        rank_after=rank_after,
        logo=logo,
        animation_start=animation_start,
        fps=fps,
        width=1000,
        height=306,
    )

    render(
        [pipe_card_input(card_creator.width, card_creator.height, fps)],
        card=card_creator,
        last_frame=min(last_frame, animation_start + 10),
        output_path=output_path,
        fps=fps,
        print_progress=print_progress,
        vcodec=vcodec,
        acodec=None,
    )


def render_reaction(
    title: str = Defaults.title,
    subtitle: str = Defaults.subtitle,
    hashtag: str = Defaults.hashtag,
    task: str = Defaults.task,
    time: float = Defaults.time,
    outcome: str = Defaults.outcome,
    success: bool = Defaults.success,
    rank_before: int = Defaults.rank_before,
    rank_after: int = Defaults.rank_after,
    logo_source: str = Defaults.logo_source,
    webcam_source: str = Defaults.webcam_source,
    screen_source: str = Defaults.screen_source,
    background_source: str = Defaults.background_source,
    success_audio_path: str = Defaults.success_audio_path,
    fail_audio_path: str = Defaults.fail_audio_path,
    output_path: str = Defaults.output_path,
    vcodec: str = Defaults.vcodec,
    acodec: str = Defaults.acodec,
    print_progress: bool = True,
    sound: bool = Defaults.sound,
    background_position: tuple[
        float, float, float, float
    ] = Defaults.background_position,
    card_position: tuple[int, int, int, int] = Defaults.card_position,
    webcam_position: tuple[float, float, float, float] = Defaults.webcam_position,
    screen_position: tuple[float, float, float, float] = Defaults.screen_position,
    hwaccel: str | None = Defaults.hwaccel,
) -> None:
    """Render reaction as a video file."""
    metadata = get_metadata(webcam_source, expect_audio=True)
    fps = float(metadata.fps)
    last_frame = math.floor(metadata.duration * fps)
    animation_start = max(0, round(last_frame - 30 * fps))

    try:
        get_metadata(screen_source)
    except Exception as e:  # noqa: BLE001 - Fall back to a static image if probing fails.
        logger.warning("Failed to get metadata for screen source: %s", e)
        screen_source = f"{task}.png"

    logo = load_image_or_color(logo_source, dimensions=(152, 152))
    card_creator = Card(
        title=title,
        subtitle=subtitle,
        hashtag=hashtag,
        task=task,
        time=time,
        outcome=outcome,
        success=success,
        rank_before=rank_before,
        rank_after=rank_after,
        logo=logo,
        animation_start=animation_start,
        fps=fps,
        width=card_position[2],
        height=card_position[3],
    )

    background = ffmpeg.input(background_source, hwaccel=hwaccel).scale(
        w=background_position[2], h=background_position[3]
    )
    webcam = (
        ffmpeg.input(webcam_source, hwaccel=hwaccel)
        .video.scale(w=webcam_position[2], h=webcam_position[3])
        .setpts(expr="PTS-STARTPTS")
    )
    screen = (
        ffmpeg.input(screen_source, hwaccel=hwaccel)
        .video.scale(w=screen_position[2], h=screen_position[3])
        .setpts(expr="PTS-STARTPTS")
    )
    card = pipe_card_input(card_position[2], card_position[3], fps)

    action_sound = ffmpeg.input(
        success_audio_path if success else fail_audio_path
    ).adelay(delays=animation_start / fps * 1000, all=True)

    video = (
        background.overlay(webcam, x=webcam_position[0], y=webcam_position[1])
        .overlay(screen, x=screen_position[0], y=screen_position[1])
        .overlay(card, x=card_position[0], y=card_position[1], eof_action="repeat")
    )

    audio = (
        ffmpeg.filters.amix(
            ffmpeg.input(webcam_source, hwaccel=hwaccel).audio,
            action_sound,
            duration="longest",
        )
        if metadata.audio and sound
        else action_sound
    )

    # The last card animation finishes six frames after animation_start.
    # FFmpeg's overlay eof_action="repeat" holds this final card frame
    # while the webcam, screen and audio continue to the end.
    render(
        [audio, video],
        card=card_creator,
        last_frame=min(last_frame, animation_start + 6),
        output_path=output_path,
        fps=fps,
        print_progress=print_progress,
        vcodec=vcodec,
        acodec=acodec,
    )
