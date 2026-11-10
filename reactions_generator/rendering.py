"""Card video encoding and reaction composition with FFmpeg."""

import atexit
import logging
import math
import os
import re
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

import ffmpeg
import numpy as np
from PIL import Image

from reactions_generator.card import Card
from reactions_generator.defaults import Defaults
from reactions_generator.encoding import EncoderConfig, make_encoder, prepare_video
from reactions_generator.media import get_metadata, load_image_or_color, stage_media
from reactions_generator.nvidia_full import build_nvidia_full_command, settings_from_cli

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


def run_card_command(
    command: list[str],
    card: Card,
    last_frame: int,
    output_path: str,
    print_progress: bool,
) -> None:
    """Run an FFmpeg command whose last argument is the output path."""
    if last_frame < 0:
        raise ValueError("Last frame must be nonnegative")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=f".{output.stem}-"
    ) as temporary_dir:
        temporary_output = Path(temporary_dir) / output.name
        actual_command = [*command[:-1], str(temporary_output)]
        log_directory = Path("out/logs")
        log_directory.mkdir(parents=True, exist_ok=True)
        log_stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", output.stem)
        log_path = log_directory / f"{log_stem}-{uuid.uuid4().hex}.log"
        with tempfile.TemporaryFile() as errors:
            process: subprocess.Popen[bytes] = subprocess.Popen(
                actual_command,
                stdin=subprocess.PIPE,
                stderr=None if print_progress else errors,
                env={**os.environ, "FFREPORT": f"file={log_path.as_posix()}:level=32"},
            )

            def clean_up() -> None:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                if process.stdin is not None and not process.stdin.closed:
                    with suppress(BrokenPipeError):
                        process.stdin.close()

            def encoder_error() -> ffmpeg.FFMpegExecuteError:
                errors.seek(0)
                return ffmpeg.FFMpegExecuteError(
                    process.returncode,
                    " ".join(actual_command),
                    b"",
                    errors.read(),
                )

            atexit.register(clean_up)
            try:
                if process.stdin is None:
                    raise RuntimeError("FFmpeg card input pipe was not created")
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
                    temporary_output.chmod(stat.S_IMODE(output.stat().st_mode))
                os.replace(temporary_output, output)
            finally:
                atexit.unregister(clean_up)
                clean_up()


def render(
    ffmpeg_input: Sequence[ffmpeg.AudioStream | ffmpeg.VideoStream],
    card: Card,
    last_frame: int,
    output_path: str,
    fps: float,
    print_progress: bool,
    acodec: str | None,
    encoding: EncoderConfig,
) -> None:
    if fps <= 0 or last_frame < 0:
        raise ValueError("Frame rate must be positive and last frame nonnegative")
    graph = (
        ffmpeg.output(
            *ffmpeg_input,
            filename=output_path,
            vcodec=encoding.codec,
            acodec=acodec,
            r=fps,
            pix_fmt=encoding.pixel_format,
            extra_options={**encoding.options, "movflags": "+faststart"},
        )
        .global_args(loglevel="info" if print_progress else "error")
        .overwrite_output()
    )
    command = graph.compile(auto_fix=False)
    command = [command[0], *encoding.device_args(), *command[1:]]
    run_card_command(command, card, last_frame, output_path, print_progress)


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
    vcodec: str | None = Defaults.vcodec,
    hardware: str = Defaults.hardware,
    quality: str = Defaults.quality,
    preset: str | None = Defaults.preset,
    vaapi_device: str | None = Defaults.vaapi_device,
    print_progress: bool = True,
) -> None:
    """Render card as a video file."""
    encoder = make_encoder(hardware, quality, preset, vcodec, vaapi_device)
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
        [
            prepare_video(
                pipe_card_input(card_creator.width, card_creator.height, fps), encoder
            )
        ],
        card=card_creator,
        last_frame=min(last_frame, animation_start + 10),
        output_path=output_path,
        fps=fps,
        print_progress=print_progress,
        acodec=None,
        encoding=encoder,
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
    vcodec: str | None = Defaults.vcodec,
    acodec: str = Defaults.acodec,
    hardware: str = Defaults.hardware,
    quality: str = Defaults.quality,
    preset: str | None = Defaults.preset,
    vaapi_device: str | None = Defaults.vaapi_device,
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
    with tempfile.TemporaryDirectory(prefix="reaction-inputs-") as directory:
        inputs = Path(directory)
        webcam_source = stage_media(webcam_source, inputs, "webcam")

        screen_is_video = (
            not urlsplit(screen_source)
            .path.lower()
            .endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        )
        try:
            screen_source = stage_media(screen_source, inputs, "screen")
            if screen_is_video:
                get_metadata(screen_source)
        except Exception as exc:
            if not screen_is_video:
                raise
            logger.warning("Failed to prepare screen source: %s", exc)
            screen_source = f"{task}.png"
            screen_is_video = False

        metadata = get_metadata(webcam_source, expect_audio=True)
        fps = float(metadata.fps)
        last_frame = math.floor(metadata.duration * fps)
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
            width=card_position[2],
            height=card_position[3],
        )

        if hardware == "nvidia_full":
            if vaapi_device is not None:
                raise ValueError("--vaapi-device requires --hardware vaapi")
            # Use the branch's existing quality/preset options. Map them to NVENC.
            settings = settings_from_cli(
                quality=quality,
                preset=preset,
                vcodec=vcodec,
            )
            command = build_nvidia_full_command(
                webcam_source=webcam_source,
                screen_source=screen_source,
                screen_is_video=screen_is_video,
                background_source=background_source,
                action_audio_path=success_audio_path if success else fail_audio_path,
                output_path=output_path,
                fps=fps,
                duration=metadata.duration,
                animation_start=animation_start,
                has_webcam_audio=metadata.audio,
                sound=sound,
                background_position=background_position,
                webcam_position=webcam_position,
                screen_position=screen_position,
                card_position=card_position,
                acodec=acodec,
                print_progress=print_progress,
                settings=settings,
            )
            run_card_command(
                command=command,
                card=card_creator,
                last_frame=min(last_frame, animation_start + 6),
                output_path=output_path,
                print_progress=print_progress,
            )
            return

        encoder = make_encoder(hardware, quality, preset, vcodec, vaapi_device)
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

        video = prepare_video(video, encoder)

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
            acodec=acodec,
            encoding=encoder,
        )
