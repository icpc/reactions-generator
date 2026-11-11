"""GPU-resident FFmpeg CUDA composition for reaction videos.

Webcam and video-screen frames stay on CUDA after hardware decoding; the
background and Pillow-generated RGBA card are uploaded only at their input
boundaries. The resulting video is encoded without hwdownload.
"""

from __future__ import annotations

import math
import subprocess

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
    if not math.isfinite(value) or value < 0:
        raise ValueError("Video offsets must be finite and nonnegative")
    return int(value) // 2 * 2


def _position(
    rectangle: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x, y, width, height = rectangle
    return (
        _even_offset(x),
        _even_offset(y),
        _even_dimension(width),
        _even_dimension(height),
    )


def build_nvidia_full_command(
    *,
    webcam_source: str,
    screen_source: str,
    screen_is_video: bool,
    background_source: str,
    action_audio_path: str,
    output_path: str,
    fps: float,
    duration: float,
    animation_start: int,
    has_webcam_audio: bool,
    sound: bool,
    background_position: tuple[float, float, float, float],
    webcam_position: tuple[float, float, float, float],
    screen_position: tuple[float, float, float, float],
    card_position: tuple[int, int, int, int],
    acodec: str,
    print_progress: bool,
    encoding: EncoderConfig,
) -> list[str]:
    """Return a CUDA-only video graph with a small software-input boundary.

    Input indexes: 0 webcam, 1 screen, 2 static background, 3 RGBA card pipe,
    4 verdict sound. Background and still-screen frames are scaled and uploaded
    once, then repeated as GPU frame references with fresh timestamps.
    """
    if encoding.hardware != "nvidia_full":
        raise ValueError("CUDA composition requires the nvidia_full encoder profile")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be finite and positive")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be finite and positive")
    if animation_start < 0:
        raise ValueError("Animation start must be nonnegative")

    bx, by, bw, bh = _position(background_position)
    wx, wy, ww, wh = _position(webcam_position)
    sx, sy, sw, sh = _position(screen_position)
    cx, cy, cw, ch = _position(
        (
            float(card_position[0]),
            float(card_position[1]),
            float(card_position[2]),
            float(card_position[3]),
        )
    )
    if bx != 0 or by != 0:
        raise ValueError("The NVIDIA renderer requires a background at (0, 0)")
    if (cw, ch) != (card_position[2], card_position[3]):
        raise ValueError("RGBA card dimensions must be even")
    for x, y, w, h in ((wx, wy, ww, wh), (sx, sy, sw, sh), (cx, cy, cw, ch)):
        if x + w > bw or y + h > bh:
            raise ValueError("An overlay lies outside the NVIDIA canvas")

    rate = f"{fps:.12g}"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-loglevel",
        "info" if print_progress else "error",
        *encoding.device_args(),
        "-hwaccel",
        "cuda",
        "-hwaccel_device",
        "gpu",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        webcam_source,
    ]
    if screen_is_video:
        command.extend(
            [
                "-hwaccel",
                "cuda",
                "-hwaccel_device",
                "gpu",
                "-hwaccel_output_format",
                "cuda",
            ]
        )
    command.extend(["-i", screen_source, "-i", background_source])
    command.extend(
        [
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            f"{cw}x{ch}",
            "-framerate",
            rate,
            "-i",
            "pipe:0",
            "-i",
            action_audio_path,
        ]
    )

    # A single GPU upload of the background is looped with new presentation
    # timestamps *before* overlaying the moving webcam. This prevents the
    # frozen-video failure caused by a single-frame canvas.
    filters = [
        (
            f"[2:v]scale={bw}:{bh}:flags=bicubic,format=yuv420p,"
            f"hwupload,loop=loop=-1:size=1:start=0,"
            f"settb=AVTB,setpts=N/({rate}*TB)[base]"
        ),
        (f"[0:v]scale_cuda=w={ww}:h={wh}:format=yuv420p,setpts=PTS-STARTPTS[webcam]"),
    ]
    if screen_is_video:
        filters.append(
            f"[1:v]scale_cuda=w={sw}:h={sh}:format=yuv420p,setpts=PTS-STARTPTS[screen]"
        )
    else:
        filters.append(
            f"[1:v]scale={sw}:{sh}:flags=bicubic,format=yuv420p,"
            f"hwupload,loop=loop=-1:size=1:start=0,"
            f"settb=AVTB,setpts=N/({rate}*TB)[screen]"
        )
    filters.extend(
        [
            "[3:v]format=yuva420p,hwupload,setpts=PTS-STARTPTS[card]",
            (
                f"[base][webcam]overlay_cuda=x={wx}:y={wy}:"
                "shortest=0:eof_action=repeat[v1]"
            ),
            f"[v1][screen]overlay_cuda=x={sx}:y={sy}:shortest=0:eof_action=repeat[v2]",
            f"[v2][card]overlay_cuda=x={cx}:y={cy}:shortest=0:eof_action=repeat[vout]",
        ]
    )

    delay_ms = round(animation_start / fps * 1000)
    filters.append(f"[4:a]adelay=delays={delay_ms}:all=1[effect]")
    if has_webcam_audio and sound:
        filters.append("[0:a][effect]amix=inputs=2:duration=longest[aout]")
        filters.append("[aout]anull[audio]")
    else:
        filters.append("[effect]anull[audio]")

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "[audio]",
            "-c:v",
            encoding.codec,
            *[
                argument
                for key, value in encoding.output_options(acodec).items()
                for argument in (f"-{key}", str(value))
            ],
            "-pix_fmt",
            "cuda",  # Avoid a CPU-frame conversion before NVENC.
            "-r",
            rate,
            "-c:a",
            acodec,
            "-t",
            f"{duration:.9f}",
            output_path,
        ]
    )
    return command


def probe_nvidia_full(
    source: str, encoding: EncoderConfig, timeout: float = 30.0
) -> tuple[bool, str]:
    """Check filter support, then exercise CUDA decoding and NVENC encoding."""
    if encoding.hardware != "nvidia_full":
        raise ValueError("CUDA composition requires the nvidia_full encoder profile")
    try:
        for option, required in (
            ("-filters", {"scale_cuda", "overlay_cuda", "hwupload"}),
            ("-encoders", {"h264_nvenc"}),
        ):
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", option],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode:
                return False, result.stderr.strip()[-800:]
            missing = required - set(result.stdout.split())
            if missing:
                return False, f"FFmpeg is missing: {', '.join(sorted(missing))}"
        help_result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "filter=scale_cuda"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if help_result.returncode or "format" not in help_result.stdout.split():
            return False, (
                "This FFmpeg build lacks scale_cuda's format option, required for "
                "YUV420P composition with the transparent card. Upgrade to a CUDA-enabled "
                "FFmpeg build whose `ffmpeg -h filter=scale_cuda` lists format."
            )
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostdin",
                "-loglevel",
                "error",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                source,
                "-vf",
                "scale_cuda=128:128:format=yuv420p",
                "-frames:v",
                "8",
                "-an",
                "-c:v",
                encoding.codec,
                *[
                    argument
                    for key, value in encoding.options.items()
                    for argument in (f"-{key}", str(value))
                ],
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode:
        return False, (result.stderr.strip() or f"FFmpeg exited {result.returncode}")[
            -800:
        ]
    return True, "CUDA decode and NVENC encode succeeded"
