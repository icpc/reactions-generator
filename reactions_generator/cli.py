import os
import math
import typing
import requests
import atexit
import hashlib
from io import BytesIO
from fractions import Fraction
from typing import Any, NamedTuple

import typer
import numpy as np
from tqdm import tqdm

from PIL import Image, ImageColor
import ffmpeg  # type: ignore

from reactions_generator.defaults import Defaults
from reactions_generator.card import Card
import tempfile
from reactions_generator.utils import (
    center_anchor,
    place_above,
    place_below,
)

app = typer.Typer(no_args_is_help=True)


def load_image_or_color(source: str, dimensions: tuple[int, int]) -> Image.Image:
    if source.startswith("#"):
        return Image.new("RGBA", dimensions, color=ImageColor.getrgb(source))

    if source.lower().startswith(("http://", "https://")):
        response = requests.get(source)
        if response.status_code != 200:
            raise ValueError(f"Failed to download image from {source}")
        return Image.open(BytesIO(response.content))

    return Image.open(source)


def load_video(source: str, target_width: int):  # type: ignore
    return ffmpeg.input(source).filter(f"scale={target_width}:-1")  # type: ignore


def to_ffmpeg_frame(image: Image.Image):
    return np.asarray(image, np.uint8).tobytes()


class Metadata(NamedTuple):
    fps: Fraction
    duration: float


def get_metadata(video_path: str) -> Metadata:
    file = tempfile.mktemp()
    if video_path.startswith(("http://", "https://")):
        response = requests.get(video_path)
        if response.status_code != 200:
            raise ValueError(f"Failed to download video from {video_path}")
        with open(file, "wb") as tmp_file:
            tmp_file.write(response.content)
        video_path = file

    probe = typing.cast(
        dict[str, str],
        next(
            (
                stream
                for stream in ffmpeg.probe(video_path)["streams"]
                if stream["codec_type"] == "video"
            ),
        ),
    )
    if video_path == file:
        os.remove(file)
    return Metadata(
        fps=Fraction(probe["avg_frame_rate"]),
        duration=float(probe["duration"]),
    )


def render(
    ffmpeg_input: list[Any],
    card: Card,
    last_frame: int,
    output_path: str,
    fps: float,
    print_progress: bool,
    vcodec: str,
    acodec: str | None,
):
    output_basename = os.path.basename(output_path)
    output_dirname = os.path.dirname(output_path)
    tmp_output = os.path.join(output_dirname, f"tmp_{output_basename}")
    os.makedirs(output_dirname, exist_ok=True)
    process = (  # type: ignore
        ffmpeg.output(
            *ffmpeg_input,
            tmp_output,
            vcodec=vcodec,
            acodec=acodec,
            r=fps,
            pix_fmt="yuv420p",
            loglevel="info" if print_progress else "quiet",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    def clean_up():
        if process.poll() is None:
            process.terminate()
            process.wait()

    atexit.register(clean_up)
    try:
        for frame in range(last_frame + 1):
            process.stdin.write(to_ffmpeg_frame(card.render_frame(frame)))
        process.stdin.close()
        process.wait()
        os.replace(tmp_output, output_path)
    except Exception as e:
        clean_up()
        raise e


@app.command()
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
):
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

    render(  # type: ignore
        [
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgba",
                s=f"{Card.width}x{Card.height}",
                r=fps,
            )
        ],
        card=card_creator,
        last_frame=min(last_frame, animation_start + 10),
        output_path=output_path,
        fps=fps,
        print_progress=print_progress,
        vcodec=vcodec,
        acodec=None,
    )


@app.command()
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
):
    """Render reaction as a video file."""
    metadata = get_metadata(webcam_source)
    fps = float(metadata.fps)
    last_frame = math.floor(metadata.duration * fps)
    animation_start = max(0, round(last_frame - 30 * fps))

    try:
        get_metadata(screen_source)
    except Exception as e:
        typer.echo(f"Failed to get metadata for screen source: {e}")
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
        width=1000,
        height=306,
    )

    width = 1080
    height = 1920

    webcam_full = ffmpeg.input(webcam_source)  # type: ignore
    webcam = webcam_full.video.filter(  # type: ignore
        "scale", w=card_creator.width, h="-1"
    ).filter("setpts", "PTS-STARTPTS")
    screen = ffmpeg.input(screen_source).video.filter(  # type: ignore
        "scale", w=card_creator.width, h="-1"
    ).filter("setpts", "PTS-STARTPTS")
    background = ffmpeg.input(background_source).filter("scale", w=width, h=height)  # type: ignore
    card = ffmpeg.input(  # type: ignore
        "pipe:",
        format="rawvideo",
        pix_fmt="rgba",
        s=f"{card_creator.width}x{card_creator.height}",
        r=fps,
    )

    gap = 50
    card_position = center_anchor((0, 0, width, height), dimensions=card_creator.size)
    content_size = (card_creator.width, card_creator.width * 9 / 16)
    webcam_position = place_above(
        card_position,
        dimensins=content_size,
        gap=gap,
    )
    screen_position = place_below(
        card_position,
        dimensins=content_size,
        gap=gap,
    )

    action_sound = (  # type: ignore
        ffmpeg.input(success_audio_path if success else fail_audio_path).filter(
            "adelay", delays=animation_start / fps * 1000, all=1
        )
    )
    video = (  # type: ignore
        background.overlay(card, x=card_position[0], y=card_position[1])
        .overlay(webcam, x=webcam_position[0], y=webcam_position[1])
        .overlay(screen, x=screen_position[0], y=screen_position[1])
    )
    audio = ffmpeg.filter(  # type: ignore
        (webcam_full.audio, action_sound),  # type: ignore
        "amix",
        inputs=2,
        duration="longest",
    )

    render(
        [audio, video],
        card=card_creator,
        last_frame=min(last_frame, animation_start + 10),
        output_path=output_path,
        fps=fps,
        print_progress=print_progress,
        vcodec=vcodec,
        acodec=acodec,
    )


@app.command()
def render_horizontal_reaction(
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
    success_audio_path: str = Defaults.success_audio_path,
    fail_audio_path: str = Defaults.fail_audio_path,
    output_path: str = Defaults.output_path,
    vcodec: str = Defaults.vcodec,
    acodec: str = Defaults.acodec,
    print_progress: bool = True,
):
    """Render reaction as a video file."""
    metadata = get_metadata(webcam_source)
    fps = float(metadata.fps)
    last_frame = math.floor(metadata.duration * fps)
    animation_start = max(0, round(last_frame - 30 * fps))

    try:
        get_metadata(screen_source)
    except Exception as e:
        typer.echo(f"Failed to get metadata for screen source: {e}")
        screen_source = f"{task}.png"

    width = 1920
    height = 1080

    screen_width = 640
    screen_height = screen_width * 9 // 16
    margin = 16

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
        height=300,
    )

    webcam_full = ffmpeg.input(webcam_source)  # type: ignore
    webcam = webcam_full.video.filter(  # type: ignore
        "scale", w=width, h=height
    ).filter("setpts", "PTS-STARTPTS")
    screen = ffmpeg.input(screen_source).video.filter(  # type: ignore
        "scale", w=screen_width, h="-1"
    ).filter("setpts", "PTS-STARTPTS")
    card = ffmpeg.input(  # type: ignore
        "pipe:",
        format="rawvideo",
        pix_fmt="rgba",
        s=f"{card_creator.width}x{card_creator.height}",
        r=fps,
    )

    action_sound = (  # type: ignore
        ffmpeg.input(success_audio_path if success else fail_audio_path).filter(
            "adelay", delays=animation_start / fps * 1000, all=1
        )
    )

    video = (  # type: ignore
        webcam.overlay(
            screen,
            x=margin,
            y=height - screen_height - margin,
        ).overlay(
            card,
            x=width - card_creator.width - margin,
            y=height - card_creator.height - margin,
        )
    )
    audio = ffmpeg.filter(  # type: ignore
        (webcam_full.audio, action_sound),  # type: ignore
        "amix",
        inputs=2,
        duration="longest",
    )

    render(
        [audio, video],
        card=card_creator,
        last_frame=last_frame,
        output_path=output_path,
        fps=fps,
        print_progress=print_progress,
        vcodec=vcodec,
        acodec=acodec,
    )


@app.command()
def build_submission(
    url: str,
    id: str,
    background_source: str = Defaults.background_source,
    success_audio_path: str = Defaults.success_audio_path,
    fail_audio_path: str = Defaults.fail_audio_path,
    output_directory: str = Defaults.output_directory,
    vcodec: str = Defaults.vcodec,
    acodec: str = Defaults.acodec,
    vertical: bool = True,
    print_progress: bool = True,
    overwrite: bool = False,
):
    """Render reaction as a video file."""
    output_path = os.path.join(output_directory, f"{id}.mp4")
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            if print_progress:
                typer.echo(
                    f"File {output_path} already exists. Use --overwrite to replace."
                )
            return
    response = requests.get(f"{url}/api/overlay/externalRun/{id}")
    data = response.json()
    title = data["team"].get("displayName", "")
    subtitle = data["team"]["customFields"].get("clicsTeamFullName", "")
    hashtag = data["team"].get("hashTag")
    if hashtag is None:
        hashtag = ""
    else:
        if not hashtag.startswith("#"):
            hashtag = f"#{hashtag}"
    task = data["problem"]["letter"]
    time = data["time"]
    outcome = data["result"]["verdict"]["shortName"]
    success = data["result"]["verdict"]["isAccepted"]
    rank_before = data["team"]["rankBefore"]
    rank_after = data["team"]["rankAfter"]
    logo_source = data["team"]["organization"]["logo"]["url"]
    webcam_source = data["reactionVideos"][0]["url"]
    screen_source = data["reactionVideos"][1]["url"]
    if vertical:
        render_reaction(
            title=title,
            subtitle=subtitle,
            hashtag=hashtag,
            task=task,
            time=time,
            outcome=outcome,
            success=success,
            rank_before=rank_before,
            rank_after=rank_after,
            logo_source=logo_source,
            webcam_source=webcam_source,
            screen_source=screen_source,
            background_source=background_source,
            success_audio_path=success_audio_path,
            fail_audio_path=fail_audio_path,
            output_path=output_path,
            print_progress=print_progress,
            vcodec=vcodec,
            acodec=acodec,
        )
    else:
        render_horizontal_reaction(
            title=title,
            subtitle=subtitle,
            hashtag=hashtag,
            task=task,
            time=time,
            outcome=outcome,
            success=success,
            rank_before=rank_before,
            rank_after=rank_after,
            logo_source=logo_source,
            webcam_source=webcam_source,
            screen_source=screen_source,
            success_audio_path=success_audio_path,
            fail_audio_path=fail_audio_path,
            output_path=output_path,
            print_progress=print_progress,
            vcodec=vcodec,
            acodec=acodec,
        )


def log_error(error_string: str, id: str, output_directory: str):
    typer.echo(f"Error in {id}: {error_string}", err=True)
    os.makedirs(output_directory, exist_ok=True)
    with open(f"{output_directory}/{id}.err", "w") as f:
        f.write(error_string)


def stable_hash(input_string: str):
    hash_object = hashlib.sha256()
    hash_object.update(input_string.encode("utf-8"))
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int


@app.command()
def continuous_build_submission(
    url: str,
    background_source: str = Defaults.background_source,
    success_audio_path: str = Defaults.success_audio_path,
    fail_audio_path: str = Defaults.fail_audio_path,
    output_directory: str = Defaults.output_directory,
    vcodec: str = Defaults.vcodec,
    acodec: str = Defaults.acodec,
    vertical: bool = True,
    total_workers: int = 1,
    worker_id: int = 0,
):
    """Render reaction as a video file."""
    os.makedirs(output_directory, exist_ok=True)

    while True:
        response = requests.get(f"{url}/api/overlay/runs")
        ids = [str(run["id"]) for run in response.json() if not run["isHidden"]]

        filtered = [id for id in ids if stable_hash(id) % total_workers == worker_id]

        for id in tqdm(filtered, desc="Rendering submissions"):
            try:
                build_submission(
                    id=id,
                    url=url,
                    background_source=background_source,
                    success_audio_path=success_audio_path,
                    fail_audio_path=fail_audio_path,
                    output_directory=output_directory,
                    overwrite=False,
                    print_progress=False,
                    vcodec=vcodec,
                    acodec=acodec,
                    vertical=vertical,
                )
            except ffmpeg.Error as e:
                log_error(
                    os.linesep.join([str(x) for x in [e, e.stdout, e.stderr]]),
                    id=id,
                    output_directory=output_directory,
                )
            except Exception as e:
                log_error(str(e), id=id, output_directory=output_directory)


def main():
    app()


if __name__ == "__main__":
    main()
