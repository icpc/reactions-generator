import os
import math
import requests
import atexit
from io import BytesIO
from fractions import Fraction
from typing import Annotated, Any, NamedTuple, cast

import typer
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

from PIL import Image, ImageColor
import ffmpeg  # type: ignore

from reactions_generator.defaults import Defaults
from reactions_generator.card import Card
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
    probe = cast(
        dict[str, str],
        next(
            (
                stream
                for stream in ffmpeg.probe(video_path)["streams"]
                if stream["codec_type"] == "video"
            ),
        ),
    )
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
    process = (  # type: ignore
        ffmpeg.output(
            *ffmpeg_input,
            output_path,
            vcodec=vcodec,
            acodec=acodec,
            r=fps,
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=not print_progress)
    )

    def clean_up():
        if process.poll() is None:
            process.terminate()
            process.wait()

    atexit.register(clean_up)

    for frame in range(last_frame + 1):
        process.stdin.write(to_ffmpeg_frame(card.render_frame(frame)))
    process.stdin.close()
    process.wait()


@app.command()
def render_card(
    title: Annotated[str, typer.Argument(Defaults.title)],
    subtitle: Annotated[str, typer.Argument(Defaults.subtitle)],
    hashtag: Annotated[str, typer.Argument(Defaults.hashtag)],
    task: Annotated[str, typer.Argument(Defaults.task)],
    time: Annotated[float, typer.Argument(Defaults.time)],
    outcome: Annotated[str, typer.Argument(Defaults.outcome)],
    success: Annotated[bool, typer.Argument(Defaults.success)],
    rank_before: Annotated[int, typer.Argument(Defaults.rank_before)],
    rank_after: Annotated[int, typer.Argument(Defaults.rank_after)],
    logo_source: Annotated[str, typer.Argument(Defaults.logo_source)],
    fps: Annotated[float, typer.Argument(Defaults.fps)],
    duration_seconds: Annotated[float, typer.Argument(Defaults.duration_seconds)],
    output_path: Annotated[str, typer.Argument(Defaults.output_path)],
    vcodec: Annotated[str, typer.Argument(Defaults.vcodec)],
    print_progress: Annotated[bool, typer.Argument(True)],
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
        last_frame=last_frame,
        output_path=output_path,
        fps=fps,
        print_progress=print_progress,
        vcodec=vcodec,
        acodec=None,
    )


@app.command()
def render_reaction(
    title: Annotated[str, typer.Argument(Defaults.title)],
    subtitle: Annotated[str, typer.Argument(Defaults.subtitle)],
    hashtag: Annotated[str, typer.Argument(Defaults.hashtag)],
    task: Annotated[str, typer.Argument(Defaults.task)],
    time: Annotated[float, typer.Argument(Defaults.time)],
    outcome: Annotated[str, typer.Argument(Defaults.outcome)],
    success: Annotated[bool, typer.Argument(Defaults.success)],
    rank_before: Annotated[int, typer.Argument(Defaults.rank_before)],
    rank_after: Annotated[int, typer.Argument(Defaults.rank_after)],
    logo_source: Annotated[str, typer.Argument(Defaults.logo_source)],
    webcam_source: Annotated[str, typer.Argument(Defaults.webcam_source)],
    screen_source: Annotated[str, typer.Argument(Defaults.screen_source)],
    background_source: Annotated[str, typer.Argument(Defaults.background_source)],
    success_audio_path: Annotated[str, typer.Argument(Defaults.success_audio_path)],
    fail_audio_path: Annotated[str, typer.Argument(Defaults.fail_audio_path)],
    output_path: Annotated[str, typer.Argument(Defaults.output_path)],
    vcodec: Annotated[str, typer.Argument(Defaults.vcodec)],
    acodec: Annotated[str, typer.Argument(Defaults.acodec)],
    print_progress: Annotated[bool, typer.Argument(True)],
):
    """Render reaction as a video file."""
    metadata = get_metadata(webcam_source)
    fps = float(metadata.fps)
    last_frame = math.floor(metadata.duration * fps)

    width = 1080
    height = 1920

    logo = load_image_or_color(logo_source, dimensions=(152, 152))
    webcam_full = ffmpeg.input(webcam_source)  # type: ignore
    webcam = webcam_full.video.filter(  # type: ignore
        "scale", w=Card.width, h="-1"
    )
    screen = ffmpeg.input(screen_source).video.filter(  # type: ignore
        "scale", w=Card.width, h="-1"
    )
    background = ffmpeg.input(background_source).filter("scale", w=width, h=height)  # type: ignore
    card = ffmpeg.input(  # type: ignore
        "pipe:",
        format="rawvideo",
        pix_fmt="rgba",
        s=f"{Card.width}x{Card.height}",
        r=fps,
    )

    animation_start = max(0, round(last_frame - 30 * fps))

    gap = 50

    card_position = center_anchor((0, 0, width, height), dimensions=Card.size)
    content_size = (Card.width, Card.width * 9 / 16)
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
def render_horizontal_reaction(
    title: Annotated[str, typer.Argument(Defaults.title)],
    subtitle: Annotated[str, typer.Argument(Defaults.subtitle)],
    hashtag: Annotated[str, typer.Argument(Defaults.hashtag)],
    task: Annotated[str, typer.Argument(Defaults.task)],
    time: Annotated[float, typer.Argument(Defaults.time)],
    outcome: Annotated[str, typer.Argument(Defaults.outcome)],
    success: Annotated[bool, typer.Argument(Defaults.success)],
    rank_before: Annotated[int, typer.Argument(Defaults.rank_before)],
    rank_after: Annotated[int, typer.Argument(Defaults.rank_after)],
    logo_source: Annotated[str, typer.Argument(Defaults.logo_source)],
    webcam_source: Annotated[str, typer.Argument(Defaults.webcam_source)],
    success_audio_path: Annotated[str, typer.Argument(Defaults.success_audio_path)],
    fail_audio_path: Annotated[str, typer.Argument(Defaults.fail_audio_path)],
    output_path: Annotated[str, typer.Argument(Defaults.output_path)],
    vcodec: Annotated[str, typer.Argument(Defaults.vcodec)],
    acodec: Annotated[str, typer.Argument(Defaults.acodec)],
    print_progress: Annotated[bool, typer.Argument(True)],
):
    """Render reaction as a video file."""
    metadata = get_metadata(webcam_source)
    fps = float(metadata.fps)
    last_frame = math.floor(metadata.duration * fps)

    width = 1920
    height = 1080

    logo = load_image_or_color(logo_source, dimensions=(152, 152))
    webcam_full = ffmpeg.input(webcam_source)  # type: ignore
    webcam = webcam_full.video.filter(  # type: ignore
        "scale", w=width, h="-1"
    )
    card = ffmpeg.input(  # type: ignore
        "pipe:",
        format="rawvideo",
        pix_fmt="rgba",
        s=f"{Card.width}x{Card.height}",
        r=fps,
    )

    animation_start = max(0, round(last_frame - 30 * fps))
    action_sound = (  # type: ignore
        ffmpeg.input(success_audio_path if success else fail_audio_path).filter(
            "adelay", delays=(animation_start / fps * 1000), all=1
        )
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video = (  # type: ignore
        webcam.overlay(card, x=width - Card.width, y=height - Card.height)
    )
    audio = ffmpeg.filter(  # type: ignore
        (webcam_full.audio, action_sound),  # type: ignore
        "amix",
        inputs=2,
        duration="longest",
    )

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


def apply_cds_auth(url: str, cds_auth: str | None) -> str:
    if cds_auth is None:
        return url
    return url.replace("http://", f"http://{cds_auth}@").replace(
        "https://", f"https://{cds_auth}@"
    )


@app.command()
def build_submission(
    url: str,
    id: str,
    cds_auth: Annotated[str | None, typer.Argument(None)],
    background_source: Annotated[str, typer.Argument(Defaults.background_source)],
    success_audio_path: Annotated[str, typer.Argument(Defaults.success_audio_path)],
    fail_audio_path: Annotated[str, typer.Argument(Defaults.fail_audio_path)],
    output_directory: Annotated[str, typer.Argument(Defaults.output_path)],
    vcodec: Annotated[str, typer.Argument(Defaults.vcodec)],
    acodec: Annotated[str, typer.Argument(Defaults.acodec)],
    vertical: Annotated[bool, typer.Argument(True)],
    print_progress: Annotated[bool, typer.Argument(True)],
    overwrite: Annotated[bool, typer.Argument(False)],
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
    title = data["team"]["displayName"]
    subtitle = data["team"]["customFields"]["clicsTeamFullName"]
    hashtag = data["team"]["hashTag"]
    task = data["problem"]["letter"]
    time = data["time"]
    outcome = data["result"]["verdict"]["shortName"]
    success = data["result"]["verdict"]["isAccepted"]
    rank_before = data["team"]["rankBefore"]
    rank_after = data["team"]["rankAfter"]
    logo_source = apply_cds_auth(data["team"]["organization"]["logo"]["url"], cds_auth)
    webcam_source = apply_cds_auth(data["reactionVideos"][0]["url"], cds_auth)
    screen_source = apply_cds_auth(data["reactionVideos"][1]["url"], cds_auth)
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
            success_audio_path=success_audio_path,
            fail_audio_path=fail_audio_path,
            output_path=output_path,
            print_progress=print_progress,
            vcodec=vcodec,
            acodec=acodec,
        )


@app.command()
def continuous_build_submission(
    url: str,
    cds_auth: Annotated[str | None, typer.Argument(None)],
    processes: Annotated[int | None, typer.Argument(os.cpu_count())],
    background_source: Annotated[str, typer.Argument(Defaults.background_source)],
    success_audio_path: Annotated[str, typer.Argument(Defaults.success_audio_path)],
    fail_audio_path: Annotated[str, typer.Argument(Defaults.fail_audio_path)],
    output_directory: Annotated[str, typer.Argument(Defaults.output_path)],
    vcodec: Annotated[str, typer.Argument(Defaults.vcodec)],
    acodec: Annotated[str, typer.Argument(Defaults.acodec)],
    vertical: Annotated[bool, typer.Argument(True)],
):
    """Render reaction as a video file."""

    def build_id(id: str):
        try:
            build_submission(
                id=id,
                url=url,
                cds_auth=cds_auth,
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
        except Exception as e:
            typer.echo(f"Failed to render submission {id}: {e}")

    while True:
        response = requests.get(f"{url}/api/overlay/runs")
        ids = [str(run["id"]) for run in response.json() if not run["isHidden"]]

        list(
            tqdm(
                Parallel(return_as="generator", n_jobs=processes)(
                    delayed(build_id)(id) for id in ids
                ),
                desc="Rendering submissions",
                total=len(ids),
                leave=False,
            )
        )


def main():
    app()


if __name__ == "__main__":
    main()
