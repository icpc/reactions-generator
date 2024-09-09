import os
import math
import typer
import requests
import numpy as np
from io import BytesIO
from random import randint
from joblib import Parallel, delayed

from tqdm import tqdm
from PIL import Image, ImageColor
from moviepy.video.VideoClip import VideoClip, ImageClip  # type: ignore
from moviepy.video.io.VideoFileClip import VideoFileClip  # type: ignore
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore
from moviepy.audio.AudioClip import AudioClip, CompositeAudioClip  # type: ignore
from moviepy.audio.io.AudioFileClip import AudioFileClip  # type: ignore

from reactions_generator.card import Card, target_card_width
from reactions_generator.reaction import Reaction

app = typer.Typer(no_args_is_help=True)


def process_video(
    frames: list[Image.Image],
    fps: float,
    output_path: str,
    print_progress: bool,
    audio: AudioClip | None = None,
):
    """Assemble video from frames."""
    raw_frames = [
        np.asarray(frame)
        for frame in tqdm(frames, desc="Converting frames", disable=not print_progress)
    ]
    clip = ImageSequenceClip(raw_frames, fps=fps)
    if audio:
        clip = clip.set_audio(audio)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs("tmp", exist_ok=True)
    clip.write_videofile(
        output_path,
        codec="libx264",
        fps=fps,
        temp_audiofile=f"tmp/temp_audio_{randint(0, 100000000000)}.mp3",
        logger="bar" if print_progress else None,
    )


def load_image_or_color(source: str, dimensions: tuple[int, int]) -> Image.Image:
    if source.startswith("#"):
        return Image.new("RGBA", dimensions, color=ImageColor.getrgb(source))

    if source.lower().startswith(("http://", "https://")):
        response = requests.get(source)
        if response.status_code != 200:
            raise ValueError(f"Failed to download image from {source}")
        return Image.open(BytesIO(response.content))

    return Image.open(source)


def load_video(source: str, target_width: int, audio: bool) -> VideoFileClip:
    return VideoFileClip(
        source,
        target_resolution=(None, target_width),
        resize_algorithm="fast_bilinear",
        audio=audio,
    )


def load_image_or_video(source: str, target_width: int) -> VideoClip:
    if source.lower().endswith((".png", ".jpg", ".jpeg")):
        image = load_image_or_color(source, (1920, 1080)).convert("RGB")
        image.thumbnail((target_width, target_width))
        return ImageClip(np.array(image))
    return load_video(source, target_width, audio=False)


@app.command()
def render_card(
    title: str = "UNI",
    subtitle: str = "Subtitle",
    hashtag: str = "#hashtag",
    task: str = "A",
    time: float = 1000000,
    outcome: str = "AC",
    success: bool = True,
    rank_before: int = 100,
    rank_after: int = 1,
    logo_source: str = "example/logo.png",
    fps: float = 30,
    duration_seconds: float = 60,
    output_path: str = "out/output.mp4",
    print_progress: bool = True,
):
    """Render card as a video file."""
    last_frame = math.floor(duration_seconds * fps)
    animation_start = max(0, round(last_frame - 30 * fps))
    logo = load_image_or_color(logo_source, dimensions=(152, 152))
    card = Card(
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
    frames = [
        card.render_frame(frame).convert("RGB")
        for frame in tqdm(
            range(last_frame + 1), desc="Rendering frames", disable=not print_progress
        )
    ]
    process_video(
        frames, fps=fps, output_path=output_path, print_progress=print_progress
    )


@app.command()
def render_reaction(
    title: str = "UNI",
    subtitle: str = "Subtitle",
    hashtag: str = "#hashtag",
    task: str = "A",
    time: float = 1000000,
    outcome: str = "AC",
    success: bool = True,
    rank_before: int = 100,
    rank_after: int = 1,
    logo_source: str = "example/logo.png",
    webcam_source: str = "example/reaction.mp4",
    screen_source: str = "example/screen.mp4",
    header_source: str = "example/header.png",
    background_source: str = "#1F1F1F",
    success_audio_path: str = "example/success.mp3",
    fail_audio_path: str = "example/fail.mp3",
    output_path: str = "out/output.mp4",
    print_progress: bool = True,
):
    """Render reaction as a video file."""
    logo = load_image_or_color(logo_source, dimensions=(152, 152))
    header = load_image_or_color(header_source, dimensions=(target_card_width, 40))
    background = load_image_or_color(
        background_source, dimensions=(1080, 1920)
    ).convert("RGB")
    screen = load_image_or_video(screen_source, target_width=target_card_width)
    webcam = load_video(webcam_source, target_width=target_card_width, audio=True)

    fps = float(webcam.fps)  # type: ignore
    last_frame = math.floor(float(webcam.duration) * fps)  # type: ignore
    animation_start = max(0, round(last_frame - 30 * fps))
    card = Card(
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
    reaction = Reaction(
        header=header,
        success=success,
        animation_start=animation_start,
        fps=fps,
    )

    frames = [
        reaction.render_frame(
            frame,
            background=background,
            card=card.render_frame(frame),
            webcam=Image.fromarray(webcam.get_frame(frame / fps)),
            screen=Image.fromarray(screen.get_frame(frame / fps)),
        )
        for frame in tqdm(
            range(last_frame + 1), desc="Rendering frames", disable=not print_progress
        )
    ]

    submission_audio = (
        AudioFileClip(success_audio_path) if success else AudioFileClip(fail_audio_path)
    ).set_start((animation_start - 17) / fps)
    audio = (
        CompositeAudioClip([webcam.audio, submission_audio])
        if webcam.audio
        else submission_audio
    )

    process_video(
        frames,
        fps=fps,
        output_path=output_path,
        print_progress=print_progress,
        audio=audio,
    )
    screen.close()
    webcam.close()


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
    cds_auth: str | None = None,
    header_source: str = "example/header.png",
    background_source: str = "#1F1F1F",
    success_audio_path: str = "example/success.mp3",
    fail_audio_path: str = "example/fail.mp3",
    output_directory: str = "out",
    overwrite: bool = False,
    print_progress: bool = True,
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
    render_reaction(
        title=data["team"]["displayName"],
        subtitle=data["team"]["customFields"]["clicsTeamFullName"],
        hashtag=data["team"]["hashTag"],
        task=data["problem"]["letter"],
        time=data["time"],
        outcome=data["result"]["verdict"]["shortName"],
        success=data["result"]["verdict"]["isAccepted"],
        rank_before=data["team"]["rankBefore"],
        rank_after=data["team"]["rankAfter"],
        logo_source=apply_cds_auth(
            data["team"]["organization"]["logo"]["url"], cds_auth
        ),
        webcam_source=apply_cds_auth(data["reactionVideos"][0]["url"], cds_auth),
        screen_source=apply_cds_auth(data["reactionVideos"][1]["url"], cds_auth),
        header_source=header_source,
        background_source=background_source,
        success_audio_path=success_audio_path,
        fail_audio_path=fail_audio_path,
        output_path=output_path,
        print_progress=print_progress,
    )


@app.command()
def continuous_build_submission(
    url: str,
    cds_auth: str | None = None,
    processes: int | None = os.cpu_count(),
    header_source: str = "example/header.png",
    background_source: str = "#1F1F1F",
    success_audio_path: str = "example/success.mp3",
    fail_audio_path: str = "example/fail.mp3",
    output_directory: str = "out",
):
    """Render reaction as a video file."""

    def build_id(id: str):
        try:
            build_submission(
                id=id,
                url=url,
                cds_auth=cds_auth,
                header_source=header_source,
                background_source=background_source,
                success_audio_path=success_audio_path,
                fail_audio_path=fail_audio_path,
                output_directory=output_directory,
                overwrite=False,
                print_progress=False,
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
