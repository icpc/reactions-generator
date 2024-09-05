import os
import math
import typer
import numpy as np

from tqdm import tqdm
from PIL import Image
from moviepy.editor import ImageSequenceClip, VideoFileClip  # type: ignore

from reactions_generator.colors import Colors
from reactions_generator.card import Card, target_card_width
from reactions_generator.reaction import Reaction

app = typer.Typer(no_args_is_help=True)


def process_video(frames: list[Image.Image], fps: float, output_path: str):
    """Assemble video from frames."""
    raw_frames = [np.array(frame) for frame in frames]
    clip = ImageSequenceClip(raw_frames, fps=fps)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clip.write_videofile(output_path, codec="libx264", fps=fps)


def load_remote_image(path: str) -> Image.Image:
    return Image.open(path)
    # return Image.open(BytesIO(requests.get(url).content))


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
    logo_path: str = "example/logo.png",
    fps: float = 30,
    duration_seconds: float = 60,
    output_path: str = "out/output.mp4",
):
    """Render card as a video file."""
    last_frame = math.floor(duration_seconds * fps)
    animation_start = max(0, round(last_frame - 30 * fps))
    logo = load_remote_image(logo_path)
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
        for frame in tqdm(range(last_frame + 1), desc="Rendering frames")
    ]
    process_video(frames, fps=fps, output_path=output_path)


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
    logo_path: str = "example/logo.png",
    header_path: str = "example/header.png",
    webcam_path: str = "example/reaction.mp4",
    screen_path: str = "example/screen.mp4",
    output_path: str = "out/output.mp4",
):
    """Render reaction as a video file."""
    background = Image.new("RGB", (1080, 1920), Colors.gray)
    logo = load_remote_image(logo_path)
    header = load_remote_image(header_path)
    webcam = VideoFileClip(
        webcam_path,
        target_resolution=(None, target_card_width),
        resize_algorithm="fast_bilinear",
    )
    screen = VideoFileClip(
        screen_path,
        target_resolution=(None, target_card_width),
        resize_algorithm="fast_bilinear",
    )
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
        for frame in tqdm(range(last_frame + 1), desc="Rendering frames")
    ]
    webcam.close()
    screen.close()
    process_video(frames, fps=fps, output_path=output_path)


def run_render_card():
    typer.run(render_card)


def run_render_reaction():
    typer.run(render_reaction)


def main():
    app()


if __name__ == "__main__":
    main()
