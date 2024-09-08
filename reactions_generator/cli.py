import os
import math
import typer
import requests
import numpy as np
from io import BytesIO
from random import randint

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
    audio: AudioClip | None = None,
):
    """Assemble video from frames."""
    raw_frames = [np.asarray(frame) for frame in tqdm(frames, desc="Converting frames")]
    clip = ImageSequenceClip(raw_frames, fps=fps)
    if audio:
        clip = clip.set_audio(audio)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clip.write_videofile(
        output_path,
        codec="libx264",
        fps=fps,
        temp_audiofile=f"temp_audio_{randint(0, 100000000000)}.mp3",
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
        for frame in tqdm(range(last_frame + 1), desc="Rendering frames")
    ]
    process_video(frames, fps=fps, output_path=output_path)


def run_render_card():
    typer.run(render_card)


def audio_or_silence(clip: VideoFileClip) -> AudioClip:
    if clip.audio is None:
        duration: float = clip.duration  # type: ignore
        return AudioClip(make_frame=lambda: 0, duration=duration, fps=44100)
    return clip.audio


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
        for frame in tqdm(range(last_frame + 1), desc="Rendering frames")
    ]

    submission_audio = (
        AudioFileClip(success_audio_path) if success else AudioFileClip(fail_audio_path)
    ).set_start((animation_start - 17) / fps)
    audio = (
        CompositeAudioClip([webcam.audio, submission_audio])
        if webcam.audio
        else submission_audio
    )

    screen.close()
    process_video(frames, fps=fps, output_path=output_path, audio=audio)
    webcam.close()


def run_render_reaction():
    typer.run(render_reaction)


def main():
    app()


if __name__ == "__main__":
    main()
