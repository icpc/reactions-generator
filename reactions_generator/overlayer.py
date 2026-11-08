"""Overlayer API data and rendering of individual submissions."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlsplit

import requests

from reactions_generator.defaults import Defaults
from reactions_generator.rendering import render_reaction


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Scheduling metadata from live-v3's /reactions/runs.json."""

    id: str
    verdict: str = ""
    is_accepted: bool = False
    is_first_to_solve: bool = False

    @property
    def priority(self) -> int:
        if self.is_first_to_solve:
            return 0
        return 1 if self.is_accepted else 2

    @property
    def is_pending(self) -> bool:
        return self.verdict == "IN_PROGRESS"


@dataclass(frozen=True, slots=True)
class Submission:
    """Validated overlayer data needed to render a reaction."""

    title: str
    subtitle: str
    hashtag: str
    task: str
    time: float
    outcome: str
    success: bool
    rank_before: int
    rank_after: int
    logo_source: str
    webcam_source: str
    screen_source: str


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return cast(list[object], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def parse_submission(payload: object) -> Submission:
    data = _object(payload, "submission")
    team = _object(data.get("team"), "team")
    custom_fields = _object(team.get("customFields", {}), "team.customFields")
    problem = _object(data.get("problem"), "problem")
    result = _object(data.get("result"), "result")
    organization = _object(team.get("organization"), "team.organization")
    logo = _object(organization.get("logo"), "team.organization.logo")
    raw_hashtag = team.get("hashTag")
    hashtag = "" if raw_hashtag is None else _text(raw_hashtag, "team.hashTag")
    if hashtag and not hashtag.startswith("#"):
        hashtag = f"#{hashtag}"

    if "verdict" in result:
        verdict = _object(result["verdict"], "result.verdict")
        outcome = _text(verdict.get("shortName"), "verdict.shortName")
        success = _boolean(verdict.get("isAccepted"), "verdict.isAccepted")
    else:
        outcome = f"+{_number(result.get('difference'), 'result.difference')}"
        success = _boolean(
            result.get("isFirstBestTeamRun"), "result.isFirstBestTeamRun"
        )

    webcam_source = ""
    screen_source = ""
    for source in _array(data.get("reactionVideos"), "reactionVideos"):
        video = _object(source, "reaction video")
        url = _text(video.get("url"), "reaction video.url")
        uri = urlsplit(url).path.rsplit(".", 1)[0]
        if uri.endswith("webcam"):
            webcam_source = url
        elif uri.endswith("desktop"):
            screen_source = url
    if not webcam_source:
        raise ValueError("Submission has no webcam reaction video")

    return Submission(
        title=_text(team.get("displayName", ""), "team.displayName"),
        subtitle=_text(custom_fields.get("clicsTeamFullName", ""), "team full name"),
        hashtag=hashtag,
        task=_text(problem.get("letter"), "problem.letter"),
        time=_number(data.get("time"), "time"),
        outcome=outcome,
        success=success,
        rank_before=_integer(team.get("rankBefore"), "team.rankBefore"),
        rank_after=_integer(team.get("rankAfter"), "team.rankAfter"),
        logo_source=_text(logo.get("url"), "team.organization.logo.url"),
        webcam_source=webcam_source,
        screen_source=screen_source,
    )


def fetch_submission(url: str, id: str) -> Submission:
    response = requests.get(f"{url.rstrip('/')}/reactions/fullRuns/{id}", timeout=30)
    response.raise_for_status()
    return parse_submission(response.json())


def parse_run_summary(payload: object) -> RunSummary:
    run = _object(payload, "run")
    id = run.get("id")
    if isinstance(id, bool) or not isinstance(id, (str, int)):
        raise TypeError("run.id must be a string or integer")
    return RunSummary(
        id=str(id),
        verdict=_text(run.get("verdict", ""), "run.verdict"),
        is_accepted=_boolean(run.get("isAccepted", False), "run.isAccepted"),
        # live-v3 omits this field when false (EncodeDefault.Mode.NEVER).
        is_first_to_solve=_boolean(
            run.get("isFirstToSolve", False), "run.isFirstToSolve"
        ),
    )


def fetch_runs(url: str) -> list[RunSummary]:
    response = requests.get(f"{url.rstrip('/')}/reactions/runs.json", timeout=30)
    response.raise_for_status()
    return [parse_run_summary(run) for run in _array(response.json(), "runs")]


def fetch_submission_ids(url: str) -> list[str]:
    return [run.id for run in fetch_runs(url)]


def build_submission(
    url: str,
    id: str,
    success_audio_path: str = Defaults.success_audio_path,
    fail_audio_path: str = Defaults.fail_audio_path,
    output_directory: str = Defaults.output_directory,
    vcodec: str = Defaults.vcodec,
    acodec: str = Defaults.acodec,
    vertical: bool = True,
    print_progress: bool = True,
    overwrite: bool = False,
    sound: bool = Defaults.sound,
    hwaccel: str | None = Defaults.hwaccel,
) -> None:
    """Render reaction as a video file."""
    output_path = os.path.join(output_directory, f"{id}.mp4")
    if os.path.exists(output_path) and not overwrite:
        if print_progress:
            print(f"File {output_path} already exists. Use --overwrite to replace.")
        return
    submission = fetch_submission(url, id)

    background_source = Defaults.background_source_h
    background_position = Defaults.background_position_h
    card_position = Defaults.card_position_h
    webcam_position = Defaults.webcam_position_h
    screen_position = Defaults.screen_position_h

    if vertical:
        background_source = Defaults.background_source
        background_position = Defaults.background_position
        card_position = Defaults.card_position
        webcam_position = Defaults.webcam_position
        screen_position = Defaults.screen_position

    render_reaction(
        title=submission.title,
        subtitle=submission.subtitle,
        hashtag=submission.hashtag,
        task=submission.task,
        time=submission.time,
        outcome=submission.outcome,
        success=submission.success,
        rank_before=submission.rank_before,
        rank_after=submission.rank_after,
        logo_source=submission.logo_source,
        webcam_source=submission.webcam_source,
        screen_source=submission.screen_source,
        background_source=background_source,
        success_audio_path=success_audio_path,
        fail_audio_path=fail_audio_path,
        output_path=output_path,
        print_progress=print_progress,
        vcodec=vcodec,
        acodec=acodec,
        sound=sound,
        background_position=background_position,
        card_position=card_position,
        webcam_position=webcam_position,
        screen_position=screen_position,
        hwaccel=hwaccel,
    )
