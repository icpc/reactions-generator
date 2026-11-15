from collections.abc import Sequence
from pathlib import Path

from reactions_generator.overlayer import RunSummary, Submission
from reactions_generator.scheduler import SubmissionQueue

SUBMISSION = Submission(
    title="Team",
    subtitle="University",
    hashtag="#contest",
    task="A",
    time=1000,
    outcome="AC",
    success=True,
    rank_before=2,
    rank_after=1,
    logo_source="#ffffff",
    webcam_source="webcam.mp4",
    screen_source="screen.mp4",
)


def queue_runs(
    queue: SubmissionQueue,
    runs: Sequence[RunSummary],
    *,
    output_directory: Path | None = None,
) -> None:
    queue.update(runs, output_directory=output_directory)
    for run in queue.to_prepare:
        queue.prepare(run, SUBMISSION)


def take_run(queue: SubmissionQueue) -> RunSummary | None:
    task = queue.take()
    return task.run if task is not None else None
