"""Render progress for interactive terminals and line-based logs."""

import os
from time import monotonic

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text


def format_rate(task: Task) -> str:
    """Show Rich's recent rate, with a session average until two samples exist."""
    speed = task.speed
    if speed is None and task.completed and (elapsed := task.elapsed):
        speed = task.completed / elapsed
    return f"{speed:.2f} submissions/s" if speed is not None else "-- submissions/s"


class RenderRateColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(format_rate(task))


class RenderProgress(Progress):
    """Display one live task or report its progress periodically in logs."""

    def __init__(self, console: Console | None = None) -> None:
        console = console if console is not None else Console(stderr=True)
        plain = (
            not console.is_terminal or os.environ.get("REACTIONS_PROGRESS") == "plain"
        )
        super().__init__(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[failed]} failed"),
            RenderRateColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=plain,
            auto_refresh=False,
            get_time=monotonic,
        )
        self._task_id = self.add_task("Rendering submissions", total=0, failed=0)
        self._processed = 0
        self._failed = 0
        self._next_report = 0.0

    def report(self, remaining: int, *, succeeded: bool | None = None) -> None:
        """Update the queue snapshot and optionally count a completed attempt."""
        if succeeded is not None:
            self._processed += 1
            if not succeeded:
                self._failed += 1

        total = self._processed + remaining
        self.update(
            self._task_id,
            completed=self._processed,
            total=total,
            failed=self._failed,
            refresh=True,
        )

        if self.disable:
            now = monotonic()
            if now >= self._next_report or (succeeded is not None and remaining == 0):
                self.console.print(
                    f"Rendering submissions: {self._processed}/{total} processed, "
                    f"{self._failed} failed, {remaining} remaining, "
                    f"{format_rate(self.tasks[0])}",
                    highlight=False,
                    soft_wrap=True,
                )
                self._next_report = now + 30
