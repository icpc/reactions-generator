"""Submission progress for terminal sessions and Docker logs."""

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
)
from rich.text import Text


class SpeedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        if not speed or speed <= 0:
            return Text("--/s", style="dim")
        return Text(f"{speed:.1f}/s", style="cyan")


def compact_time(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    whole_seconds = max(0, int(seconds))
    minutes, seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02}m"
    if minutes:
        return f"{minutes}m{seconds:02}s"
    return f"{seconds}s"


class ElapsedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(compact_time(task.elapsed), style="dim")


class RemainingColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(f"ETA {compact_time(task.time_remaining)}", style="dim")


class FailedColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        failed = task.fields.get("failed", 0)
        return Text(f"✗ {failed}", style="bold red") if failed else Text("")


class RenderProgress(Progress):
    """Show a Rich bar in terminals and periodic newline summaries in logs."""

    def __init__(self, console: Console | None = None) -> None:
        console = console if console is not None else Console(stderr=True)
        plain = (
            not console.is_terminal or os.environ.get("REACTIONS_PROGRESS") == "plain"
        )
        super().__init__(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=16, complete_style="blue", finished_style="green"),
            MofNCompleteColumn(),
            ElapsedColumn(),
            RemainingColumn(),
            SpeedColumn(),
            TextColumn("[dim]]"),
            FailedColumn(),
            console=console,
            disable=plain,
            auto_refresh=True,
            refresh_per_second=5,
            get_time=monotonic,
        )
        self._task_id = self.add_task("Rendering submissions", total=0, failed=0)
        self._processed = 0
        self._failed = 0
        self._next_report = 0.0

    def report(self, remaining: int, *, succeeded: bool | None = None) -> None:
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
                task = self.tasks[0]
                speed = task.finished_speed or task.speed
                rate = f"{speed:.1f}/s" if speed else "--/s"
                self.console.print(
                    f"Rendering submissions: {self._processed}/{total} processed, "
                    f"{self._failed} failed, {remaining} remaining, {rate}",
                    highlight=False,
                    soft_wrap=True,
                )
                self._next_report = now + 30
