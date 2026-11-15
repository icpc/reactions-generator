"""A blocking priority queue for submission renderers."""

from collections.abc import Sequence
from pathlib import Path
from threading import Condition
from typing import NamedTuple

from reactions_generator.overlayer import RunSummary, Submission


class RenderTask(NamedTuple):
    """A submission and its scheduling metadata, ready for a renderer."""

    run: RunSummary
    submission: Submission

    @property
    def id(self) -> str:
        return self.run.id


class SubmissionProgress(NamedTuple):
    succeeded: int
    failed: int
    remaining: int


class SubmissionQueue:
    """Refresh waiting runs while preserving FIFO order within each priority."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._pending: dict[str, RunSummary] = {}
        self._prepared: dict[str, tuple[RunSummary, Submission]] = {}
        self._in_flight: set[str] = set()
        self._completed: dict[str, bool] = {}
        self._closed = False

    def update(
        self, runs: Sequence[RunSummary], *, output_directory: Path | None = None
    ) -> None:
        latest = {run.id: run for run in runs if not run.is_pending}
        with self._condition:
            if self._closed:
                return
            latest = {
                id: run for id, run in latest.items() if id not in self._in_flight
            }
            if output_directory is not None:
                completed: dict[str, bool] = {}
                for id in latest:
                    if (output_directory / f"{id}.mp4").exists():
                        completed[id] = True
                    elif (output_directory / f"{id}.err").exists():
                        completed[id] = False
                self._completed = completed
            latest = {
                id: run for id, run in latest.items() if id not in self._completed
            }
            # Updating existing keys keeps their first-discovery insertion order.
            self._pending = {id: latest[id] for id in self._pending if id in latest}
            self._pending.update(latest)
            self._prepared = {
                id: prepared
                for id, prepared in self._prepared.items()
                if self._pending.get(id) == prepared[0]
            }
            self._condition.notify_all()

    @property
    def to_prepare(self) -> list[RunSummary]:
        """Waiting submissions whose details need fetching, in render order."""
        with self._condition:
            return sorted(
                (run for id, run in self._pending.items() if id not in self._prepared),
                key=lambda run: run.priority,
            )

    def prepare(self, run: RunSummary, submission: Submission) -> None:
        """Attach details only while the waiting summary still matches."""
        with self._condition:
            if not self._closed and self._pending.get(run.id) == run:
                self._prepared[run.id] = (run, submission)
                self._condition.notify_all()

    def _next_ready(self) -> RunSummary | None:
        # A lower-priority prepared task waits for the head task's details.
        run = min(self._pending.values(), key=lambda run: run.priority, default=None)
        return run if run is not None and run.id in self._prepared else None

    def take(self) -> RenderTask | None:
        """Wait for work, or return None when the queue is closed."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or self._next_ready() is not None
            )
            if self._closed:
                return None
            run = self._next_ready()
            assert run is not None
            _, submission = self._prepared.pop(run.id)
            del self._pending[run.id]
            self._in_flight.add(run.id)
            return RenderTask(run, submission)

    def complete(self, id: str, *, succeeded: bool | None = None) -> None:
        """Release ownership and record a render's result when available."""
        with self._condition:
            self._in_flight.discard(id)
            self._pending.pop(id, None)
            self._prepared.pop(id, None)
            if succeeded is not None:
                self._completed[id] = succeeded
            self._condition.notify_all()

    def close(self) -> None:
        """Wake idle renderers and stop dispatching new work."""
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._prepared.clear()
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._pending)

    @property
    def remaining(self) -> int:
        """Number of waiting and active submissions in one consistent snapshot."""
        with self._condition:
            return len(self._pending) + len(self._in_flight)

    @property
    def progress(self) -> SubmissionProgress:
        """Count existing results and active/waiting work in one snapshot."""
        with self._condition:
            succeeded = sum(self._completed.values())
            return SubmissionProgress(
                succeeded=succeeded,
                failed=len(self._completed) - succeeded,
                remaining=len(self._pending) + len(self._in_flight),
            )
