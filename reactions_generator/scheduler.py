"""A blocking priority queue for submission renderers."""

from collections.abc import Sequence
from threading import Condition

from reactions_generator.overlayer import RunSummary


class SubmissionQueue:
    """Refresh waiting runs while preserving FIFO order within each priority."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._pending: dict[str, RunSummary] = {}
        self._in_flight: set[str] = set()
        self._closed = False

    def update(self, runs: Sequence[RunSummary]) -> None:
        latest = {run.id: run for run in runs if not run.is_pending}
        with self._condition:
            if self._closed:
                return
            latest = {
                id: run for id, run in latest.items() if id not in self._in_flight
            }
            # Updating existing keys keeps their first-discovery insertion order.
            self._pending = {id: latest[id] for id in self._pending if id in latest}
            self._pending.update(latest)
            self._condition.notify_all()

    def take(self) -> RunSummary | None:
        """Wait for work, or return None when the queue is closed."""
        with self._condition:
            self._condition.wait_for(lambda: self._pending or self._closed)
            if self._closed:
                return None
            # min preserves insertion order for equal priorities. The scan is
            # inexpensive compared with a render and avoids a second queue index.
            run = min(self._pending.values(), key=lambda run: run.priority)
            del self._pending[run.id]
            self._in_flight.add(run.id)
            return run

    def complete(self, id: str) -> None:
        """Release ownership; polling skips completed and failed output files."""
        with self._condition:
            self._in_flight.discard(id)

    def close(self) -> None:
        """Wake idle renderers and stop dispatching new work."""
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._pending)

    @property
    def remaining(self) -> int:
        """Number of waiting and active submissions in one consistent snapshot."""
        with self._condition:
            return len(self._pending) + len(self._in_flight)
