"""A live priority queue shared by submission discovery and rendering."""

import heapq
from collections.abc import Sequence
from threading import Condition

from reactions_generator.overlayer import RunSummary


class SubmissionQueue:
    """Keep the latest pending snapshot while a renderer owns in-flight runs.

    Priority changes take effect on each update. Runs with equal priority retain
    their first-seen order until they leave the pending queue. Completing a run
    only releases ownership: a failed run becomes eligible on a later update,
    rather than immediately looping on the same failure.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._pending: dict[str, tuple[int, RunSummary]] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._in_flight: set[str] = set()
        self._next_sequence = 0
        self._closed = False

    def update(self, runs: Sequence[RunSummary]) -> None:
        """Replace pending work with a complete, deduplicated discovery snapshot."""
        latest = {run.id: run for run in runs}
        with self._condition:
            if self._closed:
                return
            pending: dict[str, tuple[int, RunSummary]] = {}
            for id, run in latest.items():
                if run.is_pending or id in self._in_flight:
                    continue
                previous = self._pending.get(id)
                if previous is None:
                    sequence = self._next_sequence
                    self._next_sequence += 1
                else:
                    sequence = previous[0]
                pending[id] = (sequence, run)
            self._pending = pending
            # Rebuilding avoids retaining stale heap entries after reprioritizing
            # or removing runs on every poll of a long-running contest.
            self._heap = [
                (run.priority, sequence, id) for id, (sequence, run) in pending.items()
            ]
            heapq.heapify(self._heap)
            self._condition.notify_all()

    def take(self, timeout: float | None = None) -> RunSummary | None:
        """Claim the highest-priority run, waiting for work unless closed."""
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or bool(self._heap), timeout=timeout
            )
            if self._closed or not self._heap:
                return None
            _, _, id = heapq.heappop(self._heap)
            _, run = self._pending.pop(id)
            self._in_flight.add(id)
            return run

    def complete(self, id: str) -> None:
        """Release a claimed run after the render attempt finishes."""
        with self._condition:
            self._in_flight.discard(id)

    def close(self) -> None:
        """Stop accepting work and wake any renderer waiting on an empty queue."""
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._heap.clear()
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._pending)
