import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread

from reactions_generator.overlayer import RunSummary
from reactions_generator.scheduler import SubmissionProgress, SubmissionQueue


class SubmissionQueueTests(unittest.TestCase):
    def test_existing_results_count_once_and_deleted_errors_allow_retry(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory / "good.mp4").write_bytes(b"video")
            (directory / "good.err").write_text("stale error")
            error = directory / "bad.err"
            error.write_text("failed")
            runs = [
                RunSummary("good"),
                RunSummary("bad"),
                RunSummary("waiting"),
                RunSummary("judging", verdict="IN_PROGRESS"),
            ]
            queue = SubmissionQueue()
            for _ in range(2):
                queue.update(runs, output_directory=directory)
                self.assertEqual(queue.progress, SubmissionProgress(1, 1, 1))
            error.unlink()
            queue.update(runs, output_directory=directory)
            self.assertEqual(queue.progress, SubmissionProgress(1, 0, 2))

    def test_in_flight_output_is_counted_only_after_completion(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            queue = SubmissionQueue()
            runs = [RunSummary("active")]
            queue.update(runs, output_directory=directory)
            self.assertEqual(queue.take(), runs[0])
            (directory / "active.mp4").write_bytes(b"video")
            queue.update(runs, output_directory=directory)
            self.assertEqual(queue.progress, SubmissionProgress(0, 0, 1))
            queue.complete("active", succeeded=True)
            self.assertEqual(queue.progress, SubmissionProgress(1, 0, 0))
            queue.update(runs)
            self.assertEqual(queue.progress, SubmissionProgress(1, 0, 0))
            queue.update(runs, output_directory=directory)
            self.assertEqual(queue.progress, SubmissionProgress(1, 0, 0))

    def test_priority_and_fifo_order(self) -> None:
        queue = SubmissionQueue()
        queue.update(
            [
                RunSummary("rest-1"),
                RunSummary("accepted-1", is_accepted=True),
                RunSummary("first-1", is_first_to_solve=True),
                RunSummary("rest-2"),
                RunSummary("first-2", is_first_to_solve=True),
                RunSummary("accepted-2", is_accepted=True),
            ]
        )

        claimed = [queue.take() for _ in range(6)]

        self.assertEqual(
            [run.id for run in claimed if run is not None],
            ["first-1", "first-2", "accepted-1", "accepted-2", "rest-1", "rest-2"],
        )
        self.assertEqual(len(queue), 0)

    def test_updates_reprioritize_without_changing_fifo_ties(self) -> None:
        queue = SubmissionQueue()
        queue.update(
            [
                RunSummary("old-1", is_accepted=True),
                RunSummary("promoted"),
                RunSummary("old-2", is_accepted=True),
            ]
        )
        queue.update(
            [
                RunSummary("old-2", is_accepted=True),
                RunSummary("new", is_accepted=True),
                RunSummary("promoted", is_first_to_solve=True),
                RunSummary("old-1", is_accepted=True),
            ]
        )

        claimed = [queue.take() for _ in range(4)]

        self.assertEqual(
            [run.id for run in claimed if run is not None],
            ["promoted", "old-1", "old-2", "new"],
        )

    def test_snapshots_remove_vanished_and_unjudged_runs(self) -> None:
        queue = SubmissionQueue()
        queue.update([RunSummary("vanished"), RunSummary("still-present")])
        queue.update(
            [
                RunSummary("still-present"),
                RunSummary("unjudged", verdict="IN_PROGRESS"),
            ]
        )
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.take(), RunSummary("still-present"))
        self.assertEqual(len(queue), 0)

        queue.update([RunSummary("unjudged", verdict="AC", is_accepted=True)])
        run = queue.take()
        self.assertIsNotNone(run)
        self.assertEqual(run.id if run else None, "unjudged")

    def test_duplicate_and_in_flight_runs_are_not_claimed_twice(self) -> None:
        queue = SubmissionQueue()
        first = RunSummary("first")
        second = RunSummary("second")
        queue.update([first, second])
        self.assertEqual(queue.take(), first)

        queue.update([first, first, second, second])
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.take(), second)
        self.assertEqual(len(queue), 0)

        queue.complete("first")
        self.assertEqual(len(queue), 0)
        queue.update([first, second])
        self.assertEqual(queue.take(), first)
        self.assertEqual(len(queue), 0)

    def test_new_snapshot_wakes_a_waiting_consumer(self) -> None:
        queue = SubmissionQueue()
        waiting = Event()
        finished = Event()
        claimed: list[RunSummary | None] = []

        def consume() -> None:
            waiting.set()
            claimed.append(queue.take())
            finished.set()

        consumer = Thread(target=consume, daemon=True)
        consumer.start()
        try:
            self.assertTrue(waiting.wait(2))
            queue.update([RunSummary("new")])
            self.assertTrue(finished.wait(2))
            self.assertEqual(claimed, [RunSummary("new")])
        finally:
            queue.close()
            consumer.join(timeout=2)
        self.assertFalse(consumer.is_alive())

    def test_close_wakes_waiters_and_rejects_further_work(self) -> None:
        queue = SubmissionQueue()
        waiting = Event()
        finished = Event()
        claimed: list[RunSummary | None] = []

        def consume() -> None:
            waiting.set()
            claimed.append(queue.take())
            finished.set()

        consumer = Thread(target=consume, daemon=True)
        consumer.start()
        try:
            self.assertTrue(waiting.wait(2))
            queue.close()
            self.assertTrue(finished.wait(2))
            self.assertEqual(claimed, [None])
            queue.update([RunSummary("ignored")])
            self.assertEqual(len(queue), 0)
            self.assertIsNone(queue.take())
        finally:
            queue.close()
            consumer.join(timeout=2)
        self.assertFalse(consumer.is_alive())


if __name__ == "__main__":
    unittest.main()
