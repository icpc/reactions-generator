import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread

from reactions_generator.overlayer import RunSummary
from reactions_generator.scheduler import SubmissionProgress, SubmissionQueue
from tests.fixtures import SUBMISSION, queue_runs, take_run


class SubmissionQueueTests(unittest.TestCase):
    def test_prepared_details_are_reused_and_invalidated_when_run_changes(self) -> None:
        queue = SubmissionQueue()
        original = RunSummary("submission", verdict="WA")
        changed = RunSummary("submission", verdict="AC", is_accepted=True)
        queue.update([original])
        self.assertEqual(queue.to_prepare, [original])
        queue.prepare(original, SUBMISSION)
        queue.update([original])
        self.assertEqual(queue.to_prepare, [])
        queue.update([changed])
        self.assertEqual(queue.to_prepare, [changed])
        queue.prepare(original, SUBMISSION)
        self.assertEqual(queue.to_prepare, [changed])
        queue.prepare(changed, SUBMISSION)
        task = queue.take()
        self.assertIsNotNone(task)
        if task is not None:
            self.assertEqual(task.run, changed)
            self.assertIs(task.submission, SUBMISSION)
        self.assertEqual(queue.to_prepare, [])

    def test_highest_priority_waits_for_its_details(self) -> None:
        queue = SubmissionQueue()
        first = RunSummary("first", is_first_to_solve=True)
        rest = RunSummary("rest")
        queue.update([rest, first])
        queue.prepare(rest, SUBMISSION)
        started = Event()
        taken = Event()
        result: list[RunSummary | None] = []

        def consume() -> None:
            started.set()
            result.append(take_run(queue))
            taken.set()

        thread = Thread(target=consume)
        thread.start()
        try:
            self.assertTrue(started.wait(3))
            self.assertFalse(taken.wait(0.05))
            queue.prepare(first, SUBMISSION)
            self.assertTrue(taken.wait(3))
            self.assertEqual(result, [first])
        finally:
            queue.close()
            thread.join(3)
        self.assertFalse(thread.is_alive())

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
                queue_runs(queue, runs, output_directory=directory)
                self.assertEqual(queue.progress, SubmissionProgress(1, 1, 1))
            error.unlink()
            queue_runs(queue, runs, output_directory=directory)
            self.assertEqual(queue.progress, SubmissionProgress(1, 0, 2))

    def test_in_flight_output_is_counted_only_after_completion(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            queue = SubmissionQueue()
            runs = [RunSummary("active")]
            queue_runs(queue, runs, output_directory=directory)
            self.assertEqual(take_run(queue), runs[0])
            (directory / "active.mp4").write_bytes(b"video")
            queue_runs(queue, runs, output_directory=directory)
            self.assertEqual(queue.progress, SubmissionProgress(0, 0, 1))
            queue.complete("active", succeeded=True)
            self.assertEqual(queue.progress, SubmissionProgress(1, 0, 0))
            queue_runs(queue, runs)
            self.assertEqual(queue.progress, SubmissionProgress(1, 0, 0))
            queue_runs(queue, runs, output_directory=directory)
            self.assertEqual(queue.progress, SubmissionProgress(1, 0, 0))

    def test_priority_and_fifo_order(self) -> None:
        queue = SubmissionQueue()
        queue_runs(
            queue,
            [
                RunSummary("rest-1"),
                RunSummary("accepted-1", is_accepted=True),
                RunSummary("first-1", is_first_to_solve=True),
                RunSummary("rest-2"),
                RunSummary("first-2", is_first_to_solve=True),
                RunSummary("accepted-2", is_accepted=True),
            ],
        )

        claimed = [take_run(queue) for _ in range(6)]

        self.assertEqual(
            [run.id for run in claimed if run is not None],
            ["first-1", "first-2", "accepted-1", "accepted-2", "rest-1", "rest-2"],
        )
        self.assertEqual(len(queue), 0)

    def test_updates_reprioritize_without_changing_fifo_ties(self) -> None:
        queue = SubmissionQueue()
        queue_runs(
            queue,
            [
                RunSummary("old-1", is_accepted=True),
                RunSummary("promoted"),
                RunSummary("old-2", is_accepted=True),
            ],
        )
        queue_runs(
            queue,
            [
                RunSummary("old-2", is_accepted=True),
                RunSummary("new", is_accepted=True),
                RunSummary("promoted", is_first_to_solve=True),
                RunSummary("old-1", is_accepted=True),
            ],
        )

        claimed = [take_run(queue) for _ in range(4)]

        self.assertEqual(
            [run.id for run in claimed if run is not None],
            ["promoted", "old-1", "old-2", "new"],
        )

    def test_snapshots_remove_vanished_and_unjudged_runs(self) -> None:
        queue = SubmissionQueue()
        queue_runs(queue, [RunSummary("vanished"), RunSummary("still-present")])
        queue_runs(
            queue,
            [
                RunSummary("still-present"),
                RunSummary("unjudged", verdict="IN_PROGRESS"),
            ],
        )
        self.assertEqual(len(queue), 1)
        self.assertEqual(take_run(queue), RunSummary("still-present"))
        self.assertEqual(len(queue), 0)

        queue_runs(queue, [RunSummary("unjudged", verdict="AC", is_accepted=True)])
        run = take_run(queue)
        self.assertIsNotNone(run)
        self.assertEqual(run.id if run else None, "unjudged")

    def test_duplicate_and_in_flight_runs_are_not_claimed_twice(self) -> None:
        queue = SubmissionQueue()
        first = RunSummary("first")
        second = RunSummary("second")
        queue_runs(queue, [first, second])
        self.assertEqual(take_run(queue), first)

        queue_runs(queue, [first, first, second, second])
        self.assertEqual(len(queue), 1)
        self.assertEqual(take_run(queue), second)
        self.assertEqual(len(queue), 0)

        queue.complete("first")
        self.assertEqual(len(queue), 0)
        queue_runs(queue, [first, second])
        self.assertEqual(take_run(queue), first)
        self.assertEqual(len(queue), 0)

    def test_new_snapshot_wakes_a_waiting_consumer(self) -> None:
        queue = SubmissionQueue()
        waiting = Event()
        finished = Event()
        claimed: list[RunSummary | None] = []

        def consume() -> None:
            waiting.set()
            claimed.append(take_run(queue))
            finished.set()

        consumer = Thread(target=consume, daemon=True)
        consumer.start()
        try:
            self.assertTrue(waiting.wait(2))
            queue_runs(queue, [RunSummary("new")])
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
            claimed.append(take_run(queue))
            finished.set()

        consumer = Thread(target=consume, daemon=True)
        consumer.start()
        try:
            self.assertTrue(waiting.wait(2))
            queue.close()
            self.assertTrue(finished.wait(2))
            self.assertEqual(claimed, [None])
            queue_runs(queue, [RunSummary("ignored")])
            self.assertEqual(len(queue), 0)
            self.assertIsNone(take_run(queue))
        finally:
            queue.close()
            consumer.join(timeout=2)
        self.assertFalse(consumer.is_alive())


if __name__ == "__main__":
    unittest.main()
