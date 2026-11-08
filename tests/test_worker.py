import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from threading import Event
from typing import cast
from unittest.mock import Mock, patch

from reactions_generator import worker
from reactions_generator.overlayer import RunSummary
from reactions_generator.scheduler import SubmissionQueue


class WorkerTests(unittest.TestCase):
    def test_invalid_configuration_fails_before_io(self) -> None:
        invalid: list[dict[str, int | float]] = [
            {"total_workers": 0},
            {"total_workers": -1},
            {"worker_id": -1},
            {"total_workers": 2, "worker_id": 2},
            {"poll_interval": 0},
            {"poll_interval": -1},
            {"poll_interval": float("nan")},
            {"poll_interval": float("inf")},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "output")
            for options in invalid:
                with (
                    self.subTest(options=options),
                    patch.object(worker, "fetch_runs") as fetch,
                    patch.object(worker, "Thread") as thread,
                    self.assertRaises(ValueError),
                ):
                    worker.continuous_build_submission(
                        "https://example.com",
                        output_directory=str(output),
                        total_workers=int(options.get("total_workers", 1)),
                        worker_id=int(options.get("worker_id", 0)),
                        poll_interval=options.get("poll_interval", 2),
                    )
                fetch.assert_not_called()
                thread.assert_not_called()
                self.assertFalse(output.exists())

    def test_polling_partitions_runs_and_skips_existing_videos(self) -> None:
        runs = [RunSummary(str(value)) for value in range(40)]
        assigned: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "5.mp4").write_bytes(b"existing render")
            for worker_id in range(4):
                stop = Mock(spec=Event)
                stop.is_set.return_value = False
                stop.wait.return_value = True
                queue = Mock(spec=SubmissionQueue)
                with patch.object(worker, "fetch_runs", return_value=runs):
                    worker.poll_submissions(
                        "https://example.com",
                        queue,
                        cast(Event, stop),
                        directory,
                        4,
                        worker_id,
                        2,
                    )
                snapshot: list[RunSummary] = queue.update.call_args.args[0]
                assigned.extend(run.id for run in snapshot)
                stop.wait.assert_called_once_with(2)

        self.assertCountEqual(assigned, [run.id for run in runs if run.id != "5"])

    def test_polling_failure_retries_without_discarding_the_queue(self) -> None:
        stop = Mock(spec=Event)
        stop.is_set.return_value = False
        stop.wait.side_effect = [False, True]
        queue = Mock(spec=SubmissionQueue)
        runs = [RunSummary("eventually-available")]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                worker, "fetch_runs", side_effect=[OSError("offline"), runs]
            ) as fetch,
            self.assertLogs(worker.logger, level="ERROR"),
        ):
            worker.poll_submissions(
                "https://example.com", queue, cast(Event, stop), directory, 1, 0, 2
            )

        self.assertEqual(fetch.call_count, 2)
        queue.update.assert_called_once_with(runs)
        self.assertEqual(stop.wait.call_count, 2)

    def test_failed_submission_does_not_prevent_the_next_render(self) -> None:
        queue = SubmissionQueue()
        queue.update([RunSummary("bad"), RunSummary("next")])
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(worker, "SubmissionQueue", return_value=queue),
                patch.object(worker, "Thread") as thread,
                patch.object(worker, "tqdm"),
                patch.object(
                    worker,
                    "build_submission",
                    side_effect=[ValueError("invalid submission"), KeyboardInterrupt],
                ) as build,
                self.assertLogs(worker.logger, level="ERROR"),
                self.assertRaises(KeyboardInterrupt),
            ):
                worker.continuous_build_submission(
                    "https://example.com", output_directory=directory
                )
            self.assertEqual(
                [call.kwargs["id"] for call in build.call_args_list], ["bad", "next"]
            )
            self.assertIn("invalid submission", Path(directory, "bad.err").read_text())
            thread.return_value.join.assert_called_once()
            self.assertIsNone(queue.take(timeout=0))

    def test_new_first_to_solve_arrives_during_render_and_runs_next(self) -> None:
        started = Event()
        refreshed = Event()
        finished = Event()
        rendered: list[str] = []
        initial = [RunSummary("running"), RunSummary("rest")]
        latest = [
            *initial,
            RunSummary("accepted", is_accepted=True),
            RunSummary("first", is_accepted=True, is_first_to_solve=True),
        ]
        fetch_count = 0

        class ObservedQueue(SubmissionQueue):
            def take(self, timeout: float | None = None) -> RunSummary | None:
                run = super().take(timeout=max(timeout or 0, 3))
                if run is None:
                    raise TimeoutError("Polling did not provide the expected work")
                return run

            def update(self, runs: Sequence[RunSummary]) -> None:
                super().update(runs)
                if any(run.id == "first" for run in runs):
                    refreshed.set()

        def fetch(_url: str) -> list[RunSummary]:
            nonlocal fetch_count
            fetch_count += 1
            if fetch_count == 1:
                return initial
            if fetch_count == 2:
                if not started.wait(3):
                    raise RuntimeError("Foreground render did not start")
                return latest
            if not finished.wait(3):
                raise RuntimeError("Foreground render did not finish")
            return []

        with tempfile.TemporaryDirectory() as directory:

            def build(*, id: str, **_kwargs: object) -> None:
                rendered.append(id)
                if id == "running":
                    started.set()
                    self.assertTrue(
                        refreshed.wait(3), "No polling occurred during the render"
                    )
                if id == "rest":
                    finished.set()
                    raise KeyboardInterrupt
                Path(directory, f"{id}.mp4").write_bytes(b"rendered")

            try:
                with (
                    patch.object(worker, "SubmissionQueue", ObservedQueue),
                    patch.object(worker, "fetch_runs", side_effect=fetch),
                    patch.object(worker, "build_submission", side_effect=build),
                    patch.object(worker, "tqdm"),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    worker.continuous_build_submission(
                        "https://example.com",
                        output_directory=directory,
                        poll_interval=0.01,
                    )
            finally:
                finished.set()

        self.assertEqual(rendered, ["running", "first", "accepted", "rest"])
        self.assertGreaterEqual(fetch_count, 2)


if __name__ == "__main__":
    unittest.main()
