import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from threading import Event, Thread
from unittest.mock import Mock, patch

from reactions_generator import worker
from reactions_generator.overlayer import RunSummary
from reactions_generator.scheduler import SubmissionQueue


def run_until_interrupt(
    url: str, output_directory: str, total_workers: int, poll_interval: float = 2.0
) -> None:
    try:
        worker.continuous_build_submission(
            url,
            output_directory=output_directory,
            total_workers=total_workers,
            poll_interval=poll_interval,
        )
    except KeyboardInterrupt:
        pass


class WorkerTests(unittest.TestCase):
    def test_failed_submission_is_skipped_until_error_file_is_removed(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(worker, "fetch_runs", return_value=[RunSummary("failed")]),
        ):
            queue = SubmissionQueue()
            worker.poll_submissions("https://example.com", queue, directory, 1, None)
            self.assertEqual(queue.take(), RunSummary("failed"))
            error = Path(directory, "failed.err")
            error.write_text("encoding failed")
            queue.complete("failed")

            worker.poll_submissions("https://example.com", queue, directory, 1, None)
            self.assertEqual(len(queue), 0)
            restarted = SubmissionQueue()
            worker.poll_submissions(
                "https://example.com", restarted, directory, 1, None
            )
            self.assertEqual(len(restarted), 0)

            error.unlink()
            worker.poll_submissions(
                "https://example.com", restarted, directory, 1, None
            )
            self.assertEqual(restarted.take(), RunSummary("failed"))

    def test_progress_reaches_total_including_failed_renders(self) -> None:
        queue = SubmissionQueue()
        queue.update([RunSummary("good"), RunSummary("bad")])
        finished = Event()
        progress = Mock(n=0)
        progress.__enter__ = Mock(return_value=progress)
        progress.__exit__ = Mock(return_value=False)

        def update(count: int) -> None:
            progress.n += count
            if progress.n == 2:
                finished.set()

        progress.update.side_effect = update

        def build(*, id: str, **_kwargs: object) -> None:
            if id == "bad":
                raise ValueError("invalid submission")

        def poll(*_args: object) -> None:
            self.assertTrue(finished.wait(3))
            raise KeyboardInterrupt

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(worker, "SubmissionQueue", return_value=queue),
            patch.object(worker, "poll_submissions", side_effect=poll),
            patch.object(worker, "build_submission", side_effect=build),
            patch.object(worker, "tqdm", return_value=progress),
            self.assertLogs(worker.logger, level="ERROR"),
            self.assertRaises(KeyboardInterrupt),
        ):
            worker.continuous_build_submission(
                "https://example.com", output_directory=directory
            )
        self.assertEqual(progress.n, 2)
        self.assertEqual(progress.total, 2)

    def test_shutdown_does_not_record_interrupted_render_as_failure(self) -> None:
        stop = Event()
        started = Event()
        queue = SubmissionQueue()
        queue.update([RunSummary("interrupted"), RunSummary("waiting")])

        def build(**_kwargs: object) -> None:
            started.set()
            if not stop.wait(3):
                raise TimeoutError("shutdown was not signalled")
            raise OSError("encoder interrupted")

        def poll(*_args: object) -> None:
            self.assertTrue(started.wait(3))
            stop.set()
            raise KeyboardInterrupt

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(worker, "Event", return_value=stop),
            patch.object(worker, "SubmissionQueue", return_value=queue),
            patch.object(worker, "poll_submissions", side_effect=poll),
            patch.object(worker, "build_submission", side_effect=build) as render,
            patch.object(worker, "tqdm"),
            patch.object(worker, "log_error") as log_error,
            self.assertRaises(KeyboardInterrupt),
        ):
            try:
                worker.continuous_build_submission(
                    "https://example.com", output_directory=directory
                )
            finally:
                self.assertEqual(list(Path(directory).glob("*.err")), [])
                log_error.assert_not_called()
                render.assert_called_once()

    def test_interrupt_marks_shutdown_and_restores_signal_handlers(self) -> None:
        import signal

        stop = Event()
        previous = signal.getsignal(signal.SIGINT)
        with worker.shutdown_signals(stop):
            handler = signal.getsignal(signal.SIGINT)
            self.assertTrue(callable(handler))
            if callable(handler):
                handler(signal.SIGINT, None)
        self.assertTrue(stop.is_set())
        self.assertIs(signal.getsignal(signal.SIGINT), previous)

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
                    patch.object(worker, "ThreadPoolExecutor") as executor,
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
                executor.assert_not_called()
                self.assertFalse(output.exists())

    def test_polling_partitions_runs_and_skips_existing_videos(self) -> None:
        runs = [RunSummary(str(value)) for value in range(40)]
        assigned: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "5.mp4").write_bytes(b"existing render")
            for worker_id in range(4):
                queue = Mock(spec=SubmissionQueue)
                with patch.object(worker, "fetch_runs", return_value=runs):
                    worker.poll_submissions(
                        "https://example.com",
                        queue,
                        directory,
                        4,
                        worker_id,
                    )
                snapshot: list[RunSummary] = queue.update.call_args.args[0]
                assigned.extend(run.id for run in snapshot)

        self.assertCountEqual(assigned, [run.id for run in runs if run.id != "5"])

    def test_polling_failure_retries_without_discarding_the_queue(self) -> None:
        queue = Mock(spec=SubmissionQueue)
        runs = [RunSummary("eventually-available")]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                worker, "fetch_runs", side_effect=[OSError("offline"), runs]
            ) as fetch,
            self.assertLogs(worker.logger, level="ERROR"),
        ):
            for _ in range(2):
                worker.poll_submissions("https://example.com", queue, directory, 1, 0)

        self.assertEqual(fetch.call_count, 2)
        queue.update.assert_called_once_with(runs)

    def test_failed_submission_does_not_prevent_the_next_render(self) -> None:
        queue = SubmissionQueue()
        queue.update([RunSummary("bad"), RunSummary("next")])
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(worker, "SubmissionQueue", return_value=queue),
                patch.object(worker, "poll_submissions"),
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
                    "https://example.com",
                    output_directory=directory,
                    worker_id=0,
                    total_workers=1,
                )
            self.assertEqual(
                [call.kwargs["id"] for call in build.call_args_list], ["bad", "next"]
            )
            self.assertIn("invalid submission", Path(directory, "bad.err").read_text())

    def test_local_workers_render_concurrently_without_duplicate_claims(self) -> None:
        both_started = Event()
        release = Event()
        started: list[str] = []
        runs = [RunSummary("first"), RunSummary("second")]

        def build(*, id: str, **_kwargs: object) -> None:
            started.append(id)
            if len(started) == 2:
                both_started.set()
            if not release.wait(3):
                raise TimeoutError("renderers did not overlap")
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:  # noqa: SIM117
            with (
                patch.object(worker, "fetch_runs", return_value=runs),
                patch.object(worker, "build_submission", side_effect=build),
                patch.object(worker, "tqdm"),
            ):
                runner = Thread(
                    target=run_until_interrupt,
                    args=("https://example.com",),
                    kwargs={
                        "output_directory": directory,
                        "total_workers": 2,
                        "poll_interval": 0.01,
                    },
                )
                runner.start()
                try:
                    self.assertTrue(both_started.wait(3), "renders did not overlap")
                finally:
                    release.set()
                    runner.join(3)
                self.assertFalse(runner.is_alive())
        self.assertCountEqual(started, ["first", "second"])

    def test_free_slot_takes_new_priority_while_another_render_is_active(self) -> None:
        slow_started = Event()
        fast_started = Event()
        release_slow = Event()
        release_fast = Event()
        priority_started = Event()
        queue = SubmissionQueue()
        queue.update([RunSummary("slow"), RunSummary("fast"), RunSummary("rest")])
        rendered: list[str] = []

        def build(*, id: str, **_kwargs: object) -> None:
            rendered.append(id)
            if id == "slow":
                slow_started.set()
                if not release_slow.wait(3):
                    raise TimeoutError("slow render was not released")
            elif id == "fast":
                fast_started.set()
                if not release_fast.wait(3):
                    raise TimeoutError("fast render was not released")
            elif id == "priority":
                priority_started.set()
                raise KeyboardInterrupt

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(worker, "SubmissionQueue", return_value=queue),
            patch.object(worker, "poll_submissions"),
            patch.object(worker, "build_submission", side_effect=build),
            patch.object(worker, "tqdm") as progress,
        ):
            runner = Thread(
                target=run_until_interrupt,
                args=("https://example.com",),
                kwargs={"output_directory": directory, "total_workers": 2},
            )
            runner.start()
            try:
                self.assertTrue(slow_started.wait(3))
                self.assertTrue(fast_started.wait(3))
                queue.update(
                    [
                        RunSummary("rest"),
                        RunSummary("priority", is_first_to_solve=True),
                    ]
                )
                release_fast.set()
                self.assertTrue(priority_started.wait(3))
                self.assertFalse(release_slow.is_set())
            finally:
                release_fast.set()
                release_slow.set()
                runner.join(3)
            self.assertFalse(runner.is_alive())
            progress.assert_called_once_with(
                total=0,
                desc="Rendering submissions",
                unit="submission",
                dynamic_ncols=True,
                disable=None,
            )
        self.assertCountEqual(rendered, ["slow", "fast", "priority"])

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
            def take(self) -> RunSummary | None:
                run = super().take()
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
            return [] if finished.is_set() else latest

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
                        worker_id=0,
                        total_workers=1,
                    )
            finally:
                finished.set()

        self.assertEqual(rendered, ["running", "first", "accepted", "rest"])
        self.assertGreaterEqual(fetch_count, 2)


if __name__ == "__main__":
    unittest.main()
