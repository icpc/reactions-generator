import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reactions_generator import worker


def without_progress(items: list[str], **_kwargs: object) -> list[str]:
    return items


class WorkerTests(unittest.TestCase):
    def test_invalid_worker_configuration_fails_before_fetching_or_creating_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "output")
            for count, worker_id in ((0, 0), (-1, 0), (2, -1), (2, 2)):
                with (
                    self.subTest(total_workers=count, worker_id=worker_id),
                    patch.object(worker, "fetch_submission_ids") as fetch,
                    self.assertRaises(ValueError),
                ):
                    worker.continuous_build_submission(
                        "https://example.com",
                        total_workers=count,
                        worker_id=worker_id,
                        output_directory=str(output),
                    )
                fetch.assert_not_called()
                self.assertFalse(output.exists())

    def test_failed_submission_does_not_prevent_the_next_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(
                    worker,
                    "fetch_submission_ids",
                    side_effect=[["first", "second"], KeyboardInterrupt],
                ),
                patch.object(
                    worker,
                    "build_submission",
                    side_effect=[ValueError("bad submission"), None],
                ) as build,
                patch.object(worker, "tqdm", side_effect=without_progress),
                self.assertLogs(worker.logger, level="ERROR"),
                self.assertRaises(KeyboardInterrupt),
            ):
                worker.continuous_build_submission(
                    "https://example.com", output_directory=directory
                )

            self.assertEqual(
                [call.kwargs["id"] for call in build.call_args_list],
                ["first", "second"],
            )
            self.assertIn("bad submission", Path(directory, "first.err").read_text())

    def test_workers_partition_a_batch_without_duplicate_or_missing_submissions(
        self,
    ) -> None:
        ids = [str(value) for value in range(40)]
        assigned: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            for worker_id in range(4):
                with (
                    patch.object(
                        worker,
                        "fetch_submission_ids",
                        side_effect=[ids, KeyboardInterrupt],
                    ),
                    patch.object(worker, "build_submission") as build,
                    patch.object(worker, "tqdm", side_effect=without_progress),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    worker.continuous_build_submission(
                        "https://example.com",
                        total_workers=4,
                        worker_id=worker_id,
                        output_directory=directory,
                    )
                assigned.extend(call.kwargs["id"] for call in build.call_args_list)

        self.assertCountEqual(assigned, ids)


if __name__ == "__main__":
    unittest.main()
