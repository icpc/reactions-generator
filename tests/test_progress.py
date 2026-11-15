import logging
import unittest
from io import StringIO
from unittest.mock import patch

from reactions_generator.progress import RenderProgress, write_progress_message
from reactions_generator.scheduler import SubmissionProgress


class TerminalOutput(StringIO):
    def isatty(self) -> bool:
        return True


class ProgressTests(unittest.TestCase):
    def test_terminal_messages_and_logging_preserve_progress(self) -> None:
        output = TerminalOutput()
        with (
            patch.dict("os.environ", {"REACTIONS_PROGRESS": "auto"}),
            patch("sys.stderr", output),
            RenderProgress() as progress,
        ):
            progress.report(SubmissionProgress(0, 0, 2))
            write_progress_message("Starting render: example")
            logging.getLogger(__name__).warning("Example render warning")
            progress.report(SubmissionProgress(0, 1, 1))
            progress.report(SubmissionProgress(1, 1, 0))
        text = output.getvalue()
        self.assertIn("\r", text)
        self.assertIn("Starting render: example\n", text)
        self.assertIn("Example render warning\n", text)
        self.assertIn("2/2", text)
        self.assertIn("failed=1", text)

    def test_plain_status_message_does_not_redraw_the_bar(self) -> None:
        output = StringIO()
        with (
            patch.dict("os.environ", {"REACTIONS_PROGRESS": "plain"}),
            patch("sys.stderr", output),
            RenderProgress() as progress,
        ):
            progress.report(SubmissionProgress(0, 0, 2))
            write_progress_message("Starting render: example")
        self.assertNotIn("\r", output.getvalue())
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("0/2", lines[0])
        self.assertEqual(lines[1], "Starting render: example")

    def test_removed_results_do_not_create_a_negative_rate(self) -> None:
        output = StringIO()
        with (
            patch("reactions_generator.progress.monotonic", return_value=0) as clock,
            patch("tqdm.std.time", side_effect=lambda: clock()),
            RenderProgress(output) as progress,
        ):
            progress.report(SubmissionProgress(100, 0, 1))
            clock.return_value = 31
            progress.report(SubmissionProgress(99, 0, 2))
            clock.return_value = 62
            progress.report(SubmissionProgress(100, 0, 1))
        self.assertIn("99/101", output.getvalue())
        self.assertIn("62.00s/it", output.getvalue())


if __name__ == "__main__":
    unittest.main()
