import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reactions_generator import benchmark as benchmark_module


class BenchmarkTests(unittest.TestCase):
    def test_prepare_clip_preserves_codecs_and_limits_duration(self) -> None:
        with patch.object(benchmark_module.subprocess, "run") as run:
            self.assertEqual(
                benchmark_module.prepare_clip("source.webm", Path("short.mkv"), 5),
                "short.mkv",
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-c") + 1], "copy")
        self.assertEqual(command[command.index("-t") + 1], "5")
        with patch.object(benchmark_module.subprocess, "run") as run:
            self.assertEqual(
                benchmark_module.prepare_clip("screen.png", Path("short.mkv"), 5),
                "screen.png",
            )
            run.assert_not_called()

    def test_invalid_workload_fails_before_rendering(self) -> None:
        for duration in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                benchmark_module.benchmark("cpu", duration_seconds=duration)
        with self.assertRaises(ValueError):
            benchmark_module.benchmark("cpu", jobs_per_worker=0)

    def test_worker_counts(self) -> None:
        self.assertEqual(benchmark_module.parse_worker_counts("1,2,4,8"), [1, 2, 4, 8])

    def test_invalid_counts(self) -> None:
        for value in ("", "1,", "0", "-1,2", "1,1", "one", "2.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark_module.parse_worker_counts(value)

    def test_full_pipeline_jobs_and_report(self) -> None:
        def fake_render(**kwargs: object) -> None:
            Path(str(kwargs["output_path"])).write_bytes(b"encoded")

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(benchmark_module.Path, "is_file", return_value=True),
                patch.object(
                    benchmark_module, "probe_encoder", return_value=(True, "ok")
                ),
                patch.object(
                    benchmark_module, "render_reaction", side_effect=fake_render
                ) as render,
                patch.object(benchmark_module, "inspect_output", return_value=8.0),
                patch.object(
                    benchmark_module.subprocess,
                    "run",
                    return_value=Mock(stdout="ffmpeg version test\n"),
                ),
            ):
                benchmark_module.benchmark("cpu", "1,2", output_directory=directory)
            self.assertEqual(
                render.call_count, 1 + 2 + 2
            )  # warm-up and two complete batches
            report = json.loads(Path(directory, "results.json").read_text())
            self.assertEqual([r["workers"] for r in report["results"]], [1, 2])
            self.assertTrue(all(r["successful"] == 2 for r in report["results"]))
            self.assertIsNotNone(report["recommended"])
            self.assertEqual(
                sorted(p.name for p in Path(directory).iterdir()), ["results.json"]
            )

    def test_unavailable_encoder_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(benchmark_module.Path, "is_file", return_value=True),
                patch.object(
                    benchmark_module, "probe_encoder", return_value=(False, "no GPU")
                ),
                patch.object(benchmark_module, "render_reaction") as render,
                patch.object(
                    benchmark_module.subprocess,
                    "run",
                    return_value=Mock(stdout="ffmpeg version test\n"),
                ),
            ):
                benchmark_module.benchmark("nvidia", "1", output_directory=directory)
            render.assert_not_called()
            report = json.loads(Path(directory, "results.json").read_text())
            self.assertIsNone(report["recommended"])
            self.assertEqual(report["results"][0]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
