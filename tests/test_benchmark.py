import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reactions_generator import benchmark as benchmark_module


class BenchmarkTests(unittest.TestCase):
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
                render.call_count, 1 + 8 + 8
            )  # warm-up and two complete batches
            report = json.loads(Path(directory, "results.json").read_text())
            self.assertEqual([r["workers"] for r in report["results"]], [1, 2])
            self.assertTrue(all(r["successful"] == 8 for r in report["results"]))
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
