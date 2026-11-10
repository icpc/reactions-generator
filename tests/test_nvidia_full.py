"""Deterministic NVIDIA-command tests; a CUDA GPU is not needed."""

import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import ffmpeg
from PIL import Image

from reactions_generator import rendering
from reactions_generator.nvidia_full import (
    NvidiaFullSettings,
    build_nvidia_full_command,
)


class NvidiaTests(unittest.TestCase):
    def build(
        self,
        *,
        screen_is_video: bool = True,
        screen_source: str = "screen.mp4",
        has_webcam_audio: bool = True,
        sound: bool = True,
        background_position: tuple[float, float, float, float] = (0, 0, 1080, 1920),
        settings: NvidiaFullSettings | None = None,
    ) -> list[str]:
        return build_nvidia_full_command(
            webcam_source="webcam.mp4",
            screen_source=screen_source,
            screen_is_video=screen_is_video,
            background_source="background.png",
            action_audio_path="success.mp3",
            output_path="out.mp4",
            fps=30.0,
            duration=31.0,
            animation_start=30,
            has_webcam_audio=has_webcam_audio,
            sound=sound,
            background_position=background_position,
            webcam_position=(40.0, 194.5, 1000.0, 562.5),
            screen_position=(40.0, 1163.0, 1000.0, 562.5),
            card_position=(40, 807, 1000, 306),
            acodec="aac",
            print_progress=False,
            settings=settings or NvidiaFullSettings(),
        )

    def test_moving_inputs_stay_on_cuda_until_nvenc(self) -> None:
        command = self.build()
        graph = command[command.index("-filter_complex") + 1]
        self.assertEqual(command.count("-hwaccel_output_format"), 2)
        self.assertEqual(command.count("-hwaccel"), 2)
        self.assertEqual(command[command.index("-c:v") + 1], "h264_nvenc")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "cuda")
        self.assertIn("[0:v]scale_cuda=w=1000:h=562:format=yuv420p", graph)
        self.assertIn("[1:v]scale_cuda=w=1000:h=562:format=yuv420p", graph)
        self.assertIn("[3:v]format=yuva420p,hwupload", graph)
        self.assertEqual(graph.count("overlay_cuda="), 3)
        self.assertNotIn("hwdownload", graph)
        self.assertIn(
            "loop=loop=-1:size=1:start=0,settb=AVTB,setpts=N/(30*TB)[base]", graph
        )
        self.assertIn("[4:a]adelay=delays=1000:all=1[effect]", graph)
        self.assertIn("[0:a][effect]amix=inputs=2", graph)
        self.assertEqual(command[-1], "out.mp4")

    def test_static_screen_is_uploaded_and_looped_without_cuda_decode(self) -> None:
        command = self.build(screen_is_video=False, screen_source="A.png")
        graph = command[command.index("-filter_complex") + 1]
        self.assertEqual(command.count("-hwaccel_output_format"), 1)
        self.assertIn("[1:v]scale=1000:562", graph)
        self.assertIn("hwupload,loop=loop=-1:size=1:start=0", graph)
        self.assertNotIn("[1:v]scale_cuda", graph)

    def test_no_webcam_audio_uses_effect_audio_only(self) -> None:
        for has_audio, sound in ((False, True), (True, False)):
            with self.subTest(has_audio=has_audio, sound=sound):
                command = self.build(has_webcam_audio=has_audio, sound=sound)
                graph = command[command.index("-filter_complex") + 1]
                self.assertIn("[effect]anull[audio]", graph)
                self.assertNotIn("amix=", graph)

    def test_pixel_geometry_uses_even_chroma_alignment(self) -> None:
        command = self.build()
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("overlay_cuda=x=40:y=194", graph)
        self.assertIn("overlay_cuda=x=40:y=1162", graph)
        self.assertIn("overlay_cuda=x=40:y=806", graph)

    def test_invalid_nvidia_settings_are_rejected(self) -> None:
        for settings in (
            NvidiaFullSettings(device=-1),
            NvidiaFullSettings(cq=52),
            NvidiaFullSettings(cq=-1),
            NvidiaFullSettings(preset="slow"),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.build(settings=settings)
        with self.assertRaisesRegex(ValueError, "background"):
            self.build(background_position=(10, 0, 1080, 1920))

    def test_shared_cli_defaults_and_quality_mapping(self) -> None:
        from reactions_generator.nvidia_full import settings_from_cli

        self.assertEqual(
            settings_from_cli(quality=None, preset=None, vcodec=None),
            NvidiaFullSettings(),
        )
        self.assertEqual(
            settings_from_cli(quality="high", preset="p6", vcodec="h264_nvenc").cq, 19
        )
        with self.assertRaisesRegex(ValueError, "requires --vcodec"):
            settings_from_cli(quality="medium", preset=None, vcodec="libx264")
        with self.assertRaisesRegex(ValueError, "preset"):
            settings_from_cli(quality="medium", preset="veryfast", vcodec=None)


class NvidiaDispatchTests(unittest.TestCase):
    def test_full_dispatch_skips_software_graph_and_handles_signed_images(self) -> None:
        metadata = Mock(fps=30, duration=33.3, audio=True)
        for source in ("screen.mp4", "https://example.com/A.PNG?token=123"):
            with (
                self.subTest(source=source),
                patch.object(rendering, "get_metadata", return_value=metadata) as probe,
                patch.object(
                    rendering,
                    "load_image_or_color",
                    return_value=Image.new("RGBA", (152, 152)),
                ),
                patch.object(rendering, "run_card_command") as run,
                patch.object(rendering, "render") as software,
                patch.object(rendering.ffmpeg, "input") as software_input,
            ):
                rendering.render_reaction(hardware="nvidia_full", screen_source=source)
                command = run.call_args.kwargs["command"]
                self.assertIn(
                    "overlay_cuda", command[command.index("-filter_complex") + 1]
                )
                self.assertEqual(
                    run.call_args.kwargs["last_frame"],
                    run.call_args.kwargs["card"].animation_start + 6,
                )
                self.assertEqual(command[command.index("-preset") + 1], "p5")
                self.assertEqual(command[command.index("-cq") + 1], "23")
                self.assertEqual(probe.call_count, 2 if source.endswith("mp4") else 1)
                software.assert_not_called()
                software_input.assert_not_called()

    def test_regular_nvidia_keeps_existing_profile(self) -> None:
        with (
            patch.object(
                rendering,
                "get_metadata",
                return_value=Mock(fps=30, duration=33.3, audio=True),
            ),
            patch.object(
                rendering,
                "load_image_or_color",
                return_value=Image.new("RGBA", (152, 152)),
            ),
            patch.object(rendering, "render") as software,
            patch.object(rendering, "build_nvidia_full_command") as cuda,
        ):
            rendering.render_reaction(hardware="nvidia")
        cuda.assert_not_called()
        config = software.call_args.kwargs["encoding"]
        self.assertEqual(config.options["preset"], "p3")
        self.assertEqual(config.options["cq"], 24)

    def test_failed_cuda_encode_preserves_existing_file(self) -> None:
        process = Mock(stdin=io.BytesIO(), returncode=1)
        process.wait.return_value = 1
        process.poll.return_value = 1

        def start(command: list[str], **_kwargs: object) -> Mock:
            Path(command[-1]).write_bytes(b"partial")
            return process

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "result.mp4")
            output.write_bytes(b"original")
            with (
                patch.object(
                    rendering,
                    "get_metadata",
                    return_value=Mock(fps=30, duration=1, audio=True),
                ),
                patch.object(
                    rendering,
                    "load_image_or_color",
                    return_value=Image.new("RGBA", (152, 152)),
                ),
                patch.object(
                    rendering,
                    "Card",
                    return_value=Mock(
                        render_frame=Mock(return_value=Image.new("RGBA", (2, 2)))
                    ),
                ),
                patch.object(rendering.subprocess, "Popen", side_effect=start),
                self.assertRaises(ffmpeg.FFMpegExecuteError),
            ):
                rendering.render_reaction(
                    hardware="nvidia_full",
                    output_path=str(output),
                    print_progress=False,
                )
            self.assertEqual(output.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).iterdir()), [output])
            self.assertTrue(process.stdin.closed)

    def test_cuda_probe_requires_filters_and_working_runtime(self) -> None:
        from reactions_generator.nvidia_full import probe_nvidia_full

        with patch(
            "reactions_generator.nvidia_full.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "hwupload", ""),
        ):
            available, reason = probe_nvidia_full("webcam.mp4", NvidiaFullSettings())
        self.assertFalse(available)
        self.assertIn("overlay_cuda", reason)
        results = [
            subprocess.CompletedProcess([], 0, "scale_cuda overlay_cuda hwupload", ""),
            subprocess.CompletedProcess([], 0, "h264_nvenc", ""),
            subprocess.CompletedProcess([], 0, "format <pix_fmt>", ""),
            subprocess.CompletedProcess([], 1, "", "CUDA driver unavailable"),
        ]
        with patch(
            "reactions_generator.nvidia_full.subprocess.run", side_effect=results
        ) as run:
            available, reason = probe_nvidia_full("webcam.mp4", NvidiaFullSettings())
        self.assertFalse(available)
        self.assertIn("CUDA driver", reason)
        self.assertIn("-hwaccel_output_format", run.call_args.args[0])

    def test_old_scale_cuda_reports_format_requirement_before_runtime_test(
        self,
    ) -> None:
        from reactions_generator.nvidia_full import probe_nvidia_full

        results = [
            subprocess.CompletedProcess([], 0, "scale_cuda overlay_cuda hwupload", ""),
            subprocess.CompletedProcess([], 0, "h264_nvenc", ""),
            subprocess.CompletedProcess([], 0, "w <int> h <int>", ""),
        ]
        with patch(
            "reactions_generator.nvidia_full.subprocess.run", side_effect=results
        ) as run:
            available, reason = probe_nvidia_full("webcam.mp4", NvidiaFullSettings())
        self.assertFalse(available)
        self.assertIn("format option", reason)
        self.assertIn("Upgrade", reason)
        self.assertEqual(run.call_count, 3)


class NvidiaCliTests(unittest.TestCase):
    def test_all_hardware_commands_parse_full_cuda_selection(self) -> None:
        from typer.core import TyperGroup
        from typer.main import get_command
        from typer.testing import CliRunner

        from reactions_generator.cli import app

        for name, arguments in (
            ("reaction", []),
            ("single", ["https://example.com", "123"]),
            ("all", ["https://example.com"]),
            ("benchmark", []),
        ):
            with self.subTest(command=name):
                command = get_command(app)
                assert isinstance(command, TyperGroup)
                callback = Mock()
                command.commands[name].callback = callback
                with patch("typer.testing._get_command", return_value=command):
                    result = CliRunner().invoke(
                        app, [name, *arguments, "--hardware", "nvidia_full"]
                    )
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(callback.call_args.kwargs["hardware"], "nvidia_full")


if __name__ == "__main__":
    unittest.main()
