import io
import stat
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import ffmpeg
from PIL import Image

from reactions_generator import media, rendering
from reactions_generator.card import Card
from reactions_generator.encoding import make_encoder


class RenderingTests(unittest.TestCase):
    def test_remote_inputs_are_staged_and_cleaned_up(self) -> None:
        for error in (
            None,
            media.requests.exceptions.ConnectionError,
            RuntimeError,
            KeyboardInterrupt,
        ):
            with self.subTest(error=error):
                self.check_staged_inputs(error)

    def check_staged_inputs(self, error: type[BaseException] | None) -> None:
        with tempfile.TemporaryDirectory() as root:
            webcam = "https://example.com/camera.mp4?token=123"
            screen = "https://example.com/desktop.webm?token=456"
            sources = [webcam, screen]
            metadata = Mock(fps=30, duration=1.0, audio=True)
            response = MagicMock()
            response.__enter__.return_value = response
            download_fails = error is media.requests.exceptions.ConnectionError

            def download(source: str, **_kwargs: object) -> MagicMock:
                self.assertIn(source, sources)
                response.iter_content.return_value = iter([source.encode()])
                if download_fails and source == webcam:

                    def interrupted_download() -> Iterator[bytes]:
                        yield b"partial"
                        raise media.requests.exceptions.ConnectionError(
                            "lost connection"
                        )

                    response.iter_content.return_value = interrupted_download()
                return response

            def probe(source: str, **_kwargs: object) -> Mock:
                path = Path(source)
                self.assertEqual(path.parent.parent, Path(root))
                self.assertEqual(
                    (path.parent / "webcam.mp4").read_bytes(), webcam.encode()
                )
                self.assertEqual(
                    (path.parent / "screen.webm").read_bytes(), screen.encode()
                )
                return metadata

            with (
                patch.object(media.requests, "get", side_effect=download) as get,
                patch.object(
                    rendering, "get_metadata", side_effect=probe
                ) as get_metadata,
                patch.object(rendering, "render", side_effect=error) as render,
                patch.object(
                    rendering.ffmpeg, "input", wraps=rendering.ffmpeg.input
                ) as ffmpeg_input,
                patch.object(tempfile, "tempdir", root),
            ):

                def run() -> None:
                    rendering.render_reaction(
                        webcam_source=webcam,
                        screen_source=screen,
                        logo_source="#ffffff",
                        sound=True,
                    )

                if error is None:
                    run()
                else:
                    with self.assertRaises(error):
                        run()

            self.assertEqual(list(Path(root).iterdir()), [])
            if download_fails:
                get_metadata.assert_not_called()
                render.assert_not_called()
            else:
                self.assertEqual([call.args[0] for call in get.call_args_list], sources)
                self.assertTrue(
                    all(call.kwargs["stream"] for call in get.call_args_list)
                )
                inputs = [call.args[0] for call in ffmpeg_input.call_args_list]
                self.assertEqual(
                    sum(path.endswith("/webcam.mp4") for path in inputs), 2
                )
                self.assertEqual(
                    sum(path.endswith("/screen.webm") for path in inputs), 1
                )
                self.assertNotIn(webcam, inputs)
                self.assertNotIn(screen, inputs)

    def test_reaction_stops_card_stream_after_animation(self) -> None:
        metadata = Mock()
        metadata.fps = 30
        metadata.duration = 33.3
        metadata.audio = True

        with (
            patch.object(rendering, "get_metadata", return_value=metadata),
            patch.object(
                rendering,
                "load_image_or_color",
                return_value=Image.new("RGBA", (152, 152)),
            ),
            patch.object(rendering, "render") as render,
        ):
            rendering.render_reaction(
                webcam_source="webcam.mp4",
                screen_source="screen.mp4",
                background_source="background.png",
                sound=False,
            )

        card = render.call_args.kwargs["card"]
        original_last_frame = int(metadata.duration * metadata.fps)

        self.assertEqual(
            render.call_args.kwargs["last_frame"],
            min(original_last_frame, card.animation_start + 6),
        )
        self.assertLess(
            render.call_args.kwargs["last_frame"],
            original_last_frame,
        )

    def test_card_command_uses_the_card_instance_dimensions(self) -> None:
        with (
            patch.object(rendering, "render") as render,
            patch.object(rendering, "pipe_card_input") as pipe,
        ):
            rendering.render_card(logo_source="#ffffff", duration_seconds=0.1, fps=10)

        card = render.call_args.kwargs["card"]
        self.assertIsInstance(card, Card)
        pipe.assert_called_once_with(card.width, card.height, 10)
        self.assertEqual(render.call_args.kwargs["last_frame"], 1)

    def test_failed_encode_preserves_destination_and_removes_partial_output(
        self,
    ) -> None:
        self.check_render_result(returncode=1)

    def test_successful_encode_preserves_destination_permissions_and_releases_resources(
        self,
    ) -> None:
        self.check_render_result(returncode=0)

    def test_new_output_uses_normal_file_creation_permissions(self) -> None:
        self.check_render_result(returncode=0, existing=False)

    def test_interrupted_render_terminates_encoder_and_preserves_destination(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "result.mp4")
            output.write_bytes(b"previous video")
            process = Mock()
            process.stdin = io.BytesIO()
            process.poll.return_value = None
            card = Mock(spec=Card)
            card.render_frame.side_effect = KeyboardInterrupt
            source = ffmpeg.input(
                "pipe:", f="rawvideo", pix_fmt="rgba", s="2x2", r=1
            ).video
            with (
                patch.object(rendering.subprocess, "Popen", return_value=process),
                patch.object(rendering.atexit, "register"),
                patch.object(rendering.atexit, "unregister"),
                self.assertRaises(KeyboardInterrupt),
            ):
                rendering.render(
                    [source],
                    card=cast(Card, card),
                    last_frame=0,
                    output_path=str(output),
                    fps=1,
                    print_progress=False,
                    acodec=None,
                    encoding=make_encoder(),
                )

            self.assertEqual(output.read_bytes(), b"previous video")
            self.assertEqual(list(Path(directory).iterdir()), [output])
            process.terminate.assert_called_once_with()
            process.wait.assert_called_once_with()
            self.assertTrue(process.stdin.closed)

    def check_render_result(self, *, returncode: int, existing: bool = True) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "result.mp4")
            if existing:
                output.write_bytes(b"previous video")
                output.chmod(0o640)
                expected_mode = 0o640
            else:
                reference = Path(directory, "ordinary-file.mp4")
                reference.write_bytes(b"ordinary file")
                expected_mode = stat.S_IMODE(reference.stat().st_mode)
                reference.unlink()
            process = Mock()
            process.stdin = io.BytesIO()
            process.wait.return_value = returncode
            process.poll.return_value = returncode
            process.returncode = returncode
            card = Mock(spec=Card)
            card.render_frame.return_value = Image.new("RGBA", (2, 2), "white")
            reports: list[str] = []

            def start(command: list[str], **kwargs: object) -> Mock:
                reports.append(cast(dict[str, str], kwargs["env"])["FFREPORT"])
                Path(command[-1]).write_bytes(b"new video")
                stderr = kwargs.get("stderr")
                if isinstance(stderr, io.BufferedIOBase):
                    stderr.write(b"encoder failed")
                return process

            with (
                patch.object(rendering.subprocess, "Popen", side_effect=start),
                patch.object(rendering.atexit, "register") as register,
                patch.object(rendering.atexit, "unregister") as unregister,
            ):
                source = ffmpeg.input(
                    "pipe:", f="rawvideo", pix_fmt="rgba", s="2x2", r=1
                ).video

                def render() -> None:
                    rendering.render(
                        [source],
                        card=cast(Card, card),
                        last_frame=0,
                        output_path=str(output),
                        fps=1,
                        print_progress=False,
                        acodec=None,
                        encoding=make_encoder(),
                    )

                if returncode:
                    with self.assertRaises(ffmpeg.FFMpegExecuteError):
                        render()
                else:
                    render()

            self.assertEqual(
                output.read_bytes(), b"previous video" if returncode else b"new video"
            )
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), expected_mode)
            self.assertEqual(list(Path(directory).iterdir()), [output])
            self.assertEqual(len(reports), 1)
            self.assertTrue(reports[0].startswith("file=out/logs/result-"))
            self.assertTrue(reports[0].endswith(".log:level=32"))
            self.assertTrue(process.stdin.closed)
            process.terminate.assert_not_called()
            register.assert_called_once()
            unregister.assert_called_once_with(register.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
