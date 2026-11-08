import io
import stat
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import ffmpeg
from PIL import Image

from reactions_generator import rendering
from reactions_generator.card import Card


class RenderingTests(unittest.TestCase):
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
                    vcodec="libx264",
                    acodec=None,
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

            def start(command: list[str], **kwargs: object) -> Mock:
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
                        vcodec="libx264",
                        acodec=None,
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
            self.assertTrue(process.stdin.closed)
            process.terminate.assert_not_called()
            register.assert_called_once()
            unregister.assert_called_once_with(register.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
