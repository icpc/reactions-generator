import tempfile
import unittest
from collections.abc import Iterator
from concurrent.futures import CancelledError
from fractions import Fraction
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, Mock, patch

from reactions_generator import media


class MetadataTests(unittest.TestCase):
    def test_invalid_nominal_rate_and_missing_stream_duration_use_fallbacks(
        self,
    ) -> None:
        probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "r_frame_rate": "0/0",
                    "avg_frame_rate": "30000/1001",
                },
                {"codec_type": "audio"},
            ],
            "format": {"duration": "1.25"},
        }
        with patch.object(media.ffmpeg, "probe", return_value=probe):
            metadata = media.direct_ffprobe("video.mp4")

        self.assertEqual(metadata.fps, Fraction(30000, 1001))
        self.assertEqual(metadata.duration, 1.25)
        self.assertTrue(metadata.audio)

    def test_audio_only_input_has_a_useful_failure(self) -> None:
        with (
            patch.object(
                media.ffmpeg,
                "probe",
                return_value={"streams": [{"codec_type": "audio"}]},
            ),
            self.assertRaises(ValueError),
        ):
            media.direct_ffprobe("audio.mp3")

    def test_ffmpeg_fallback_preserves_fractional_seconds(self) -> None:
        graph = Mock()
        graph.output.return_value.run.return_value = (
            b"",
            b"Audio: aac\nframe=   30 fps=0.0 time=00:00:01.25 bitrate=0.0kbits/s\n",
        )
        with patch.object(media.ffmpeg, "input", return_value=graph):
            metadata = media.metadata_via_ffmpeg("video.mp4")

        self.assertEqual(metadata.duration, 1.25)
        self.assertEqual(metadata.fps, 24)
        self.assertTrue(metadata.audio)


class DownloadTests(unittest.TestCase):
    def test_both_downloads_start_and_failure_cancels_the_other(self) -> None:
        for failed in ("webcam", "screen"):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as root:
                self.check_failed_pair(Path(root), failed)

    def check_failed_pair(self, directory: Path, failed: str) -> None:
        started = {"webcam": Event(), "screen": Event()}
        stopped = Event()

        def download(source: str, directory: Path, name: str, cancelled: Event) -> str:
            started[name].set()
            peer = "screen" if name == "webcam" else "webcam"
            self.assertTrue(started[peer].wait(3), "Downloads did not overlap")
            if name == failed:
                raise ConnectionError(f"{name} failed")
            self.assertTrue(cancelled.wait(3), "Peer download was not cancelled")
            stopped.set()
            raise CancelledError

        with (
            patch.object(media, "stage_media", side_effect=download),
            self.assertRaisesRegex(ConnectionError, f"{failed} failed"),
        ):
            media.stage_reaction_media("webcam.mp4", "screen.mp4", directory)
        self.assertTrue(stopped.is_set())

    def test_cancelled_stream_closes_response_before_writing_more_data(self) -> None:
        cancelled = Event()
        response = MagicMock()
        response.__enter__.return_value = response

        def chunks(*, chunk_size: int) -> Iterator[bytes]:
            yield b"first chunk"
            cancelled.set()
            yield b"discarded chunk"
            self.fail("Cancelled download continued consuming data")

        response.iter_content.side_effect = chunks
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            with (
                patch.object(media.requests, "get", return_value=response),
                self.assertRaises(CancelledError),
            ):
                media.stage_media(
                    "https://example.com/video.mp4", directory, "webcam", cancelled
                )
            self.assertEqual((directory / "webcam.mp4").read_bytes(), b"first chunk")
            response.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
