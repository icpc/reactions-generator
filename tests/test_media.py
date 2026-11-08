import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_downloaded_probe_input_is_removed_after_success_or_failure(self) -> None:
        for fails in (False, True):
            with self.subTest(fails=fails):
                self.check_downloaded_probe(fails=fails)

    def check_downloaded_probe(self, *, fails: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            response = Mock(content=b"downloaded video")
            paths: list[Path] = []
            expected = media.Metadata(Fraction(30), 1.0, False)

            def probe(source: str) -> media.Metadata:
                path = Path(source)
                paths.append(path)
                self.assertEqual(path.read_bytes(), response.content)
                if fails:
                    raise ValueError("unsupported video")
                return expected

            with (
                patch.object(media.requests, "get", return_value=response) as get,
                patch.object(media, "direct_ffprobe", side_effect=probe),
                patch.object(media.tempfile, "tempdir", directory),
            ):
                if fails:
                    with self.assertRaises(ValueError):
                        media.download_and_ffprobe("https://example.com/video.mp4")
                else:
                    self.assertEqual(
                        media.download_and_ffprobe("https://example.com/video.mp4"),
                        expected,
                    )
            self.assertEqual(len(paths), 1)
            self.assertFalse(paths[0].exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(get.call_args.kwargs["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
