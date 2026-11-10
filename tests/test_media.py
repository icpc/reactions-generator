import unittest
from fractions import Fraction
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


if __name__ == "__main__":
    unittest.main()
