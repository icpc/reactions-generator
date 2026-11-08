import subprocess
import unittest
from unittest.mock import patch

from reactions_generator.encoding import CODECS, QUALITY, make_encoder, probe_encoder


class EncodingTests(unittest.TestCase):
    def test_default_cpu(self) -> None:
        config = make_encoder()
        self.assertEqual(config.codec, "libx264")
        self.assertEqual(config.options, {"crf": 23, "preset": "veryfast"})

    def test_mappings(self) -> None:
        for backend, codec in CODECS.items():
            if backend == "vaapi":
                device = "/dev/dri/renderD128"
            else:
                device = None
            for quality in ("low", "balanced", "high"):
                with self.subTest(backend=backend, quality=quality):
                    config = make_encoder(backend, quality, vaapi_device=device)
                    self.assertEqual(config.codec, codec)
                    expected = QUALITY[backend][quality]
                    key = {
                        "cpu": "crf",
                        "nvidia": "cq",
                        "intel": "global_quality",
                        "amd": "qp_i",
                        "apple": "q:v",
                        "vaapi": "qp",
                    }[backend]
                    self.assertEqual(config.options[key], expected)

    def test_explicit_hardware_and_codec_conflict(self) -> None:
        with self.assertRaises(ValueError):
            make_encoder("nvidia", vcodec="libx264")

    def test_bad_preset(self) -> None:
        with self.assertRaises(ValueError):
            make_encoder("nvidia", preset="veryfast")

    def test_no_fallback_on_probe_failure(self) -> None:
        failure = subprocess.CompletedProcess([], 1, "", "no NVENC capable device")
        with patch("reactions_generator.encoding.subprocess.run", return_value=failure):
            success, reason = probe_encoder(make_encoder("nvidia"))
        self.assertFalse(success)
        self.assertIn("NVENC", reason)

    def test_vaapi_device_flags(self) -> None:
        config = make_encoder("vaapi", vaapi_device="/dev/dri/renderD129")
        self.assertEqual(
            config.device_args(),
            [
                "-init_hw_device",
                "vaapi=rg_vaapi:/dev/dri/renderD129",
                "-filter_hw_device",
                "rg_vaapi",
            ],
        )


if __name__ == "__main__":
    unittest.main()
