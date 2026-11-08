import unittest

from reactions_generator.fonts import load_font, load_regular
from reactions_generator.interpolate import Easing, interpolate
from reactions_generator.text_layout import auto_resize_text


class InterpolationTests(unittest.TestCase):
    def test_tuple_keyframes_clamp_and_interpolate(self) -> None:
        keyframes = (0, 10, 20)
        values = (10, 30, 20)
        self.assertEqual(interpolate(-1, keyframes, values), 10)
        self.assertEqual(interpolate(5, keyframes, values), 20)
        self.assertEqual(interpolate(15, keyframes, values), 25)
        self.assertEqual(interpolate(21, keyframes, values), 20)
        self.assertAlmostEqual(
            interpolate(5, keyframes, values, Easing.EASE_IN_OUT_SIN), 20
        )

    def test_invalid_keyframes_raise_clear_errors(self) -> None:
        for keyframes, values in (
            ((), ()),
            ((0, 10), (1,)),
            ((0, 0, 10), (1, 2, 3)),
            ((10, 0), (1, 2)),
        ):
            with self.subTest(keyframes=keyframes), self.assertRaises(ValueError):
                interpolate(5, keyframes, values)


class FontAndTextTests(unittest.TestCase):
    def test_font_candidates_fall_back_to_bundled_font(self) -> None:
        font = load_font(
            ("fonts/missing-custom-font.ttf", "fonts/nimbus/NimbusSans-Regular.ttf")
        )
        self.assertEqual(font.getname()[0], "Nimbus Sans")

    def test_empty_font_candidates_raise_oserror(self) -> None:
        with self.assertRaisesRegex(OSError, "No font candidates"):
            load_font(())

    def test_whitespace_renders_transparent_without_division_by_zero(self) -> None:
        for text in ("", " ", "\t\n"):
            with self.subTest(text=text):
                image = auto_resize_text(
                    text, (40, 20), load_regular(10), True, True, False
                )
                self.assertEqual(image.size, (40, 20))
                self.assertEqual(image.mode, "RGBA")
                self.assertIsNone(image.getbbox())

    def test_tiny_text_target_keeps_valid_font_size(self) -> None:
        image = auto_resize_text(
            "A very long university name", (1, 1), load_regular(10), False, True, False
        )
        self.assertEqual(image.size, (1, 1))


if __name__ == "__main__":
    unittest.main()
