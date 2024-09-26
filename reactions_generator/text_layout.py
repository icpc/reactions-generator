import functools
import math

from PIL import Image, ImageDraw, ImageFont

from reactions_generator.colors import Colors
from reactions_generator.utils import (
    Box,
    center_anchor,
    place_grid,
    init_transparent_image,
)


def measure(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, multiline: bool
) -> tuple[float, float]:
    if multiline:
        return draw.multiline_textbbox((0, 0), text, font=font)[2:]
    else:
        return draw.textbbox((0, 0), text, font=font)[2:]


def draw_align_centre(
    draw: ImageDraw.ImageDraw, text: str, box: Box, font: ImageFont.FreeTypeFont
) -> None:
    dimensions = measure(draw, text, font, False)
    place = place_grid(center_anchor(box, dimensions))
    draw.text(
        place[:2],
        text,
        font=font,
        fill=Colors.white,
    )


def draw_align_left(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    font: ImageFont.FreeTypeFont,
    multiline: bool = False,
) -> None:
    dimensions = measure(draw, text, font, multiline)
    place = place_grid(center_anchor(box, dimensions))
    xy = (box[0], place[1])
    if multiline:
        draw.multiline_text(xy, text, font=font, fill=Colors.white)
    else:
        draw.text(xy, text, font=font, fill=Colors.white)


def try_adding_endline(text: str) -> tuple[bool, str]:
    if len(text) < 15:
        return False, text

    middle = len(text) // 2
    best_position = -1

    for i in range(len(text)):
        if text[i] == " ":
            if abs(i - middle) < abs(best_position - middle):
                best_position = i

    if best_position == -1:
        return False, text

    return True, text[:best_position] + "\n" + text[best_position + 1 :]


@functools.cache
def auto_resize_text(
    text: str,
    dimensions: tuple[float, float],
    font: ImageFont.FreeTypeFont,
    allow_multiline: bool,
    allow_compression: bool,
    align_center: bool,
    max_size: int = 150,
) -> Image.Image:
    width, height = map(math.floor, dimensions)
    max_horizontal_compression = 1.5 if allow_compression else 1
    measure_size = 10

    if text == "":
        return init_transparent_image((width, height))

    multiline, text = try_adding_endline(text) if allow_multiline else (False, text)

    measure_width, measure_height = measure(
        ImageDraw.Draw(Image.new("1", (1, 1))),
        text,
        font.font_variant(size=measure_size),
        multiline,
    )
    # Let's add double of the measure error to our dimensions just to be sure
    measure_width *= 1 + 2 / 64 
    measure_height *= 1 + 2 / 64
    width_ratio = measure_width / measure_size
    height_ratio = measure_height / measure_size

    font_size = math.floor(
        min(
            width / width_ratio * max_horizontal_compression,
            height / height_ratio,
            max_size,
        )
    )
    image = init_transparent_image(
        (
            max(width, math.floor(font_size * width_ratio)),
            max(height, math.floor(font_size * height_ratio)),
        )
    )
    draw = ImageDraw.Draw(image)
    if align_center:
        draw_align_centre(
            draw,
            text,
            (0, 0, image.width, image.height),
            font=font.font_variant(size=font_size),
        )
    else:
        draw_align_left(
            draw,
            text,
            (0, 0, image.width, image.height),
            font=font.font_variant(size=font_size),
        )
    return image.resize((width, height))
