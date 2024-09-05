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


def draw_align_centre(
    draw: ImageDraw.ImageDraw, text: str, box: Box, font: ImageFont.FreeTypeFont
) -> None:
    place = place_grid(center_anchor(box, dimensions=font.getbbox(text)[2:]))
    draw.text(
        place[:2],
        text,
        font=font,
        fill=Colors.white,
    )


def draw_align_left(
    draw: ImageDraw.ImageDraw, text: str, box: Box, font: ImageFont.FreeTypeFont
) -> None:
    place = place_grid(center_anchor(box, dimensions=font.getbbox(text)[2:]))
    draw.text(
        (box[0], place[1]),
        text,
        font=font,
        fill=Colors.white,
    )


@functools.cache
def auto_resize_text(
    text: str,
    dimensions: tuple[float, float],
    font: ImageFont.FreeTypeFont,
    max_size: int,
) -> Image.Image:
    width, height = map(math.floor, dimensions)
    max_horizontal_compression = 1.5
    measure_size = 10
    _, _, measure_width, measure_height = font.font_variant(size=measure_size).getbbox(
        text
    )
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
    draw_align_left(
        draw,
        text,
        (0, 0, image.width, image.height),
        font=font.font_variant(size=font_size),
    )
    return image.resize((width, height))
