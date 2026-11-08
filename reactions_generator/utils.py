import math
from typing import TypeAlias

from PIL import Image, ImageDraw

Size: TypeAlias = tuple[int, int]
Dimensions: TypeAlias = tuple[float, float]
Point: TypeAlias = tuple[int, int]
Box: TypeAlias = tuple[float, float, float, float]
PixelBox: TypeAlias = tuple[int, int, int, int]


def init_transparent_image(dimensions: Size) -> Image.Image:
    width, height = dimensions
    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


def rounded_mask(dimensions: Size) -> Image.Image:
    mask = Image.new("1", dimensions, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, mask.width, mask.height), radius=48, fill=1)
    return mask


def dimensions(box: Box) -> Dimensions:
    left, top, right, bottom = box
    return right - left, bottom - top


def place_grid(box: Box) -> PixelBox:
    return math.ceil(box[0]), math.ceil(box[1]), math.floor(box[2]), math.floor(box[3])


def paste_with_alpha(
    image: Image.Image, transparent_content: Image.Image, dest: Point
) -> Image.Image:
    transparent_content = transparent_content.convert("RGBA")
    image.paste(transparent_content, dest, mask=transparent_content)
    return image


def center_anchor(box: Box, dimensions: Dimensions) -> Box:
    width, height = dimensions
    left, top, right, bottom = box
    anchor_left = (left + right - width) / 2
    anchor_top = (top + bottom - height) / 2
    return anchor_left, anchor_top, anchor_left + width, anchor_top + height


def place_above(box: Box, dimensins: Dimensions, gap: float = 0) -> Box:
    width, height = dimensins
    left, top, right, _ = box
    return (
        (left + right - width) / 2,
        top - height - gap,
        (left + right + width) / 2,
        top - gap,
    )


def place_below(box: Box, dimensins: Dimensions, gap: float = 0) -> Box:
    width, height = dimensins
    left, _, right, bottom = box
    return (
        (left + right - width) / 2,
        bottom + gap,
        (left + right + width) / 2,
        bottom + height + gap,
    )
