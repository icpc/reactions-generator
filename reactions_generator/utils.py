import math

from PIL import Image


def init_transparent_image(dimensions: tuple[int, int]) -> Image.Image:
    width, height = dimensions
    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


Box = tuple[float, float, float, float]


def dimensions(box: Box) -> tuple[float, float]:
    left, top, right, bottom = box
    return right - left, bottom - top


def place_grid(box: Box) -> tuple[int, int, int, int]:
    return math.ceil(box[0]), math.ceil(box[1]), math.floor(box[2]), math.floor(box[3])


def paste_with_alpha(
    image: Image.Image, content: Image.Image, dest: tuple[int, int]
) -> Image.Image:
    image.paste(content, dest, mask=content)
    return image


def center_anchor(box: Box, dimensions: tuple[float, float]) -> Box:
    width, height = dimensions
    left, top, right, bottom = box
    archor_left = (left + right - width) / 2
    archor_top = (top + bottom - height) / 2
    return archor_left, archor_top, archor_left + width, archor_top + height


def place_above(box: Box, dimensins: tuple[float, float], gap: float = 0) -> Box:
    width, height = dimensins
    left, top, right, _ = box
    return (
        (left + right - width) / 2,
        top - height - gap,
        (left + right + width) / 2,
        top - gap,
    )


def place_below(box: Box, dimensins: tuple[float, float], gap: float = 0) -> Box:
    width, height = dimensins
    left, _, right, bottom = box
    return (
        (left + right - width) / 2,
        bottom + gap,
        (left + right + width) / 2,
        bottom + height + gap,
    )
