import math

from PIL import Image, ImageDraw


def init_transparent_image(dimensions: tuple[int, int]) -> Image.Image:
    width, height = dimensions
    return Image.new("RGBA", (width, height), (0, 0, 0, 0))


def rounded_mask(dimensions: tuple[int, int]) -> Image.Image:
    mask = Image.new("1", dimensions, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, mask.width, mask.height), radius=48, fill=1)
    return mask


Box = tuple[float, float, float, float]


def dimensions(box: Box) -> tuple[float, float]:
    left, top, right, bottom = box
    return right - left, bottom - top


def place_grid(box: Box) -> tuple[int, int, int, int]:
    return math.ceil(box[0]), math.ceil(box[1]), math.floor(box[2]), math.floor(box[3])


def paste_with_alpha(
    image: Image.Image, transparent_content: Image.Image, dest: tuple[int, int]
) -> Image.Image:
    transparent_content = transparent_content.convert("RGBA")
    image.paste(transparent_content, dest, mask=transparent_content)
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
