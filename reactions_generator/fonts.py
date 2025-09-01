import functools

from PIL import ImageFont


@functools.cache
def load_font(fonts: tuple[str]) -> ImageFont.FreeTypeFont:
    exception = Exception("No font found")
    for font in fonts:
        try:
            return ImageFont.truetype(font)
        except IOError as e:
            exception = e
    raise exception


@functools.cache
def load_regular(size: int) -> ImageFont.FreeTypeFont:
    return load_font(
        ("Helvetica", "helvetica.ttf", "./helvetica.ttf", "./fonts/helvetica.ttf")
    ).font_variant(size=size)


@functools.cache
def load_bold(size: int) -> ImageFont.FreeTypeFont:
    return load_font(
        (
            "Helvetica Bold",
            "helvetica-bold.ttf",
            "./helvetica_bold.ttf",
            "./fonts/helvetica_bold.ttf",
        )
    ).font_variant(size=size)


@functools.cache
def load_monspaced(size: int) -> ImageFont.FreeTypeFont:
    return load_font(
        ("Courier", "cour.ttf", "./cour.ttf", "./fonts/cour.ttf")
    ).font_variant(size=size)
