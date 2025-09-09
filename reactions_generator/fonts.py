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
        ("./fonts/custom/helvetica.ttf", "./fonts/nimbus/NimbusSans-Regular.ttf")
    ).font_variant(size=size)


@functools.cache
def load_bold(size: int) -> ImageFont.FreeTypeFont:
    return load_font(
        ("./fonts/custom/helvetica_bold.ttf", "./fonts/nimbus/NimbusSans-Bold.ttf")
    ).font_variant(size=size)


@functools.cache
def load_monspaced(size: int) -> ImageFont.FreeTypeFont:
    return load_font(
        ("./fonts/custom/cour.ttf", "./fonts/nimbus/NimbusMonoPS-Regular.ttf")
    ).font_variant(size=size)
