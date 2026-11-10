import functools

from PIL import ImageFont


@functools.lru_cache(maxsize=128)
def load_font(fonts: tuple[str, ...]) -> ImageFont.FreeTypeFont:
    """Load the first available font, preserving the custom-font preference."""
    exception = OSError("No font candidates provided")
    for font in fonts:
        try:
            return ImageFont.truetype(font)
        except OSError as e:
            exception = e
    raise exception


@functools.lru_cache(maxsize=128)
def load_regular(size: int) -> ImageFont.FreeTypeFont:
    return load_font(
        ("./fonts/custom/helvetica.ttf", "./fonts/nimbus/NimbusSans-Regular.ttf")
    ).font_variant(size=size)


@functools.lru_cache(maxsize=128)
def load_bold(size: int) -> ImageFont.FreeTypeFont:
    return load_font(
        ("./fonts/custom/helvetica_bold.ttf", "./fonts/nimbus/NimbusSans-Bold.ttf")
    ).font_variant(size=size)


@functools.lru_cache(maxsize=128)
def load_monspaced(size: int) -> ImageFont.FreeTypeFont:
    return load_font(
        ("./fonts/custom/cour.ttf", "./fonts/nimbus/NimbusMonoPS-Regular.ttf")
    ).font_variant(size=size)
