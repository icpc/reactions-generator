import functools

from PIL import ImageFont


@functools.cache
def load_font(font: str, fallback: str) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font)
    except IOError:
        return ImageFont.truetype(fallback)


@functools.cache
def load_helvetica(size: int) -> ImageFont.FreeTypeFont:
    return load_font("Helvetica", "helvetica.ttf").font_variant(size=size)


@functools.cache
def load_helvetica_bold(size: int) -> ImageFont.FreeTypeFont:
    return load_font("Helvetica", "helvetica.ttf").font_variant(size=size, index=1)


@functools.cache
def load_monspaced(size: int) -> ImageFont.FreeTypeFont:
    return load_font("Courier", "Courier").font_variant(size=size)
