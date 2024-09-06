import functools
from dataclasses import dataclass

from PIL import Image, ImageDraw

from reactions_generator.utils import (
    paste_with_alpha,
    place_grid,
    center_anchor,
    place_above,
    place_below,
)


@functools.cache
def rounded_mask(dimensions: tuple[int, int]) -> Image.Image:
    mask = Image.new("1", dimensions, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, mask.width, mask.height), radius=48, fill=1)
    return mask


@dataclass(frozen=True)
class Reaction:
    success: bool
    header: Image.Image
    animation_start: int
    fps: float

    def render_frame(
        self,
        frame: int,
        background: Image.Image,
        card: Image.Image,
        webcam: Image.Image,
        screen: Image.Image,
    ) -> Image.Image:
        image = background.copy()

        gap = 50

        card_position = center_anchor(
            (0, 0, image.width, image.height), dimensions=card.size
        )

        image.paste(card, place_grid(card_position)[:2], card)
        paste_with_alpha(
            image,
            self.header,
            dest=place_grid(place_below((0, 0, image.width, 40), self.header.size))[:2],
        )
        image.paste(
            webcam,
            place_grid(place_above(card_position, webcam.size, gap=gap))[:2],
            mask=rounded_mask(webcam.size),
        )
        image.paste(
            screen,
            place_grid(place_below(card_position, screen.size, gap=gap))[:2],
            mask=rounded_mask(screen.size),
        )
        return image
