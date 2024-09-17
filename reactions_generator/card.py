import functools
import math
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from reactions_generator.colors import Colors
from reactions_generator.interpolate import interpolate, Easing
from reactions_generator.utils import (
    Box,
    init_transparent_image,
    paste_with_alpha,
    center_anchor,
    place_grid,
    dimensions,
)
from reactions_generator.text_layout import (
    auto_resize_text,
)
from reactions_generator.fonts import (
    load_helvetica,
    load_helvetica_bold,
    load_monspaced,
)


@functools.cache
def load_font(font: ImageFont.FreeTypeFont, size: int) -> ImageFont.FreeTypeFont:
    return font.font_variant(size=size)


@functools.cache
def render_place(current_rank: int) -> Image.Image:
    padding = 16
    text = f"{current_rank}{get_ordinal(current_rank)} place"
    font = load_helvetica(40)
    (_, _, text_width, text_height) = font.getbbox(text)
    image = init_transparent_image(
        (math.ceil(text_width + padding * 2), math.ceil(text_height + padding * 2))
    )
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [(0, 0), (image.width, image.height)], radius=48, fill=Colors.light_gray
    )
    draw.text((padding, padding), text, font=font, fill=Colors.white)
    return image


def get_ordinal(n: int) -> str:
    ord = "th"
    if n % 10 == 1 and n % 100 != 11:
        ord = "st"
    elif n % 10 == 2 and n % 100 != 12:
        ord = "nd"
    elif n % 10 == 3 and n % 100 != 13:
        ord = "rd"
    return ord


def split_horizontal(
    box: Box, width: float, padding: float = 0, gap: float = 0
) -> tuple[Box, Box]:
    left, top, right, bottom = box
    left_box = (left + padding, top, left + padding + width, bottom)
    right_box = (left_box[2] + gap, top, right - padding, bottom)
    return left_box, right_box


def split_vertical(
    box: Box, height: float, padding: float = 0, gap: float = 0
) -> tuple[Box, Box]:
    left, top, right, bottom = box
    top_box = (left, top + padding, right, top + padding + height)
    bottom_box = (left, top_box[3] + gap, right, bottom - padding)
    return top_box, bottom_box


@dataclass(frozen=True)
class Card:
    title: str
    subtitle: str
    hashtag: str
    task: str
    time: float
    outcome: str
    success: bool
    rank_before: int
    rank_after: int
    logo: Image.Image
    animation_start: int
    fps: float
    width: int
    height: int

    top_padding = 48

    @property
    def actual_card_height(self) -> int:
        return self.height - self.top_padding

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @functools.cached_property
    def resized_logo(self) -> Image.Image:
        logo = self.logo.copy()
        factor = min(self.actual_card_height / logo.height, 152 / logo.width)
        width, height = map(lambda x: math.ceil(x * factor), logo.size)
        logo = logo.resize((width, height))
        logo.thumbnail((152, self.actual_card_height))
        return logo

    def time_to_string(self, frame: int) -> str:
        realtime = self.time + min(frame - self.animation_start, 0) / self.fps * 1000
        hours = int(realtime / (60 * 60 * 1000))
        minutes = int((realtime % (60 * 60 * 1000)) / (60 * 1000))
        seconds = int((realtime % (60 * 1000)) / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def outcome_animation(self, frame: int) -> str:
        frame_duration = 6
        outcome_states = ["   ", ".  ", ".. ", " ..", "  ."]
        if frame >= self.animation_start + 3:
            return self.outcome.rjust(len(outcome_states[0]))
        else:
            return outcome_states[(frame // frame_duration) % len(outcome_states)]

    def color_animation(
        self, frame: int
    ) -> tuple[int, int, int] | tuple[int, int, int, int]:
        blink_duration = 6
        cycle = [Colors.yellow, Colors.green]
        if frame >= self.animation_start:
            if self.success:
                return Colors.green
            else:
                return Colors.red
        if frame < self.animation_start - 8 * blink_duration:
            return Colors.yellow
        return cycle[((self.animation_start - frame) // blink_duration) % len(cycle)]

    def render_frame(
        self,
        frame: int,
    ) -> Image.Image:
        image = init_transparent_image(self.size)
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            [(0, self.top_padding), self.size],
            radius=48,
            fill=self.color_animation(frame),
        )

        rank = round(
            interpolate(
                frame,
                [self.animation_start - 15, self.animation_start + 5],
                [self.rank_before, self.rank_after],
            )
            if self.success
            else interpolate(
                frame,
                [
                    self.animation_start - 15,
                    self.animation_start,
                    self.animation_start + 6,
                ],
                [self.rank_before, 1, self.rank_before],
            ),
        )
        place_image = render_place(rank)
        place_position = (
            interpolate(
                frame,
                [self.animation_start - 15, self.animation_start + 5],
                [1, 0],
                easing=Easing.EASE_IN_OUT_QUAD,
            )
            if self.success
            else interpolate(
                frame,
                [
                    self.animation_start - 15,
                    self.animation_start,
                    self.animation_start + 6,
                ],
                [1, 0.7, 1],
                easing=Easing.EASE_IN_OUT_SIN,
            )
        )
        paste_with_alpha(
            image,
            place_image,
            (round((image.width - place_image.width) * place_position), 0),
        )

        logo_xy = 152
        logo_box, content_box = split_horizontal(
            (0, self.top_padding, self.width, self.height),
            width=logo_xy,
            padding=40,
            gap=32,
        )
        (title_box, task_box), (subtitle_box, status_box) = [
            split_horizontal(box, width=self.width / 2 - 16, gap=16)
            for box in split_vertical(
                content_box, height=self.height / 2, padding=16, gap=20
            )
        ]
        logo = self.resized_logo
        paste_with_alpha(
            image,
            logo,
            dest=(place_grid(center_anchor(logo_box, logo.size))[:2]),
        )

        paste_with_alpha(
            image,
            auto_resize_text(
                self.title,
                dimensions(title_box),
                load_helvetica_bold(10),
                allow_multiline=True,
                allow_compression=True,
                align_center=False,
            ),
            dest=(place_grid(title_box)[:2]),
        )
        paste_with_alpha(
            image,
            auto_resize_text(
                f"{self.subtitle} {self.hashtag}",
                dimensions(subtitle_box),
                load_helvetica(10),
                allow_multiline=False,
                allow_compression=True,
                align_center=False,
                max_size=32,
            ),
            dest=(place_grid(subtitle_box)[:2]),
        )
        paste_with_alpha(
            image,
            auto_resize_text(
                self.task,
                dimensions(task_box),
                load_helvetica_bold(10),
                allow_multiline=False,
                allow_compression=False,
                align_center=True,
            ),
            dest=(place_grid(task_box)[:2]),
        )
        paste_with_alpha(
            image,
            auto_resize_text(
                f"{self.outcome_animation(frame)} {self.time_to_string(frame)}",
                dimensions(status_box),
                load_monspaced(10),
                allow_multiline=False,
                allow_compression=False,
                align_center=True,
                max_size=32,
            ),
            dest=((place_grid(status_box)[0], place_grid(status_box)[1] - 5)),
        )

        return image
