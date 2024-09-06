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
    draw_align_centre,
    draw_align_left,
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


target_card_width = 1000
target_card_height = 260
target_card_padding_top = 48


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

    @functools.cached_property
    def resized_logo(self) -> Image.Image:
        logo = self.logo.copy()
        logo.thumbnail((152, target_card_height))
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
        image = init_transparent_image(
            (target_card_width, target_card_height + target_card_padding_top)
        )
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle(
            [(0, target_card_padding_top), (image.width, image.height)],
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
            (0, target_card_padding_top, image.width, image.height),
            width=logo_xy,
            padding=40,
            gap=32,
        )
        (title_box, task_box), (subtitle_box, status_box) = [
            split_horizontal(box, width=552)
            for box in split_vertical(content_box, height=150, padding=16, gap=16)
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
                self.title, dimensions(title_box), load_helvetica_bold(10), max_size=150
            ),
            dest=(place_grid(title_box)[:2]),
        )
        draw_align_left(
            draw,
            f"{self.subtitle} {self.hashtag}",
            subtitle_box,
            font=load_helvetica(32),
        )
        draw_align_centre(
            draw,
            self.task,
            task_box,
            font=load_helvetica_bold(150),
        )
        draw_align_centre(
            draw,
            f"{self.outcome_animation(frame)} {self.time_to_string(frame)}",
            status_box,
            font=load_monspaced(28),
        )

        return image
