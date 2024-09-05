import math

from enum import Enum


class Easing(Enum):
    EASE_IN_OUT_QUAD = "easeInOutQuad"
    EASE_IN_OUT_SIN = "easeInOutSin"


def interpolate(
    value: float,
    range_values: list[float],
    output_range: list[float],
    easing: Easing | None = None,
) -> float:
    if len(range_values) != len(output_range):
        raise ValueError("Range and output_range must have the same length")

    if value <= range_values[0]:
        return output_range[0]
    elif value >= range_values[-1]:
        return output_range[-1]

    for i in range(len(range_values) - 1):
        if range_values[i] <= value <= range_values[i + 1]:
            t = (value - range_values[i]) / (range_values[i + 1] - range_values[i])
            if easing == Easing.EASE_IN_OUT_QUAD:
                t = t * t * (3 - 2 * t)
            elif easing == Easing.EASE_IN_OUT_SIN:
                t = 0.5 - 0.5 * math.cos(t * math.pi)
            return output_range[i] + t * (output_range[i + 1] - output_range[i])

    raise ValueError("Value is outside the specified range")
