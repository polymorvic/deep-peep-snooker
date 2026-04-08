import numpy as np
from enum import StrEnum

from deep_peep_snooker.schemas.base import SnookerModel


class BallColor(StrEnum):
    BLUE = 'blue'
    YELLOW = 'yellow'
    GREEN = 'green'
    BROWN = 'brown'
    PINK = 'pink'
    RED = 'red'
    CUE = 'cue'
    BLACK = 'black'


class BallHSVRange(SnookerModel):
    lower_bound: np.ndarray
    upper_bound: np.ndarray


class BallRanges(SnookerModel):
    blue: BallHSVRange
    yellow: BallHSVRange
    green: BallHSVRange
    brown: BallHSVRange
    pink: BallHSVRange
    red: BallHSVRange
    cue: BallHSVRange
    black: BallHSVRange


class BallsConfig(SnookerModel):
    radius_mm: float
    ranges: BallRanges

    def range_for(self, color: BallColor) -> BallHSVRange:
        return getattr(self.ranges, color.value)
