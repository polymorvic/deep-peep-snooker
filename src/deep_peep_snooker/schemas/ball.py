import numpy as np
from enum import StrEnum
from typing import Mapping

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


class BallsConfig(SnookerModel):
    radius_mm: float
    ranges: Mapping[StrEnum, BallHSVRange]

    def range_for(self, color: StrEnum) -> BallHSVRange:
        return self.ranges[color]
