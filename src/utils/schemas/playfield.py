import numpy as np
from pydantic import BaseModel

from utils.lines import Line
from utils.points import Point


class SnookerModel(BaseModel, arbitrary_types_allowed=True):
    pass


class PlayfieldLines(SnookerModel):
    top: Line = Line()
    bottom: Line = Line()
    left: Line = Line()
    right: Line = Line()

    @classmethod
    def from_lines(cls, top: Line, bottom: Line, side_lines: tuple[Line, Line]):
        left = [line for line in side_lines if line.slope < 0][0]
        right = [line for line in side_lines if line.slope > 0][0]
        return cls(top=top, bottom=bottom, left=left, right=right)


class PlayfieldPoints(SnookerModel):
    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    @classmethod
    def from_numpy(cls, lt: np.ndarray, lb:np.ndarray, rt:np.ndarray, rb:np.ndarray):
        return cls(
            top_left=Point.from_iterable(lt),
            top_right=Point.from_iterable(rt),
            bottom_right=Point.from_iterable(rb),
            bottom_left=Point.from_iterable(lb),
        )

    def to_numpy(self) -> np.ndarray:
        return np.array([
            np.array(self.top_left),
            np.array(self.top_right),
            np.array(self.bottom_right),
            np.array(self.bottom_left),
        ])


class Playfield(SnookerModel):
    lines: PlayfieldLines
    points: PlayfieldPoints