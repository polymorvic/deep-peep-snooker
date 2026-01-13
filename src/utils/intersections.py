from typing import TYPE_CHECKING, Self

import numpy as np

from .common import Hashable, array_like
from .lines import transform_line
from .points import transform_point

if TYPE_CHECKING:
    from .lines import Line
    from .points import Point


class Intersection(Hashable):
    """
    Represents the intersection point of two lines and the angle between them.
    """

    def __init__(self, line1: "Line", line2: "Line", intersection_point: "Point") -> None:
        """
        Initialize the Intersection object.

        Args:
            line1 (Line): The first line.
            line2 (Line): The second line.
            intersection_point (tuple[int, int]): The (x, y) coordinates of the intersection point.
        """
        self.line1 = line1
        self.line2 = line2
        self.point = intersection_point
        self.angle = self._compute_angle(self.line1, self.line2)

    def __repr__(self) -> str:
        """
        Returns:
            str: Returns a string representation of the intersection point and both lines.
            Lines are shown in order of slope (lower slope first).
        """

        def format_line(line: "Line") -> str:
            """Helper function to format a line equation."""
            if line.xv is not None:
                return f"x = {line.xv:.2f}"
            else:
                return f"y = {line.slope:.2f} * x + {line.intercept:.2f}"

        lines = [self.line1, self.line2]
        lines.sort(key=lambda line: line.slope if line.slope is not None else np.inf)

        line1_eq = format_line(lines[0])
        line2_eq = format_line(lines[1])

        return f"Point {self.point} line1: [{line1_eq}] line2: [{line2_eq}]"

    def _key_(self) -> tuple["Point", tuple[float, float]]:
        """
        Returns a tuple of identifying attributes used for hashing and equality comparison.
        Links the intersection point with both lines for unique identification.

        Returns:
            tuple: A tuple containing the point coordinates and the keys of both lines,
                sorted by slope (lower slope first, vertical lines last) to ensure consistent ordering.
        """

        def sort_key(line: "Line") -> tuple[float, float]:
            primary = line.slope if line.slope is not None else np.inf
            secondary = line.xv if line.xv is not None else -np.inf
            return (primary, secondary)

        lines = [self.line1, self.line2]
        lines.sort(key=sort_key)

        line_keys = [line._key_() for line in lines]
        return (self.point, tuple(line_keys))

    def distance(self, another_intersection: Self) -> float:
        """
        Calculate the Euclidean distance to another intersection point.

        Args:
            another_intersection (Intersection): Another intersection to compute distance to.

        Returns:
            float: The Euclidean distance.
        """
        return self.point.distance(another_intersection.point)

    def other_line(self, used: "Line") -> "Line":
        """
        Return the line from this intersection that is NOT `used`.
        Raises ValueError if `used` doesn't belong to this intersection.
        """
        if self.line1 is used or self.line1._key_() == used._key_():
            return self.line2
        if self.line2 is used or self.line2._key_() == used._key_():
            return self.line1
        raise ValueError("The provided line does not belong to this intersection.")

    def _compute_angle(self, line1: "Line", line2: "Line") -> float:
        """Compute the angle in degrees between two Line objects.

        Args:
            line1 (Line): First line.
            line2 (Line): Second line.

        Returns:
            float: Angle in degrees between the two lines.
        """
        if line1.xv is None and line2.xv is not None:
            angle = 90 - line1.theta
        elif line1.xv is not None and line2.xv is None:
            angle = 90 - line2.theta
        elif line1.slope * line2.slope == -1:
            angle = 90
        else:
            angle = np.rad2deg(np.arctan((line2.slope - line1.slope) / (1 + line1.slope * line2.slope)))

        return angle + 180


def compute_intersections(lines: list['Line'], image: array_like) -> list[Intersection]:
    """
    Compute all intersection points between pairs of lines within image boundaries.
    
    This function finds all valid intersection points between every pair of lines
    in the provided list that lie within the image boundaries. Each pair of lines
    is checked only once to avoid duplicates.
    
    Args:
        lines: List of Line objects to find intersections between
        image: Image array used to determine valid intersection boundaries
    
    Returns:
        List of Intersection objects representing valid intersection points
    
    Note:
        Only intersections within the image boundaries are included. Each pair
        of lines is processed only once (no duplicates).
    """
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            intersection = lines[i].intersection(lines[j], image)
            if intersection is not None:
                intersections.append(intersection)
    return intersections


def transform_intersection(
    intersection: Intersection,
    source_img: np.ndarray,
    original_x_start: int,
    original_y_start: int,
    to_global: bool = True,
    ) -> Intersection:
    """
    Transforms an Intersection in one go.
    - If to_global=True: treats inputs as LOCAL and returns GLOBAL.
    - If to_global=False: treats inputs as GLOBAL and returns LOCAL.

    Note:
        `source_img` should be the image in the *source* space,
        i.e. the space you are transforming FROM. This keeps `limit_to_img`
        correct in both directions.
    """
    transformed_point = transform_point(intersection.point, original_x_start, original_y_start, to_global=to_global)
    line1_t = transform_line(intersection.line1, source_img, original_x_start, original_y_start, to_global)
    line2_t = transform_line(intersection.line2, source_img, original_x_start, original_y_start, to_global)
    return Intersection(line1_t, line2_t, transformed_point)