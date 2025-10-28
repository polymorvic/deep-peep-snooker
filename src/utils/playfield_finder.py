import cv2
import matplotlib.pyplot as plt
import numpy as np

from .func import (binarize_playfield,
    find_playfield_exteral_borders, 
    blackout_pixels_outside_borders,
    find_playfield_internal_sideline_borders)
from .lines import Line
from .points import Point



class PlayfieldFinder:
    """
    Finds and stores key elements of a snooker playfield from an image.
    
    Attributes:
        img: The input image as a numpy array
        center: The center point of the image
        vertical_axis: Vertical line through the image center
        horizontal_axis: Horizontal line through the image center
    """

    def __init__(self, img: np.ndarray) -> None:
        """
        Initialize PlayfieldFinder with an image.
        
        Args:
            img: Image as a numpy array (height, width, channels)
        """
        self.img = img
        self.center = self._get_center_point()
        self.vertical_axis = self._create_vertical_axis()
        self.horizontal_axis = self._create_horizontal_axis()

    def _get_center_point(self) -> Point:
        """Get the center point of the image as a Point instance."""
        height, width = self.img.height, self.img.width
        center_x = width // 2
        center_y = height // 2
        return Point.from_xy(center_x, center_y)


    def _create_vertical_axis(self) -> Line:
        """Create a vertical line passing through the image center."""
        return Line(xv=self.center.x)


    def _create_horizontal_axis(self) -> Line:
        """Create a horizontal line passing through the image center."""
        return Line(slope=0, intercept=self.center.y)


    def find_side_cushions(self, slope_threshold: float = 0.1) -> tuple[Line, Line]:
        """
        Find the closest line on the left side and the closest line on the right side of the center.
        Excludes horizontal lines (slope close to zero).
        
        Args:
            slope_threshold: Lines with absolute slope less than this value are excluded (default: 0.1)
        
        Returns:
            Tuple containing (left_line, right_line) - the closest lines on each side
        """

        binary_mask, inv_binary_img = binarize_playfield(self.img)
        external_intersections, external_lines, _ = find_playfield_exteral_borders(self.img, binary_mask)
        blackout_img = blackout_pixels_outside_borders(external_intersections, inv_binary_img, external_lines)
        internal_lines, internal_boundaries_img = find_playfield_internal_sideline_borders(self.img, blackout_img)
        return internal_lines, internal_boundaries_img

