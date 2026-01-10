import numpy as np

from .func import (binarize_playfield,
    find_playfield_exteral_borders, 
    blackout_pixels_outside_borders,
    find_playfield_internal_sideline_borders, 
    find_top_internal_cushion, find_baulk_line, 
    find_bottom_internal_cushion)
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
        return find_playfield_internal_sideline_borders(blackout_img)


    def find_top_internal_cushion(self) -> Line | None:
        """
        Find the top internal cushion of the playfield.
        
        The method processes the image to isolate the playfield area and then detects
        the top horizontal cushion (the top boundary of the playing area). It uses
        probabilistic Hough line transformation to detect near-horizontal lines and
        selects the one positioned highest in the image (lowest intercept value).
        
        Process:
            1. Binarize the image to isolate the playfield area
            2. Detect external borders of the playfield
            3. Create a blackout image with pixels outside borders removed
            4. Detect line segments in the blackout image using Hough transform
            5. Filter for near-horizontal lines (abs(slope) < 2)
            6. Sort lines by intercept value (ascending)
            7. Return the line with the lowest intercept (highest position)
        
        Returns:
            Line | None: Line object representing the top internal cushion (the line
                        with the lowest intercept value), or None if no suitable lines
                        are found
        
        Note:
            The top cushion is typically a near-horizontal line that defines the upper
            boundary of the playing area. The method filters out vertical and highly
            sloped lines to focus on horizontal boundaries.
        """
        binary_mask, inv_binary_img = binarize_playfield(self.img)
        external_intersections, external_lines, _ = find_playfield_exteral_borders(self.img, binary_mask)
        blackout_img = blackout_pixels_outside_borders(external_intersections, inv_binary_img, external_lines)
        return find_top_internal_cushion(blackout_img)


    def find_baulk_line(self) -> Line | None:
        """
        Find the baulk line of the playfield.
        
        Detects the baulk line by processing the image to isolate the playfield area
        and selecting the near-horizontal line with intercept closest to the center.
        
        Returns:
            Line | None: Line object representing the baulk line in global coordinates,
                        or None if no suitable line is found
        """
        binary_mask, _ = binarize_playfield(self.img) 
        external_intersections, _, _ = find_playfield_exteral_borders(self.img, binary_mask)
        intersection_points = np.array([[int(inter.point.x), int(inter.point.y)] for inter in external_intersections])
        return find_baulk_line(self.img, intersection_points)


    def find_bottom_internal_cushion(self) -> Line | None:
        """
        Find the bottom internal cushion of the playfield.
        
        Detects the bottom internal cushion by processing the lower portion of the playfield
        image to locate the horizontal edge between the playing surface and the bottom cushion
        (randa) border. Uses gradient-based edge detection on contrast-enhanced image data.
        
        Process:
            1. Binarize the image to isolate the playfield area
            2. Detect external borders of the playfield
            3. Extract intersection points from external borders
            4. Call find_bottom_internal_cushion with the original image and intersection points
            5. Return the detected bottom cushion line in global coordinates
        
        Returns:
            Line | None: Line object representing the bottom internal cushion (the horizontal
                        line separating the playing surface from the bottom cushion border) in
                        global image coordinates, or None if detection fails
        
        Note:
            The bottom cushion is typically a near-horizontal edge at the lower boundary of the
            playing area. The method uses gradient analysis on contrast-enhanced image data to
            detect the transition between the darker green playing surface and lighter green
            cushion border.
        """
        binary_mask, _ = binarize_playfield(self.img) 
        external_intersections, _, _ = find_playfield_exteral_borders(self.img, binary_mask)
        intersection_points = np.array([[int(inter.point.x), int(inter.point.y)] for inter in external_intersections])
        return find_bottom_internal_cushion(self.img, intersection_points)
