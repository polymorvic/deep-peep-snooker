import cv2
import matplotlib.pyplot as plt
import numpy as np

from .intersections import compute_intersections, Intersection
from .func import (crop_center, pipette_color,
                   _straighten_mask, _convert_hough_segments_to_lines,
                   group_lines, _select_lines, crop_image_by_points, sanitize_lines
                   )
from .lines import transform_line
                   



    # find_playfield_exteral_borders, 
    # blackout_pixels_outside_borders,
    # find_playfield_internal_sideline_borders, 
    # find_top_internal_cushion, find_baulk_line, 
    # find_bottom_internal_cushion)


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
        self.preprocessed_img = None
        self.straighted_mask = None
        self.external_edges_intersections = None
        self.center = self._get_center_point()
        self.vertical_axis = self._create_vertical_axis()
        self.horizontal_axis = self._create_horizontal_axis()

        self._preprocess_image()


    @staticmethod
    def intersection_to_points_array(intersections: list[Intersection]) -> np.ndarray[int]:
        return np.array([[int(inter.point.x), int(inter.point.y)] for inter in intersections])


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
    

    def _preprocess_image(
            self, 
            kernel_size: tuple[int, int] = (21, 21),
            canny_thresh_lower: int = 150, 
            canny_thresh_upper: int = 200, 
            hough_thresh: int = 100, 
            hough_min_line_len: int = 100, 
            hough_max_line_gap: int = 10,
            group_lines_thresh_intercept: int = 100) -> None:

        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        cropped_img_hsv = crop_center(img_hsv)

        h, s, v = pipette_color(cropped_img_hsv)
        h_std = np.std(cropped_img_hsv[:, :, 0])
        s_std = np.std(cropped_img_hsv[:, :, 1])
        v_std = np.std(cropped_img_hsv[:, :, 2])

        h_tolerance = int(h_std * 1.5)
        s_tolerance = int(s_std * 1.5)
        v_tolerance = int(v_std * 1.5)

        lower_bound = np.array([max(0, h - h_tolerance), 
                            max(0, s - s_tolerance), 
                            max(0, v - v_tolerance)])

        upper_bound = np.array([min(179, h + h_tolerance), 
                            min(255, s + s_tolerance), 
                            min(255, v + v_tolerance)])

        binary_mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        inv_binary_img = cv2.bitwise_not(binary_mask)

        binary_mask_close = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones(kernel_size, np.uint8))
        straightened_binary_mask_close = _straighten_mask(binary_mask_close)

        edges = cv2.Canny(straightened_binary_mask_close, canny_thresh_lower, canny_thresh_upper)
        segments = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi / 180, 
            threshold=hough_thresh, 
            minLineLength=hough_min_line_len, 
            maxLineGap=hough_max_line_gap
        )

        lines = _convert_hough_segments_to_lines(segments)
        if any(line.slope is None for line in lines):
            lines = sanitize_lines(lines)

        lines = group_lines(lines, thresh_intercept=group_lines_thresh_intercept)
        lines = _select_lines(lines)
        intersections = compute_intersections(lines, self.img)

        pic_copy = self.img.copy()
        for intersection, line in zip(intersections, lines):
            pt = intersection.point.as_int()
            end_pts = line.limit_to_img(pic_copy)
            cv2.line(pic_copy, *end_pts, (255, 0, 0), 1)
            cv2.circle(pic_copy, pt, 2,(0, 0, 255), -1)

        self.preprocessed_img = pic_copy
        self.straighted_mask = straightened_binary_mask_close
        self.external_edges_intersection_points = PlayfieldFinder.intersection_to_points_array(intersections)


    # def find_side_cushions(self, slope_threshold: float = 0.1) -> tuple[Line, Line]:
    #     """
    #     Find the closest line on the left side and the closest line on the right side of the center.
    #     Excludes horizontal lines (slope close to zero).
        
    #     Args:
    #         slope_threshold: Lines with absolute slope less than this value are excluded (default: 0.1)
        
    #     Returns:
    #         Tuple containing (left_line, right_line) - the closest lines on each side
    #     """

    #     binary_mask, inv_binary_img = binarize_playfield(self.img)
    #     external_intersections, external_lines, _ = find_playfield_exteral_borders(self.img, binary_mask)
    #     blackout_img = blackout_pixels_outside_borders(external_intersections, inv_binary_img, external_lines)
    #     return find_playfield_internal_sideline_borders(blackout_img)


    # def find_top_internal_cushion(self) -> Line | None:
    #     """
    #     Find the top internal cushion of the playfield.
        
    #     The method processes the image to isolate the playfield area and then detects
    #     the top horizontal cushion (the top boundary of the playing area). It uses
    #     probabilistic Hough line transformation to detect near-horizontal lines and
    #     selects the one positioned highest in the image (lowest intercept value).
        
    #     Process:
    #         1. Binarize the image to isolate the playfield area
    #         2. Detect external borders of the playfield
    #         3. Create a blackout image with pixels outside borders removed
    #         4. Detect line segments in the blackout image using Hough transform
    #         5. Filter for near-horizontal lines (abs(slope) < 2)
    #         6. Sort lines by intercept value (ascending)
    #         7. Return the line with the lowest intercept (highest position)
        
    #     Returns:
    #         Line | None: Line object representing the top internal cushion (the line
    #                     with the lowest intercept value), or None if no suitable lines
    #                     are found
        
    #     Note:
    #         The top cushion is typically a near-horizontal line that defines the upper
    #         boundary of the playing area. The method filters out vertical and highly
    #         sloped lines to focus on horizontal boundaries.
    #     """
    #     binary_mask, inv_binary_img = binarize_playfield(self.img)
    #     external_intersections, external_lines, _, _ = find_playfield_exteral_borders(self.img, binary_mask)
    #     blackout_img = blackout_pixels_outside_borders(external_intersections, inv_binary_img, external_lines)
    #     return find_top_internal_cushion(blackout_img)


    # def find_baulk_line(self) -> Line | None:
    #     """
    #     Find the baulk line of the playfield.
        
    #     Detects the baulk line by processing the image to isolate the playfield area
    #     and selecting the near-horizontal line with intercept closest to the center.
        
    #     Returns:
    #         Line | None: Line object representing the baulk line in global coordinates,
    #                     or None if no suitable line is found
    #     """
    #     binary_mask, _ = binarize_playfield(self.img) 
    #     external_intersections, _, _ = find_playfield_exteral_borders(self.img, binary_mask)
    #     intersection_points = np.array([[int(inter.point.x), int(inter.point.y)] for inter in external_intersections])
    #     return find_baulk_line(self.img, intersection_points)


    def find_bottom_internal_cushion(self) -> Line | None:
        cropped_by_points, x_start, y_start = crop_image_by_points(self.img, self.external_edges_intersection_points)

        H = cropped_by_points.height
        roi = cropped_by_points[int(0.95*H):] 

        hsv_img = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        _, _, v = cv2.split(hsv_img)  

        egdes = cv2.Canny(v, 10, 50)
        segments = cv2.HoughLinesP(egdes, 1, np.pi/180, 100, 100, 25)

        if segments is not None:
            lines = _convert_hough_segments_to_lines(segments)
            lines = [line for line in lines if line.slope == 0]
            if lines:
                bottom_line_local = sorted(lines, key=lambda line: line.intercept)[0]
                
                roi_y_start = int(0.95 * H)
                
                bottom_line_global = transform_line(
                    bottom_line_local, 
                    roi, 
                    x_start,         
                    y_start + roi_y_start
                )
                
                return bottom_line_global
            else:
                return None
        else:
            return None
