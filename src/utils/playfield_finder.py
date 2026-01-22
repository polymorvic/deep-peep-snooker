import cv2
import matplotlib.pyplot as plt
import numpy as np

from .intersections import compute_intersections, Intersection
from .func import (compute_adaptive_hsv_bounds, pipette_color,
                   straighten_binary_mask, convert_hough_segments_to_lines,
                   group_lines, select_lines, crop_image_by_points, sanitize_lines, crop_and_split
                   )

from .lines import Line, transform_line
from .plotting import display_img
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

        self.lower_bound_l = None
        self.upper_bound_l = None

        self._preprocess_image()


    @staticmethod
    def intersection_to_points_array(intersections: list[Intersection]) -> np.ndarray[int]:
        intersections = sorted(intersections, key=lambda inter: (inter.point.y, inter.point.x))
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
        kernel_size: int = 21,
        canny_thresh_lower: int = 150, 
        canny_thresh_upper: int = 200, 
        hough_thresh: int = 50, 
        hough_min_line_len: int = 100, 
        hough_max_line_gap: int = 25
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        upper_img_hsv_cropped, lower_img_hsv_cropped, split_h = crop_and_split(img_hsv)

        upper_img_hsv = img_hsv[:split_h, :]
        lower_img_hsv = img_hsv[split_h:, :]

        self.lower_bound_u, self.upper_bound_u = compute_adaptive_hsv_bounds(upper_img_hsv_cropped)
        self.lower_bound_l, self.upper_bound_l = compute_adaptive_hsv_bounds(lower_img_hsv_cropped)

        upper_binary = cv2.inRange(upper_img_hsv, self.lower_bound_u, self.upper_bound_u)
        lower_binary = cv2.inRange(lower_img_hsv, self.lower_bound_l, self.upper_bound_l)

        binary_mask = np.vstack([upper_binary, lower_binary])
        inv_binary_img = cv2.bitwise_not(binary_mask)

        straighted_binary_mask, binary_mask_close, _ = straighten_binary_mask(binary_mask, kernel_size)
        edges = cv2.Canny(straighted_binary_mask, canny_thresh_lower, canny_thresh_upper)
        segments = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi / 180, 
            threshold=hough_thresh, 
            minLineLength=hough_min_line_len, 
            maxLineGap=hough_max_line_gap
        )
        copy_edges = self.img.copy()
        for segment in segments:
            x1, y1, x2, y2 = segment[0]
            cv2.line(copy_edges, (x1, y1), (x2, y2), (255, 0, 0), 5)
        
        lines = convert_hough_segments_to_lines(segments)
        lines = select_lines(lines)
        intersections = compute_intersections(lines, self.img)

        pic_copy = self.img.copy()
        for intersection, line in zip(intersections, lines):
            pt = intersection.point.as_int()
            end_pts = line.limit_to_img(pic_copy)
            cv2.line(pic_copy, *end_pts, (255, 0, 0), 2)
            cv2.circle(pic_copy, pt, 2,(0, 0, 255), 2)

        # display_img(inv_binary_img)
        # display_img(binary_mask_close)
        # display_img(straighted_binary_mask)
        # display_img(edges)
        # display_img(copy_edges)
        # display_img(pic_copy)
        # return binary_mask, binary_mask_close, straighted_binary_mask, edges, copy_edges, pic_copy

        self.preprocessed_img = pic_copy
        self.straighted_mask = straighted_binary_mask
        self.external_edges_intersection_points = PlayfieldFinder.intersection_to_points_array(intersections)


    def find_top_internal_cushion(self) -> Line | None:
        x1, x2 = self.external_edges_intersection_points[:,0][:2]

        width = x2 - x1
        x_crop_start = int(x1 + 0.1 * width)
        x2 = int(x2 - 0.1 * width)

        y_crop_start, y2 = self.external_edges_intersection_points[:,1].min(), self.external_edges_intersection_points[:,1].max()
        cropped_by_points = self.img[y_crop_start:y2, x_crop_start:x2]

        H = cropped_by_points.height
        roi_y_start_local = 0
        roi_y_end_local = int(0.1*H)

        roi = cropped_by_points[roi_y_start_local:roi_y_end_local]

        # display_img(roi)
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        lower_bound, upper_bound = compute_adaptive_hsv_bounds(hsv_roi)
        bin_roi = cv2.inRange(hsv_roi, lower_bound, upper_bound)

        # display_img(bin_roi)

        count = 0
        tolerance = 3
        break_row_idx = None
        
        for row_idx in range(bin_roi.shape[0]):
            ones_count = np.sum(bin_roi[row_idx, :] > 0)

            if ones_count > bin_roi.shape[1] // 2:
                count += 1
                if count >= tolerance and break_row_idx is None:
                    break_row_idx = row_idx
            else:
                count = 0

            if ones_count > bin_roi.shape[1] * 0.3:
                bin_roi[row_idx] = 255

        if break_row_idx is not None:
            bin_roi[:break_row_idx] = 255

        # display_img(bin_roi)

        egdes = cv2.Canny(bin_roi, 100, 150)
        # display_img(egdes)
        segments = cv2.HoughLinesP(egdes, 1, np.pi/180, 100, 100, 50)
        if segments is not None:

            roi_copy = roi.copy()
            for segment in segments:
                x1, y1, x2, y2 = segment[0]
                cv2.line(roi_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # display_img(roi_copy)

            lines = convert_hough_segments_to_lines(segments)
            lines = [line for line in lines if line.slope == 0 and line.intercept > 0]
            lines = sorted(lines, key=lambda line: line.intercept)
            # print(lines)

            top_line_global = transform_line(
                lines[0], 
                roi, 
                x_crop_start, 
                y_crop_start + roi_y_start_local
            )
            return top_line_global

        
        else:
            return None
        

    def find_bottom_internal_cushion(self) -> Line | None:
        cropped_by_points, x_start, original_y_start = crop_image_by_points(self.img, self.external_edges_intersection_points)

        H = cropped_by_points.height
        roi_y_start_local = int(0.95*H)
        roi_y_end_local = H
        
        while True:
            roi = cropped_by_points[roi_y_start_local:roi_y_end_local]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            bin_roi = cv2.inRange(hsv_roi, self.lower_bound_l, self.upper_bound_l)
            white_ratio = cv2.countNonZero(bin_roi) / bin_roi.size

            if white_ratio > 0.5:
                break
            roi_y_start_local -= 1
            roi_y_end_local -= 1

        _, _, v = cv2.split(hsv_roi)

        egdes = cv2.Canny(v, 10, 50)
        segments = cv2.HoughLinesP(egdes, 1, np.pi/180, 100, 100, 25)

        if segments is not None:
            lines = convert_hough_segments_to_lines(segments)
            lines = [line for line in lines if line.slope == 0]
            lines = group_lines(lines)
            if lines:
                bottom_line_local = sorted(lines, key=lambda line: line.intercept)[0]
                
                bottom_line_global = transform_line(
                    bottom_line_local, 
                    roi, 
                    x_start,         
                    original_y_start + roi_y_start_local
                )
                
                return bottom_line_global
            else:
                return None
        else:
            return None
