from ctypes import pointer
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal, Iterable

from .const import ref_snooker_playfield
from .common import array_like, NumpyImage
from .lines import Line, LineGroup, transform_line
from .intersections import Intersection, compute_intersections
from .points import Point
from .plotting import display_img


def crop_center(arr: np.ndarray | NumpyImage, percent: float = 0.75) -> np.ndarray | NumpyImage:
    """
    Crop the center of the image by a given percentage.

    Args:
        arr: numpy array representing the image
        percent: percentage of the image to crop (default is 75%)

    Returns:
        numpy array representing the cropped image
    """
    if not (0 < percent <= 1):
        raise ValueError("percent must be in (0, 1].")

    h, w = arr.shape[:2]
    s = int(min(h, w) * percent)
    y, x = h // 2, w // 2
    r = s // 2
    return arr[y - r:y + r, x - r:x + r]


def _crop_edges(arr: np.ndarray | NumpyImage, top: float = 0, bottom: float = 0, left: float = 0, right: float = 0) -> np.ndarray | NumpyImage:
    """Crop array edges by percentage."""
    h, w = arr.shape[:2]
    y0, y1 = int(h * top), int(h * (1 - bottom))
    x0, x1 = int(w * left), int(w * (1 - right))
    return arr[y0:y1, x0:x1]


def crop_and_split(arr: np.ndarray | NumpyImage, percent: float = 0.6) -> tuple[np.ndarray | NumpyImage, np.ndarray | NumpyImage]:
    split_h = arr.shape[0] // 2
    upper_arr = arr[:split_h]
    lower_arr = arr[split_h:]
    
    crop_pct = (1 - percent) / 2
    upper_arr = _crop_edges(upper_arr, top=crop_pct, left=crop_pct, right=crop_pct)
    lower_arr = _crop_edges(lower_arr, bottom=crop_pct, left=crop_pct, right=crop_pct)
    
    return upper_arr, lower_arr


def straighten_binary_mask(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_mask_close = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask_close_open = cv2.morphologyEx(binary_mask_close, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary_mask_close_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        straighted_binary_mask = np.zeros_like(binary_mask)
        cv2.drawContours(straighted_binary_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        return straighted_binary_mask, binary_mask_close, binary_mask_close_open
    else:
        return binary_mask, binary_mask_close, binary_mask_close_open


def select_lines(lines: list[Line]) -> list[Line]:
    selected_lines = []
    slope_threshold = 0.1
    
    for l in lines:
        if l.slope is not None and abs(l.slope) < slope_threshold:
            l.slope = 0
    
    horizontal = [l for l in lines if l.slope is not None and l.slope == 0]
    if horizontal:
        horizontal.sort(key=lambda l: l.intercept if l.intercept is not None else float('inf'))
        selected_lines.append(horizontal[0])  # min intercept
        if len(horizontal) > 1:
            selected_lines.append(horizontal[-1])  # max intercept
    
    negative = [l for l in lines if l.slope is not None and l.slope < -slope_threshold and l.intercept is not None]
    if negative:
        leftmost = min(negative, key=lambda l: -l.intercept / l.slope)
        selected_lines.append(leftmost)
    
    positive = [l for l in lines if l.slope is not None and l.slope > slope_threshold and l.intercept is not None]
    if positive:
        rightmost = max(positive, key=lambda l: -l.intercept / l.slope)
        selected_lines.append(rightmost)

    return selected_lines


def crop_image_by_points(
    img: np.ndarray | NumpyImage,
    points: np.ndarray[int] | list[Point] | list[Intersection]
    ) -> tuple[np.ndarray | NumpyImage, int, int]:
    """
    Crop an image to a bounding box defined by a set of points.
    
    The function calculates the bounding box from the provided points and crops
    the image to that region. Returns the cropped image and the origin offset
    coordinates (x_start, y_start) that can be used with transform_line to
    convert coordinates between local (cropped) and global (original) reference frames.
    
    Args:
        img: Input image as numpy array or NumpyImage
        points: Array of points as numpy array with shape (N, 2) where each row is [x, y],
                or list of Point objects, or list of Intersection objects
    
    Returns:
        Tuple containing:
            - Cropped image as numpy array or NumpyImage
            - x_start (int): X-axis offset of the cropped region in the original image
            - y_start (int): Y-axis offset of the cropped region in the original image
    
    Raises:
        ValueError: If points array is empty or has invalid shape
    """
    min_x = int(np.min(points[:, 0]))
    max_x = int(np.max(points[:, 0]))
    min_y = int(np.min(points[:, 1]))
    max_y = int(np.max(points[:, 1]))
    
    img_height, img_width = img.height, img.width

    x_start = max(min_x, 0)
    y_start = max(min_y, 0)
    
    end_x = min(img_width, max_x + 1)
    end_y = min(img_height, max_y + 1)

    return img[y_start:end_y, x_start:end_x], x_start, y_start


def read_image_as_numpyimage(path: str | Path, color_mode: Literal["rgb", "hsv", "grayscale"] = "rgb") -> NumpyImage:
    """
    Read an image and return it as a NumpyImage instance.
    
    Args:
        path: Path to the image file
        color_mode: Color mode for the image - 'rgb', 'hsv', or 'grayscale' (default 'rgb')
        
    Returns:
        NumpyImage instance containing the loaded image
        
    Raises:
        ValueError: If color_mode is not 'rgb', 'hsv', or 'grayscale'
        FileNotFoundError: If the image file cannot be read
    """
    color_mode = color_mode.lower()
    if color_mode not in {"rgb", "hsv", "grayscale"}:
        raise ValueError("color_mode must be 'RGB', 'HSV', or 'GRAYSCALE'")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if color_mode == "grayscale" else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    conversions = {
        "rgb": lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
        "hsv": lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV),
    }

    img = conversions.get(color_mode, lambda x: x)(img)
    return NumpyImage(img)


def pipette_color(image: np.ndarray | NumpyImage, k: int = 4) -> tuple[int, int, int]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected HSV image with 3 channels")

    X = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(X, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    labels = labels.ravel()
    dom = np.bincount(labels).argmax()
    c = X[labels == dom]

    return tuple(map(int, np.median(c, axis=0)))


def convert_hough_segments_to_lines(hough_lines: np.ndarray | None) -> list[Line]:
    """
    Convert Hough line segments to Line objects.
    
    Args:
        hough_lines: Array of line segments from cv2.HoughLinesP, 
                    where each line is [[x1, y1, x2, y2]], or None if no lines detected
        
    Returns:
        List of Line objects converted from the segments
    """
    if hough_lines is None:
        return []
    
    line_objects = []
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        line_obj = Line.from_hough_line((x1, y1, x2, y2))
        line_objects.append(line_obj)
    return line_objects


def group_lines(lines: list[Line], 
    thresh_theta: float | int = 5, 
    thresh_intercept: float | int = 10
    ) -> list[LineGroup]:
    """
    Group similar Line objects into LineGroups based on orientation and position thresholds.

    Args:
        lines (list[Line]): A list of Line objects to group.
        thresh_theta (float): Maximum allowed angle difference between lines to be in the same group.
        thresh_intercept (float): Maximum allowed intercept difference (for non-vertical lines).

    Returns:
        list[LineGroup]: A list of LineGroup objects representing grouped lines.
    """
    groups = []

    for line in lines:
        for group in groups:
            if group.process_line(line, thresh_theta, thresh_intercept):
                break
        else:
            groups.append(LineGroup([line]))

    return groups


def sanitize_lines(lines: list[Line]) -> list[Line]:
    unique_lines = list(set(lines))
    reference_lines = [line for line in unique_lines 
                       if line.slope is not None and abs(line.slope) > 1e-10]

    horizontal_lines = [line for line in unique_lines 
                        if line.slope is not None and abs(line.slope) < 1e-10]
    
    if not reference_lines:
        return unique_lines
    
    reference_line = max(reference_lines, key=lambda line: abs(line.slope))
    reference_slope = reference_line.slope
    reference_intercept = reference_line.intercept
    mirror_slope = -reference_slope

    new_line_intercept = reference_intercept * -0.3 if reference_slope > 0 else reference_intercept * -3
    
    sanitized = set()
    for line in unique_lines:
        if line.slope is None:
            new_line = Line(slope=mirror_slope, intercept=new_line_intercept, xv=None)
            sanitized.add(new_line)
        else:
            sanitized.add(line)
    
    return list(sanitized) + horizontal_lines


def find_playfield_internal_sideline_borders(
    blackout_img: array_like, 
    hough_thresh: int = 100, 
    hough_min_line_len: int = 200, 
    hough_max_line_gap: int = 10,
    thresh_theta: float | int = 50,
    thresh_intercept: float | int = 200,
    slope_threshold: float = 0.1
    ) -> list[LineGroup] | None:
    """
    Find internal sideline borders of a playfield by detecting line segments.
    
    The function processes a blackout image (where playfield area has been isolated)
    to detect internal sideline borders using probabilistic Hough line transformation.
    It filters and groups similar lines to identify the main sideline borders.
    Returns exactly 2 lines with opposite slopes (one positive, one negative).
    
    Process:
        1. Apply probabilistic Hough line transformation to detect line segments
        2. Convert detected segments to Line objects
        3. Filter lines based on slope threshold (keep vertical or high-slope lines)
        4. Group similar lines together based on angle and intercept thresholds
        5. If exactly 2 lines are found, return them as-is
        6. If not 2 lines, filter to keep only lines with opposite slopes (positive and negative),
           selecting the group with most lines from each category
        7. Return None if opposite slopes cannot be found
    
    Args:
        blackout_img: Binary image with isolated playfield (white pixels represent playfield)
        hough_thresh: Minimum votes to detect a line in Hough transform (default 100)
        hough_min_line_len: Minimum line length in pixels (default 200)
        hough_max_line_gap: Maximum gap between line segments to connect (default 10)
        thresh_theta: Angle threshold for grouping similar lines in degrees (default 50)
        thresh_intercept: Intercept threshold for grouping similar lines (default 200)
        slope_threshold: Minimum slope magnitude to keep a line (default 0.1)
                          Lines with None slope (vertical) or abs(slope) >= threshold are kept
    
    Returns:
        list[LineGroup] | None: List containing exactly 2 LineGroup objects with opposite slopes
                                (one with positive slope, one with negative slope) if found,
                                or None if lines with opposite slopes cannot be found. If exactly
                                2 lines are detected after grouping, they are returned as-is.
    
    Note:
        The function is designed to detect internal sidelines, which are typically vertical
        or nearly vertical lines. The slope_threshold parameter filters out near-horizontal
        lines that are not relevant for sideline detection. The function ensures that only
        2 lines with opposite slopes are returned, selecting the groups with the most lines
        in each category.
    """
    segments = cv2.HoughLinesP(
        blackout_img, 
        1, 
        np.pi / 180, 
        threshold=hough_thresh, 
        minLineLength=hough_min_line_len, 
        maxLineGap=hough_max_line_gap
    )

    lines = convert_hough_segments_to_lines(segments)
    lines = [line for line in lines if line.slope is None or abs(line.slope) >= slope_threshold]
    lines = group_lines(lines, thresh_theta=thresh_theta, thresh_intercept=thresh_intercept)
    
    if len(lines) != 2:
        positive = [lg for lg in lines if lg.slope is not None and lg.slope > 0]
        negative = [lg for lg in lines if lg.slope is not None and lg.slope < 0]
        
        if positive and negative:
            return [max(positive, key=lambda lg: len(lg.lines)), 
                     max(negative, key=lambda lg: len(lg.lines))]
        else:
            return None

    return lines


# def find_top_internal_cushion(
#     blackout_img: array_like,

#     ) -> Line | None:

#     smoothed_binary_mask = _straighten_mask(blackout_img)
#     edges = cv2.Canny(smoothed_binary_mask, 150, 200)
#     segments = cv2.HoughLinesP(
#         edges, 
#         1, 
#         np.pi / 180, 
#         threshold=100, 
#         minLineLength=100, 
#         maxLineGap=10
#     )

#     lines = _convert_hough_segments_to_lines(segments)
#     lines = group_lines(lines, thresh_intercept=100)
#     lines = _select_lines(lines)
#     intersections = compute_intersections(lines, binary_mask)

    # pic_copy = pic.copy()
    # for line in lines:
    #     pts = line.limit_to_img(pic_copy)
    #     cv2.line(pic_copy, *pts, (255, 0, 0), 2)


def find_baulk_line(
    original_image: array_like,
    intersection_points: np.ndarray,
    hough_thresh: int = 100,
    hough_min_line_len: int = 200,
    hough_max_line_gap: int = 10,
    ) -> Line | None:
    """
    Find the baulk line in a cropped playfield image.
    
    Detects near-horizontal lines using Hough transform and selects the one
    with intercept closest to the image center.
    
    Args:
        original_image: Input image as RGB numpy array or NumpyImage
        intersection_points: Array of intersection points used to crop the image
        hough_thresh: Minimum votes to detect a line (default: 100)
        hough_min_line_len: Minimum line length in pixels (default: 200)
        hough_max_line_gap: Maximum gap between line segments (default: 10)
    
    Returns:
        Line | None: Line object representing the baulk line in global coordinates,
                    or None if no suitable line is found
    """
    cropped_by_points, x_start, y_start = crop_image_by_points(original_image, intersection_points)

    hsv_img = cv2.cvtColor(cropped_by_points, cv2.COLOR_RGB2HSV)
    _, _, v = cv2.split(hsv_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v_eq = clahe.apply(v)

    edges = cv2.Canny(v_eq, 50, 150)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_thresh,
        minLineLength=hough_min_line_len,
        maxLineGap=hough_max_line_gap
    )

    if segments is not None:
        lines = convert_hough_segments_to_lines(segments)
        lines = [line for line in lines if abs(line.slope) < 2]

        if lines:
            lines = group_lines(lines, thresh_theta=50, thresh_intercept=10)
            lines = sorted(lines, key=lambda line: line.intercept)

            center_y = cropped_by_points.height / 2
            baulk_line_local = min(lines, key=lambda line: abs(line.intercept - center_y))
            baulk_line_global = transform_line(baulk_line_local, original_image, x_start, y_start)

            return baulk_line_global
        else:
            return None
    else:
        return None
