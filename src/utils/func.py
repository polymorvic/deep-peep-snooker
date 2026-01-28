import cv2
import numpy as np
from pathlib import Path
from typing import Literal, Iterable

from .common import array_like, NumpyImage
from .lines import Line, LineGroup, transform_line
from .intersections import Intersection, compute_intersections
from .points import Point
from .plotting import display_img


def trim_width(arr: np.ndarray | NumpyImage, pct: float) -> np.ndarray | NumpyImage:
    arr_copy = arr.copy()
    width = arr_copy.shape[1]
    cut = int(width * pct / 2)
    return arr_copy[:, cut: width - cut]
    

def get_corners(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = points[points[:, 0].argsort()[:2]]
    right = points[points[:, 0].argsort()[2:]]

    left_top, left_bottom = left[left[:, 1].argsort()]
    right_top, right_bottom = right[right[:, 1].argsort()]

    return left_top, left_bottom, right_top, right_bottom


def get_local_reference_line(
    lines: list[Line], 
    img: np.ndarray | NumpyImage, 
    direction: Literal["left", "right"], 
    x_start: int, 
    y_start: int) -> Line:
    global_ref_line = next(line for line in lines if line.slope is not None and (line.slope < 0 if direction == "left" else line.slope > 0))
    return transform_line(global_ref_line, img, x_start, y_start, to_global=False)


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


def _find_stable_column(mask: np.ndarray, start_idx: int, end_idx: int, step: int) -> int | None:
    """Find first column with stable height difference."""
    prev_diffs = []
    stability_window, tolerance = 5, 1
    
    for col_idx in range(start_idx, end_idx, step):
        col_locs = np.argwhere(mask[:, col_idx] > 0)
        if not col_locs.size:
            continue
        
        col_diff = col_locs[:, 0].max() - col_locs[:, 0].min()
        prev_diffs.append(col_diff)
        
        if len(prev_diffs) >= stability_window:
            recent = prev_diffs[-stability_window:]
            if max(recent) - min(recent) <= tolerance:
                return col_idx
    return None

def _crop_edges(arr: np.ndarray | NumpyImage, top: float = 0, bottom: float = 0, left: float = 0, right: float = 0) -> np.ndarray | NumpyImage:
    """Crop array edges by percentage."""
    h, w = arr.shape[:2]
    y0, y1 = int(h * top), int(h * (1 - bottom))
    x0, x1 = int(w * left), int(w * (1 - right))
    return arr[y0:y1, x0:x1]


def crop_and_split(arr: np.ndarray | NumpyImage, percent: float = 0.6) -> tuple[np.ndarray | NumpyImage, np.ndarray | NumpyImage, int]:
    split_h = arr.shape[0] // 2
    upper_arr = arr[:split_h]
    lower_arr = arr[split_h:]
    
    crop_pct = (1 - percent) / 2
    upper_arr = _crop_edges(upper_arr, top=crop_pct, left=crop_pct, right=crop_pct)
    lower_arr = _crop_edges(lower_arr, bottom=crop_pct, left=crop_pct, right=crop_pct)
    
    return upper_arr, lower_arr, split_h


def compute_adaptive_hsv_bounds(hsv_img: np.ndarray | NumpyImage, std_factor: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute adaptive HSV bounds for a given HSV image.

    Args:
        hsv_img: HSV image as numpy array or NumpyImage
        std_factor: Factor to multiply the standard deviations by
    Returns:
        Tuple containing lower and upper HSV bounds
    """
    h, s, v = pipette_color(hsv_img)

    stds = np.std(hsv_img, axis=(0, 1))
    tolerances = (stds * std_factor).astype(int)

    lower = np.clip([h, s, v] - tolerances, [0, 0, 0], [179, 255, 255])
    upper = np.clip([h, s, v] + tolerances, [0, 0, 0], [179, 255, 255])

    return lower, upper


def straighten_binary_mask(binary_mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_mask_close = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask_close_open = cv2.morphologyEx(binary_mask_close, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary_mask_close_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return binary_mask, binary_mask_close, binary_mask_close_open

    cnt = max(contours, key=cv2.contourArea)
    straighted_binary_mask = np.zeros_like(binary_mask)
    cv2.drawContours(straighted_binary_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    locs = np.argwhere(straighted_binary_mask > 0)
    top_h = locs[:, 0].min()
    bottom_h = locs[:, 0].max()

    left_col = _find_stable_column(straighted_binary_mask, 0, straighted_binary_mask.shape[1], 1)
    right_col = _find_stable_column(straighted_binary_mask, straighted_binary_mask.shape[1] - 1, -1, -1)
    
    if left_col is None or right_col is None:
        return straighted_binary_mask, binary_mask_close, binary_mask_close_open
    
    left, right = int(min(left_col, right_col)), int(max(left_col, right_col))
    span = right - left
    inner_span = max(1, int(round(span * 0.2))) if span > 0 else 1
    inner_left = left + (span - inner_span) // 2
    inner_right = inner_left + inner_span

    straighted_binary_mask[top_h:bottom_h+1, inner_left:inner_right+1] = 255

    return straighted_binary_mask, binary_mask_close, binary_mask_close_open

        
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


def filter_edges_by_reference_line(edges_img: np.ndarray, ref_line: Line, inside_direction: str, margin: int = 12) -> np.ndarray:
    h, w = edges_img.shape
    filtered_edges = edges_img.copy()
    
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    if inside_direction == "left":
        test_x = w - 1  
    else:  
        test_x = 0  
    test_y = h // 2
    
    if ref_line.xv is not None:
        x_line = ref_line.xv
        distances = x_coords - x_line
        
        test_distance = test_x - x_line
        is_inside_positive = test_distance > 0
        
        if is_inside_positive:
            mask = distances > margin
        else:
            mask = distances < -margin
    else:
        slope = ref_line.slope
        intercept = ref_line.intercept
        
        test_y_on_line = slope * test_x + intercept
        
        if test_y < test_y_on_line:
            is_inside_above = True
        else:
            is_inside_above = False
        
        y_on_line = slope * x_coords + intercept
        
        if is_inside_above:
            side_mask = y_coords < y_on_line
        else:
            side_mask = y_coords > y_on_line
        
        sqrt_term = np.sqrt(1 + slope**2)
        abs_distances = np.abs((y_coords - slope * x_coords - intercept) / sqrt_term)
        mask = side_mask & (abs_distances > margin)
    
    filtered_edges[~mask] = 0
    return filtered_edges


def filter_lines_by_reference(lines: list[Line], ref_line: Line, slope_tolerance: float = 0.85) -> Line | None:
    if ref_line is None or not lines or ref_line.xv is not None or ref_line.slope is None:
        return None
    
    filtered = []
    for line in lines:
        if (line.xv is None and line.slope is not None and line.intercept is not None and abs(line.slope - ref_line.slope) < slope_tolerance):
            filtered.append(line)
    
    if not filtered:
        return None
    
    current_lines = filtered
    if len(current_lines) > 1:
        groups = group_lines(current_lines)
        current_lines = groups
    return current_lines[0]