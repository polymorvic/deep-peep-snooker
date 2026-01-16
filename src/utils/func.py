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


def _straighten_mask(mask: np.ndarray) -> np.ndarray:

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

    m = mask.astype(bool)
    H, W = m.shape

    ys = np.where(m.any(1))[0]
    y0, y1 = ys[0], ys[-1]
    y = np.arange(y0, y1 + 1)

    xl, xr = [], []
    for yy in y:
        xs = np.where(m[yy])[0]
        if xs.size:
            xl.append(xs[0])
            xr.append(xs[-1])

    y = y[:len(xl)]
    xl, xr = np.array(xl), np.array(xr)

    ml, cl = np.polyfit(y, xl, 1)
    mr, cr = np.polyfit(y, xr, 1)
    al, bl = 1/ml, -cl/ml
    ar, br = 1/mr, -cr/mr

    ym = int(np.median(y))
    xlm = np.interp(ym, y, xl)
    xrm = np.interp(ym, y, xr)

    if al >= 0:
        al = -abs(ar)
        bl = ym - al * xlm
    if ar <= 0:
        ar = abs(al)
        br = ym - ar * xrm

    out = np.zeros((H, W), dtype=np.uint8)
    for yy in range(y0, y1 + 1):
        xL = int((yy - bl) / al)
        xR = int((yy - br) / ar)
        xL, xR = sorted((np.clip(xL, 0, W-1), np.clip(xR, 0, W-1)))
        out[yy, xL:xR + 1] = 1

    return (out * 255).astype(np.uint8)


def _select_lines(lines: list[Line]) -> list[Line]:
    """
    Select exactly 4 lines from a list of lines:
    - Top line: slope = 0, intercept = min
    - Bottom line: slope = 0, intercept = max
    - Left line: slope < 0 (negative)
    - Right line: slope > 0 (positive)
    
    Args:
        lines: List of Line objects
        
    Returns:
        List of exactly 4 Line objects: [top_line, bottom_line, left_line, right_line]
    """
    horizontal_lines = [line for line in lines if line.slope is not None and abs(line.slope) < 0.1]
    positive_lines = [line for line in lines if line.slope is not None and line.slope > 0]
    negative_lines = [line for line in lines if line.slope is not None and line.slope < 0]

    selected_lines = []
    if horizontal_lines:
        horizontal_lines_sorted = sorted(horizontal_lines, key=lambda line: line.intercept if line.intercept is not None else float('inf'))
        top_line = horizontal_lines_sorted[0] 
        selected_lines.append(top_line)
    
    if horizontal_lines and len(horizontal_lines) > 1:
        bottom_line = horizontal_lines_sorted[-1]
        selected_lines.append(bottom_line)
    elif horizontal_lines:
        selected_lines.append(horizontal_lines_sorted[0])
    
    if negative_lines:
        selected_lines.append(negative_lines[0])
    
    if positive_lines:
        selected_lines.append(positive_lines[0])
    
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


def pipette_color(image: np.ndarray | NumpyImage) -> tuple[int, int, int]:
    """
    Detect the dominant color in an image using KMeans clustering.
    
    Args:
        image: Input image as numpy array or NumpyImage
        color_mode: Color space to work in - 'rgb' or 'hsv'
        
    Returns:
        Tuple of dominant color values (R,G,B) or (H,S,V)
        
    Raises:
        ValueError: If image is binary/grayscale or color_mode is invalid
    """
    
    if image.ndim == 2:
        raise ValueError("Image must be color (3 channels), not grayscale or binary")
    
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("Image must have exactly 3 channels for color processing")
    
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return  tuple(map(int, centers[0]))


def _convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if needed.
    
    Args:
        img: Input image (grayscale or color)
        
    Returns:
        Grayscale image as 2D numpy array
        
    Raises:
        ValueError: If input image has invalid dimensions
    """
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 2:
        return img.copy()
    else:
        raise ValueError("Input image must be 2D (grayscale) or 3D (color)")


def _convert_hough_segments_to_lines(hough_lines: np.ndarray | None) -> list[Line]:
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


def apply_pht(
    img: np.ndarray,
    canny_thresh_lower: int = 30,
    canny_thresh_upper: int = 70,
    hough_thresh: int = 150,
    hough_min_line_len_percent: float = 0.4,
    hough_max_line_gap: int = 10,
    ) -> tuple[np.ndarray, list]:
    """
    Apply probabilistic Hough line transformation to an image.
    
    Process:
        1. Convert to grayscale if needed
        2. Apply Gaussian blur to reduce noise
        3. Detect edges using Canny edge detection
        4. Apply Hough line transformation to find straight lines
        5. Draw detected lines on the original image
    
    Args:
        img: Input image (grayscale or color)
        blur_kernel_size: Size of Gaussian blur kernel (default 5)
        canny_thresh_lower: Lower threshold for Canny edge detection (default 50)
        canny_thresh_upper: Upper threshold for Canny edge detection (default 150)
        hough_thresh: Minimum votes to detect a line (default 100)
        hough_min_line_len_percent: Minimum line length as percentage of image height (default 0.1)
        hough_max_line_gap: Maximum gap between line segments (default 10)
        
    Returns:
        Tuple containing:
            - Image with detected lines drawn
            - List of Line objects converted from detected line segments
    """
    img_gray = _convert_to_grayscale(img)
    
    hough_min_line_len = int(img_gray.shape[0] * hough_min_line_len_percent)

    img_gray = 255 - img_gray
    edges = cv2.Canny(img_gray, canny_thresh_lower, canny_thresh_upper)
    
    segments = cv2.HoughLinesP(
        edges, 
        1, 
        np.pi / 180, 
        threshold=hough_thresh, 
        minLineLength=hough_min_line_len, 
        maxLineGap=hough_max_line_gap
    )
    lines = _convert_hough_segments_to_lines(segments)
    
    segments_img = cv2.cvtColor(img_gray.copy(), cv2.COLOR_GRAY2RGB)
    lines_img = segments_img.copy()
    if segments is not None:
        for segment, line in zip(segments, lines):
            x1, y1, x2, y2 = segment[0]
            pts = line.limit_to_img(lines_img)
            cv2.line(segments_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(lines_img, *pts, (255, 0, 0), 2)
    
    return segments_img, lines_img, lines


def binarize_playfield(img: np.ndarray | NumpyImage) -> tuple[np.ndarray, np.ndarray]:
    """
    Binarize an image to isolate the playfield area using color-based segmentation.
    
    The function detects the dominant color in the center of the image and creates
    binary masks based on that color. It uses HSV color space for more robust color
    detection and calculates dynamic tolerance thresholds based on color variance.
    
    Process:
        1. Convert image to HSV color space
        2. Crop the center region of the image to analyze dominant color
        3. Detect dominant color using KMeans clustering
        4. Calculate dynamic tolerance based on color variance
        5. Create binary mask using cv2.inRange with calculated thresholds
        6. Generate inverted binary image
    
    Args:
        img: Input image as RGB numpy array or NumpyImage
    
    Returns:
        Tuple containing two binary masks:
            - binary_mask: Binary mask where white pixels represent the detected playfield color
            - inv_binary_img: Inverted binary mask (black pixels represent detected color)
    
    Note:
        The tolerance for color matching is dynamically calculated as 1.5x the standard
        deviation of each HSV channel, allowing for variation in lighting and shadows.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    cropped_img_hsv = crop_center(img_hsv)

    dominant_color = pipette_color(cropped_img_hsv)

    h, s, v = dominant_color
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

    return binary_mask, inv_binary_img


def find_playfield_exteral_borders(
    original_img: array_like,
    binary_mask: array_like, 
    kernel_size: tuple[int, int] = (21, 21), 
    canny_thresh_lower: int = 150, 
    canny_thresh_upper: int = 200, 
    hough_thresh: int = 100, 
    hough_min_line_len: int = 100, 
    hough_max_line_gap: int = 10,
    group_lines_thresh_intercept: int = 100) -> tuple[list[Intersection], list[LineGroup], array_like]:
    """
    Find external borders of a playfield by detecting lines and their intersections.
    
    The function processes a binary mask to detect straight line segments using
    morphological operations, Canny edge detection, and Hough line transformation.
    It groups similar lines and finds their intersection points to define the
    playfield boundaries.
    
    Process:
        1. Apply morphological close operation to fill gaps in the binary mask
        2. Detect edges using Canny edge detection
        3. Find line segments using probabilistic Hough line transformation
        4. Group similar lines together
        5. Calculate intersection points between all pairs of line groups
        6. Create visualization with lines and intersection points
    
    Args:
        original_img: Original RGB image for visualization
        binary_mask: Binary mask image to process
        kernel_size: Size of the morphological kernel for closing operation (default (21, 21))
        canny_thresh_lower: Lower threshold for Canny edge detection (default 150)
        canny_thresh_upper: Upper threshold for Canny edge detection (default 200)
        hough_thresh: Minimum votes to detect a line in Hough transform (default 100)
        hough_min_line_len: Minimum line length in pixels (default 100)
        hough_max_line_gap: Maximum gap between line segments to connect (default 10)
        group_lines_thresh_intercept: Maximum intercept difference to group lines (default 100)
    
    Returns:
        Tuple containing:
            - list[Intersection]: List of intersection points between line groups
            - list[LineGroup]: List of grouped line objects representing detected borders
            - array_like: Visualization image with lines drawn in blue and intersection points in red
    
    Note:
        The function expects a binary mask where white pixels represent the area to analyze.
        Intersection points and line groups can be used to reconstruct the playfield boundary.
    """
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
    lines = group_lines(lines, thresh_intercept=group_lines_thresh_intercept)
    lines = _select_lines(lines)
    intersections = compute_intersections(lines, binary_mask)

    pic_copy = original_img.copy()
    for intersection, line in zip(intersections, lines):
        pt = intersection.point.as_int()
        end_pts = line.limit_to_img(pic_copy)
        cv2.line(pic_copy, *end_pts, (255, 0, 0), 2)
        cv2.circle(pic_copy, pt, 2,(0, 0, 255), -1)

                
    return intersections, lines, pic_copy, binary_mask_close


def blackout_pixels_outside_borders(
    intersections: list[Intersection],
    inv_binary_img: array_like,
    lines: list[LineGroup],
    line_buffer_distance: int = 5
    ) -> array_like:
    """
    Convert white pixels to black outside of the external playfield borders.
    
    Makes white pixels to black outside external borders. The function uses a two-step masking approach to isolate the playfield:
    1. Creates a convex hull from intersection points to define the main playfield area
    2. Applies line-based masking to black out a buffer zone on the opposite side of detected lines
    
    This approach helps isolate the actual playfield area by removing areas outside the detected
    borders while accounting for line detection artifacts.
    
    Args:
        intersections: List of Intersection objects representing corners/boundary points of the playfield
        inv_binary_img: Inverted binary image where detected playfield color is black
        lines: List of LineGroup objects representing detected border lines
        line_buffer_distance: Distance in pixels to black out on the opposite side of lines (default 5)
    
    Returns:
        Binary image with white pixels converted to black outside the playfield borders
    
    Note:
        The function draws black lines with thickness `line_buffer_distance * 2` to create a buffer
        zone around detected lines, effectively masking out areas on the opposite side of the lines.
        This helps remove artifacts and isolate the actual playfield area.
    """
    intersection_points = np.array([[int(inter.point.x), int(inter.point.y)] for inter in intersections])
    hull = cv2.convexHull(intersection_points)

    height, width = inv_binary_img.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask, [hull], 255)

    for line in lines:
        pts = line.limit_to_img(mask)
        if pts:
            pt1, pt2 = pts
            cv2.line(mask, (int(pt1.x), int(pt1.y)), (int(pt2.x), int(pt2.y)), 0, line_buffer_distance * 2)

    return cv2.bitwise_and(inv_binary_img, mask)


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

    lines = _convert_hough_segments_to_lines(segments)
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


def find_top_internal_cushion(
    blackout_img: array_like,
    hough_thresh: int = 100,
    hough_min_line_len: int = 200,
    hough_max_line_gap: int = 10,
    ) -> Line | None:
    """
    Find the top internal cushion of a playfield by detecting horizontal line segments.
    
    The function processes a blackout image (where playfield area has been isolated)
    to detect the top internal cushion using probabilistic Hough line transformation.
    It filters for near-horizontal lines (low slope) and selects the highest one
    (lowest intercept value) to represent the top cushion.
    
    Process:
        1. Apply probabilistic Hough line transformation to detect line segments
        2. Convert detected segments to Line objects
        3. Filter for near-horizontal lines (abs(slope) < 2) with valid intercept
        4. Sort lines by intercept value (ascending)
        5. Select the line with the lowest intercept (highest position in image)
    
    Args:
        blackout_img: Binary image with isolated playfield (white pixels represent playfield)
        hough_thresh: Minimum votes to detect a line in Hough transform (default 100)
        hough_min_line_len: Minimum line length in pixels (default 200)
        hough_max_line_gap: Maximum gap between line segments to connect (default 10)
    
    Returns:
        Line | None: Line object representing the top internal cushion (the line with
                    the lowest intercept value), or None if no suitable lines are found
    
    Note:
        The function is designed to detect the top horizontal cushion, which is typically
        a near-horizontal line. It filters for lines with small slope values (abs(slope) < 2)
        and selects the one positioned highest in the image (lowest intercept value).
    """

    segments = cv2.HoughLinesP(
        blackout_img, 
        1, 
        np.pi / 180, 
        threshold=hough_thresh, 
        minLineLength=hough_min_line_len, 
        maxLineGap=hough_max_line_gap
    )

    if segments is not None:
        lines = _convert_hough_segments_to_lines(segments)
        lines = [line for line in lines if line.intercept is not None and abs(line.slope) < 2]
        lines = sorted(lines, key=lambda line: line.intercept)
        return lines[0]
    else:
        return None


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
        lines = _convert_hough_segments_to_lines(segments)
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


def find_bottom_internal_cushion(
    original_image: array_like,
    intersection_points: np.ndarray,
    ) -> Line | None:
    """
    Find the bottom internal cushion of the playfield by detecting edge in the lower portion of cropped image.
    
    The function processes the bottom portion of a cropped playfield image (lower 10%) to detect
    the bottom internal cushion (the bottom boundary of the playing area). It uses CLAHE contrast
    enhancement, Gaussian blur, Sobel gradient detection, and peak finding to locate the horizontal
    edge between the darker playing surface and the lighter cushion (randa) border.
    
    Process:
        1. Crop the image to the bounding box defined by intersection points
        2. Extract the bottom 10% region of interest (ROI) from the cropped image
        3. Exclude bottom 10% of ROI to ignore dark border
        4. Convert ROI to HSV color space and extract channels
        5. Create green color mask to filter only green pixels (avoid objects in ROI)
        6. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        7. Apply bilateral filter to reduce noise
        8. Compute vertical gradient using Sobel operator to detect horizontal edges
        9. Apply green mask to gradient (only consider green pixels)
        10. Calculate row-wise gradient response using 90th percentile per row
        11. Smooth the gradient response using Gaussian convolution
        12. Find the row with maximum gradient response (strongest horizontal edge)
        13. Convert the edge position from cropped coordinates to global image coordinates
        14. Return a horizontal Line object representing the bottom internal cushion
    
    Args:
        original_image: Input image as numpy array or compatible array-like object
        intersection_points: Array of intersection points defining the bounding box for cropping.
                            Should be a numpy array with shape (N, 2) where each row is [x, y]
    
    Returns:
        Line | None: Line object representing the bottom internal cushion as a horizontal line
                    in global image coordinates, or None if detection fails. The line connects
                    the left and right edges of the image at the detected cushion height.
    
    Note:
        The bottom cushion is typically a near-horizontal edge that separates the darker green
        playing surface from the lighter green cushion (randa) border. The method works on the
        lower portion of the cropped playfield where this edge is most visible, using gradient
        analysis to find the strongest horizontal transition. Only green pixels are considered
        to avoid interference from objects that may be present in the ROI.
    """
    cropped_by_points, x_start, y_start = crop_image_by_points(original_image, intersection_points)

    H = cropped_by_points.height
    roi = cropped_by_points[int(0.95*H):] 

    hsv_img = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)  

    egdes = cv2.Canny(v, 10, 50)

    segments = cv2.HoughLinesP(egdes, 1, np.pi/180, 100, 100, 10) # 50

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