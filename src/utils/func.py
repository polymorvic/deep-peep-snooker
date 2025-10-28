import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal

from .common import NumpyImage
from .lines import Line, LineGroup
from .intersections import Intersection

type array_like = np.ndarray | NumpyImage


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


def group_lines(
    lines: list[Line], thresh_theta: float | int = 5, thresh_intercept: float | int = 10
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
    edges = cv2.Canny(binary_mask_close, canny_thresh_lower, canny_thresh_upper)
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

    intersections = set()
    for group1 in lines:
        for group2 in lines:
            intersection = group1.intersection(group2, binary_mask)
            if intersection is not None:
                intersections.add(intersection)

    pic_copy = original_img.copy()
    for intersection, line in zip(intersections, lines):
        pt = intersection.point.as_int()
        end_pts = line.limit_to_img(pic_copy)
        cv2.line(pic_copy, *end_pts, (255, 0, 0), 2)
        cv2.circle(pic_copy, pt, 2,(0, 0, 255), -1)
                
    return list(intersections), lines, pic_copy


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
