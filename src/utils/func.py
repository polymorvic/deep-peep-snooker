import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal

from .common import NumpyImage
from .lines import Line, LineGroup


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
    blur_kernel_size: int = 5,
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