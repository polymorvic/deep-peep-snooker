import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Literal

from .common import NumpyImage

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

# def binarize_by_color(image: np.ndarray | NumpyImage, target_color: tuple[int, int, int], 
#                      tolerance: int = 1) -> np.ndarray:
#     """
#     Binarize an image based on a target HSV color with tolerance.
    
#     Args:
#         image: Input image as numpy array or NumpyImage (must be in HSV format)
#         target_color: Target color as (H,S,V) tuple
#         tolerance: Color tolerance for matching (default 1)
        
#     Returns:
#         Binary image as numpy array (0s and 255s)
        
#     Raises:
#         ValueError: If image is grayscale or doesn't have 3 channels
#     """
#     if image.ndim == 2:
#         raise ValueError("Image must be color (3 channels), not grayscale")
    
#     if image.ndim == 3 and image.shape[2] != 3:
#         raise ValueError("Image must have exactly 3 channels for color processing")
    

#     tolerance = int(tolerance)
#     target = np.array(target_color, dtype=np.uint8)
    
#     h_diff = np.abs(image[..., 0].astype(np.int16) - target[0])
#     h_diff = np.minimum(h_diff, 180 - h_diff) 
    
#     mask = (h_diff <= tolerance) & \
#            (np.abs(image[..., 1].astype(np.int16) - target[1]) <= tolerance) & \
#            (np.abs(image[..., 2].astype(np.int16) - target[2]) <= tolerance)
    
#     binary_image = np.where(mask, 255, 0).astype(np.uint8)
    
#     return binary_image


def apply_hough_transformation(
    img_gray: np.ndarray,
    blur_kernel_size: int = 5,
    canny_thresh_lower: int = 50,
    canny_thresh_upper: int = 150,
    hough_thresh: int = 100,
    hough_min_line_len_percent: float = 0.2,
    hough_max_line_gap: int = 10,
) -> tuple[np.ndarray, list]:
    """
    Apply probabilistic Hough line transformation to a grayscale image.
    
    Process:
        1. Apply Gaussian blur to reduce noise
        2. Detect edges using Canny edge detection
        3. Apply Hough line transformation to find straight lines
        4. Draw detected lines on the original image
    
    Args:
        img_gray: Input grayscale image
        blur_kernel_size: Size of Gaussian blur kernel (default 5)
        canny_thresh_lower: Lower threshold for Canny edge detection (default 50)
        canny_thresh_upper: Upper threshold for Canny edge detection (default 150)
        hough_thresh: Minimum votes to detect a line (default 100)
        hough_min_line_len_percent: Minimum line length as percentage of image height (default 0.1)
        hough_max_line_gap: Maximum gap between line segments (default 10)
        
    Returns:
        Tuple containing:
            - Image with detected lines drawn
            - List of detected lines (each line as [x1, y1, x2, y2])
            
    Raises:
        ValueError: If input image is not grayscale
    """
    if img_gray.ndim != 2:
        raise ValueError("Input image must be grayscale (2D array)")
    
    hough_min_line_len = int(img_gray.shape[0] * hough_min_line_len_percent)
    
    blurred = cv2.GaussianBlur(img_gray, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred, canny_thresh_lower, canny_thresh_upper)
    
    lines = cv2.HoughLinesP(
        edges, 
        1, 
        np.pi / 180, 
        threshold=hough_thresh, 
        minLineLength=hough_min_line_len, 
        maxLineGap=hough_max_line_gap
    )
    
    if lines is None:
        return cv2.cvtColor(img_gray.copy(), cv2.COLOR_GRAY2RGB), []
    
    result_img = cv2.cvtColor(img_gray.copy(), cv2.COLOR_GRAY2RGB)
    detected_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        detected_lines.append([x1, y1, x2, y2])
    
    return result_img, detected_lines