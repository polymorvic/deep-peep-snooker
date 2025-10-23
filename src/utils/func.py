import cv2
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
    color_mode: 'rgb', 'hsv', or 'grayscale'.
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


