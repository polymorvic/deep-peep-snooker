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

    if isinstance(arr, np.ndarray):
        h, w = arr.shape[:2]
    else: 
        h, w = arr.height, arr.width

    s = int(min(h, w) * percent)
    y, x = h // 2, w // 2
    r = s // 2
    return arr[y - r:y + r, x - r:x + r]


def read_image_as_numpyimage(path: str | Path, color_mode: Literal["RGB", "HSV", "GRAYSCALE"] = "RGB") -> NumpyImage:
    """
    Read an image and return it as a NumpyImage instance.
    color_mode: 'RGB', 'HSV', or 'GRAYSCALE'.
    """
    color_mode = color_mode.upper()
    if color_mode not in {"RGB", "HSV", "GRAYSCALE"}:
        raise ValueError("color_mode must be 'RGB', 'HSV', or 'GRAYSCALE'")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if color_mode == "GRAYSCALE" else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    conversions = {
        "RGB": lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),
        "HSV": lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV),
    }

    img = conversions.get(color_mode, lambda x: x)(img)
    return NumpyImage(img)