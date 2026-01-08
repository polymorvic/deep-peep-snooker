import cv2
import numpy as np

from src.utils.const import SNOOKER_TABLE
from src.utils.points import Point


def ref_snooker_playfield(extra_length: int = 2_000, extra_width: int = 1_000, line_thickness: int = 250) -> tuple[np.ndarray, dict[str, Point]]:
    """
    Generate a reference snooker playfield image with key points.
    
    Args:
        extra_length: Additional pixels added to length (default: 2000)
        extra_width: Additional pixels added to width (default: 1000)
        line_thickness: Thickness of drawn lines in pixels (default: 250)
        
    Returns:
        Tuple containing the playfield image and dictionary of key points as Point objects
    """
    pf = SNOOKER_TABLE['playfield']
    baulk = SNOOKER_TABLE['baulk_line']

    length = int(pf['length_mm'] * 10)
    width = int(pf['width_mm'] * 10)
    baulk_line = int(baulk['dist_from_closer_cushion_mm'] * 10)
    baulk_radius = int(baulk['D_radius_mm'] * 10)

    img = np.zeros((length + extra_length, width + extra_width, 3), np.uint8)

    corners = {
        "top_left": (0, 0),
        "top_right": (width, 0),
        "bottom_left": (0, length),
        "bottom_right": (width, length)
    }

    lines = [
        (corners["top_left"], corners["top_right"]),
        (corners["top_left"], corners["bottom_left"]),
        (corners["top_right"], corners["bottom_right"]),
        (corners["bottom_left"], corners["bottom_right"]),
    ]

    for start, end in lines:
        cv2.line(img, start, end, (0, 255, 0), line_thickness)

    baulk_left, baulk_right = (0, baulk_line), (width, baulk_line)
    cv2.line(img, baulk_left, baulk_right, (0, 255, 0), line_thickness)

    baulk_center = (width // 2, baulk_line)
    cv2.ellipse(img, baulk_center, (baulk_radius, baulk_radius),
                0, 180, 360, (0, 255, 0), line_thickness)

    ref_points = {
        "top_left": Point.from_xy(0, 0),
        "top_right": Point.from_xy(width, 0),
        "bottom_left": Point.from_xy(0, length),
        "bottom_right": Point.from_xy(width, length),
        "center": Point.from_xy(width // 2, length // 2),
        "baulk_left": Point.from_xy(0, baulk_line),
        "baulk_right": Point.from_xy(width, baulk_line),
        "baulk_center": Point.from_xy(width // 2, baulk_line),
        "baulk_arc_left": Point.from_xy(width // 2 - baulk_radius, baulk_line),
        "baulk_arc_right": Point.from_xy(width // 2 + baulk_radius, baulk_line),
    }

    return img, ref_points