import cv2
import matplotlib.pyplot as plt

from .common import array_like
from .intersections import Intersection
from .lines import Line
from .points import Point


def display_img(img: array_like) -> None:
    """
    Display an image using matplotlib.
    
    This function provides a simple wrapper around matplotlib's imshow and show
    for displaying images. Uses the global matplotlib configuration set in src.config.
    
    Args:
        img: Image to display (numpy array or NumpyImage)
    
    Note:
        The default colormap is configured globally in src.config.py.
        To use a different colormap for a specific image, use matplotlib directly:
        plt.imshow(img, cmap='your_cmap')
    """
    plt.imshow(img)
    plt.show()


def plot_on_image(
    img: array_like, 
    intersections: list[Intersection] | None = None,
    lines: list[Line] | None = None, 
    points: list[Point] | None = None, 
    is_copy: bool = True,
    line_color: tuple[int, int, int] = (255, 0, 0),
    point_color: tuple[int, int, int] = (0, 0, 255),
    line_thickness: int = 2,
    point_radius: int = 3
    ) -> array_like:
    """
    Plot geometric objects (intersections, lines, points) on an image using OpenCV.
    
    This function draws geometric primitives on an image for visualization purposes.
    It supports drawing lines, points, and intersections with customizable colors,
    line thickness, and point radius.
    
    Args:
        img: Image to draw on (can be original or a copy)
        intersections: List of Intersection objects to visualize
        lines: List of Line objects to draw
        points: List of Point objects to mark
        is_copy: Whether to create a copy of the image before drawing (default True)
        line_color: RGB color for lines (default (255, 0, 0) - red)
        point_color: RGB color for points and circles (default (0, 0, 255) - blue)
        line_thickness: Thickness of the lines in pixels (default 2)
        point_radius: Radius of point circles in pixels (default 3)
    
    Returns:
        Image with geometric objects drawn on it
    
    Note:
        The function returns a copy by default to avoid modifying the original image.
        Set is_copy=False if you want to modify the original image in-place.
    """
    img_copy = img.copy() if is_copy else img

    if lines is not None:
        for line in lines:
            endpoints = line.limit_to_img(img_copy)
            if endpoints:
                cv2.line(img_copy, (int(endpoints[0].x), int(endpoints[0].y)), 
                        (int(endpoints[1].x), int(endpoints[1].y)), 
                        line_color, line_thickness)

    if points is not None:
        for point in points:
            point_int = point.as_int() if hasattr(point, 'as_int') else point
            cv2.circle(img_copy, (int(point_int.x), int(point_int.y)), 
                      point_radius, point_color, -1)

    if intersections is not None:
        for intersection in intersections:
            endpoints1 = intersection.line1.limit_to_img(img_copy)
            endpoints2 = intersection.line2.limit_to_img(img_copy)
            if endpoints1:
                cv2.line(img_copy, (int(endpoints1[0].x), int(endpoints1[0].y)), 
                        (int(endpoints1[1].x), int(endpoints1[1].y)), 
                        line_color, line_thickness)
            if endpoints2:
                cv2.line(img_copy, (int(endpoints2[0].x), int(endpoints2[0].y)), 
                        (int(endpoints2[1].x), int(endpoints2[1].y)), 
                        line_color, line_thickness)
            pt = intersection.point.as_int() if hasattr(intersection.point, 'as_int') else intersection.point
            cv2.circle(img_copy, (int(pt.x), int(pt.y)), point_radius, point_color, -1)

    return img_copy