import numpy as np
from shapely.geometry import Polygon

from .points import Point

def _reorder_pts(ref: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Reorder points to match reference order by finding closest matches."""
    return pts[np.argmin(np.linalg.norm(ref[:, None] - pts, axis=2), axis=1)]

def iou(
    ref_points: list[list[float]] | list[Point] | np.ndarray,
    points: list[list[float]] | list[Point] | np.ndarray,
    ) -> float:
    """
    Calculate Intersection over Union (IoU) between two 2D polygons.

    Args:
        ref_points: Reference polygon as 2D list, list of Point instances, or numpy array of shape (n_points, 2).
        points: Polygon to compare as 2D list, list of Point instances, or numpy array of shape (n_points, 2).

    Returns:
        float: IoU value between 0.0 and 1.0, where 1.0 means perfect overlap.
    """
    ref_points = np.asarray(ref_points)
    points = np.asarray(points)

    points = _reorder_pts(ref_points, points)

    ref_polygon = Polygon(ref_points)
    polygon = Polygon(points)

    intersection = ref_polygon.intersection(polygon).area
    union = ref_polygon.union(polygon).area

    return intersection / union if union > 0 else 0.0


def relative_line_distance(a_true, b_true, a_est, b_est, eps=1e-12):
    da = (a_est - a_true) / (abs(a_true) + eps)
    db = (b_est - b_true) / (abs(b_true) + eps)
    return np.sqrt(da**2 + db**2)
