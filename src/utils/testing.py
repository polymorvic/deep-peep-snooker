from datetime import datetime
from typing import Literal
from pathlib import Path
from enum import StrEnum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .annotations import transform_annotation, PlayfieldAnnotationCollection
from .common import NumpyImage
from .func import read_image_as_numpyimage, get_corners
from .intersections import compute_intersections
from .lines import Line
from .metrics import iou
from .playfield_finder import PlayfieldFinder
from .plotting import plot_on_image


TestType = Literal["bottom", "top", "left", "right", "iou"]


def prepare_test_results_report(
        dir: str | Path,
        data: list[dict],
        filename: str,
        not_found_pic_names: list[str],
        sheet_name1: str = 'results', 
        sheet_name2: str = 'stats') -> pd.DataFrame | None:
    
    if not data:
        raise ValueError('Input data is empty!')

    with pd.ExcelWriter(Path(dir) / f'{filename}.xlsx', engine='openpyxl') as writer:
        results_df = pd.DataFrame(data)

        diff = results_df['intercept_ref'] - results_df['intercept_pred']
        results_df['measure'] = diff
        results_df['abs_measure'] = diff.abs()

        results_df.to_excel(writer, sheet_name=sheet_name1, index=False)

        stats_df = (
            results_df['measure']
            .agg(['median', 'mean', 'std', 'min', 'max'])
            .to_frame()
            .T
        )
        stats_df['not_found'] = len(not_found_pic_names)
        stats_df.to_excel(writer, sheet_name=sheet_name2, index=False)

    return results_df


def prepare_single_metric_report(
        dir: str | Path,
        data: list[dict],
        filename: str,
        metric_col: str,
        not_found_pic_names: list[str],
        sheet_name1: str = "results",
        sheet_name2: str = "stats",
    ) -> pd.DataFrame:
    if not data:
        raise ValueError("Input data is empty!")

    results_df = pd.DataFrame(data)
    valid_series = pd.to_numeric(results_df[metric_col], errors="coerce").dropna()

    stats_df = pd.DataFrame(
        [{
            "median": valid_series.median() if not valid_series.empty else np.nan,
            "mean": valid_series.mean() if not valid_series.empty else np.nan,
            "std": valid_series.std(ddof=0) if not valid_series.empty else np.nan,
            "min": valid_series.min() if not valid_series.empty else np.nan,
            "max": valid_series.max() if not valid_series.empty else np.nan,
            "not_found": len(not_found_pic_names),
        }]
    )

    with pd.ExcelWriter(Path(dir) / f"{filename}.xlsx", engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name=sheet_name1, index=False)
        stats_df.to_excel(writer, sheet_name=sheet_name2, index=False)

    return results_df


def save_test_histogram(
    dir: str | Path,
    df: pd.DataFrame,
    colname: str,
    filename: str
    ) -> None:

    if colname not in df.columns:
        raise ValueError(f"Column '{colname}' not found in DataFrame.")

    series = df[colname].dropna()

    if series.dtype == "object" and not series.empty and isinstance(series.iloc[0], Line):
        series = series.map(lambda l: l.intercept).dropna()
    else:
        series = pd.to_numeric(series, errors="coerce").dropna()

    if series.empty:
        raise ValueError(f"Column '{colname}' is empty.")

    data = series.to_numpy()
    is_int_like = np.allclose(data, np.round(data))
    bins = np.arange(int(data.min()), int(data.max()) + 2, 1) if is_int_like else "auto"
    output_path = Path(dir) / f"{filename}.png"

    plt.figure()
    plt.hist(series, bins=bins, edgecolor="black")
    plt.xlabel(colname)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {colname}")
    plt.savefig(output_path)
    plt.close()


class TestType(StrEnum):
    BOTTOM = "bottom"
    TOP = "top"
    LEFT = "left"
    RIGHT = "right"
    IOU = "iou"

    @property
    def subdir(self) -> str:
        if self is TestType.IOU:
            return "iou"
        return f"internal-{self.value}-cushion"
    
    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


def build_output_dir(parent_dir: str | Path, test_type: TestType) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(parent_dir) / test_type.subdir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _compute_y_ref(points: np.ndarray, position: str) -> int:
    y_sorted = np.sort(points[:, 1])
    selected = y_sorted[-2:] if position == "bottom" else y_sorted[:2]
    return int(np.median(selected))


def test_cushion(
        pic_filepath: Path, 
        polygon_ann: PlayfieldAnnotationCollection,
        test_out_dir: Path,
        position: Literal["top", "bottom"]
    ) -> tuple[Line, int]:

    pic_name = pic_filepath.name
    pic = read_image_as_numpyimage(pic_filepath, "rgb")
    finder = PlayfieldFinder(pic)

    if position == "bottom":
        line = finder.find_bottom_internal_cushion()
    else:
        line = finder.find_top_internal_cushion()

    data = polygon_ann.filter_by_image(pic_name)
    pts_gt = np.asarray(transform_annotation(pic, data.points))

    y_ref = _compute_y_ref(pts_gt, position)

    p1, p2 = line.limit_to_img(pic)

    img = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
    cv2.line(img, p1, p2, (255, 0, 0), 1)
    cv2.imwrite(str(test_out_dir / f"test_{position}_{pic_name}"), img)

    return line, y_ref


def test_iou(
        pic_filepath: Path,
        polygon_ann: PlayfieldAnnotationCollection,
        test_out_dir: Path,
    ) -> float:
    pic_name = pic_filepath.name
    pic = read_image_as_numpyimage(pic_filepath, "rgb")
    pic_copy = pic.copy()

    data = polygon_ann.filter_by_image(pic_name)
    points_gt = np.asarray(transform_annotation(pic, data.points))
    gt_left_top, gt_left_bottom, gt_right_top, gt_right_bottom = [
        (int(point[0]), int(point[1])) for point in get_corners(points_gt)
    ]
    pic_copy = plot_on_image(
        pic_copy,
        polygons=[[gt_left_top, gt_left_bottom, gt_right_bottom, gt_right_top]],
    )

    finder = PlayfieldFinder(pic)
    bottom_cushion = finder.find_bottom_internal_cushion()
    top_cushion = finder.find_top_internal_cushion()
    side_lines = finder.find_internal_side_cushions()

    all_lines = [bottom_cushion, top_cushion, *side_lines]
    internal_intersections = compute_intersections(all_lines, pic)
    lt, lb, rt, rb = get_corners(
        PlayfieldFinder.intersection_to_points_array(internal_intersections)
    )

    pic_copy = plot_on_image(
        pic_copy,
        polygons=[[lt, lb, rb, rt]],
        polygon_color=(255, 0, 0),
    )
    iou_result = iou(points_gt, [lt, lb, rb, rt])

    iou_text = f"IOU: {iou_result:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    pic_bgr = cv2.cvtColor(pic_copy, cv2.COLOR_RGB2BGR)
    (text_width, text_height), baseline = cv2.getTextSize(
        iou_text, font, font_scale, thickness
    )
    cv2.rectangle(
        pic_bgr,
        (10, 10),
        (10 + text_width + 10, 10 + text_height + baseline + 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        pic_bgr,
        iou_text,
        (15, 10 + text_height),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
    )
    cv2.imwrite(str(test_out_dir / f"test_iou_{pic_name}"), pic_bgr)

    return float(iou_result)