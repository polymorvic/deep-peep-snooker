from datetime import datetime
from typing import Literal
from pathlib import Path
from enum import StrEnum

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .annotations import transform_annotation, PolygonAnnotationData
from .common import NumpyImage
from .func import read_image_as_numpyimage
from .lines import Line
from .playfield_finder import PlayfieldFinder


TestType = Literal["bottom", "top", "left", "right", "iou"]


def prepare_test_results_report(
        dir: str | Path,
        data: list[dict],
        filename: str,
        agg_colname: str,
        not_found_pic_names: list[str],
        sheet_name1: str = 'results', 
        sheet_name2: str = 'stats') -> pd.DataFrame | None:
    
    if not data:
        raise ValueError('Input data is empty!')

    with pd.ExcelWriter(Path(dir) / f'{filename}.xlsx', engine='openpyxl') as writer:
        results_df = pd.DataFrame(data)
        results_df.to_excel(writer, sheet_name=sheet_name1, index=False)

        stats_df = (
            results_df[agg_colname]
            .agg(['median', 'mean', 'std', 'min', 'max'])
            .to_frame()
            .T
        )
        stats_df['not_found'] = len(not_found_pic_names)
        stats_df.to_excel(writer, sheet_name=sheet_name2)


def save_test_histogram(
    dir: str | Path,
    df: pd.DataFrame,
    colname: str,
    filename: str
    ) -> None:

    if colname not in df.columns:
        raise ValueError(f"Column '{colname}' not found in DataFrame.")

    series = df[colname].dropna()

    if series.empty:
        raise ValueError(f"Column '{colname}' is empty.")

    min_val = series.min()
    max_val = series.max()

    bins = np.arange(min_val, max_val + 1, 1)
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


def test_cushion(pic_filepath: Path, 
                polygon_ann: list[PolygonAnnotationData],
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

    y_sorted = np.sort(pts_gt[:, 1])

    if position == "bottom":
        y_ref = int(np.median(y_sorted[-2:]))
    else:
        y_ref = int(np.median(y_sorted[:2]))

    p1, p2 = line.limit_to_img(pic)

    img = cv2.cvtColor(pic, cv2.COLOR_RGB2BGR)
    cv2.line(img, p1, p2, (255, 0, 0), 1)
    cv2.imwrite(str(test_out_dir / f"test_{position}_{pic_name}"), img)

    return line, y_ref