import cv2
import tyro
from datetime import datetime
from src.utils.func import read_image_as_numpyimage, get_corners
from src.utils.playfield_finder import PlayfieldFinder
import numpy as np
from src.utils.plotting import plot_on_image
import src.config
from src.utils.metrics import iou
from src.utils.intersections import compute_intersections
from src.utils.annotations import transform_annotation
from pathlib import Path
import pandas as pd
from src.utils.annotations import PolygonAnnotation
import matplotlib.pyplot as plt


def run(
    pics_dir: str,
    poly_annotations_dir: str,
    poly_annotations_file: str,
    output_dir: str
    ) -> None:
    """
    Test the IOU of the polygon annotations.
    calling:
        python 05_test_iou.py --pics-dir pics   --poly-annotations-dir playfield_gt   --poly-annotations-file all_new   --output-dir tests/iou_tests
    """
    now_str = datetime.now().strftime('%Y-%m-%d %H_%M_%S')
    test_root_dir = Path(f'{output_dir}/{now_str}')
    test_root_dir.mkdir(exist_ok=True)

    polygon_ann = PolygonAnnotation(root_dir=poly_annotations_dir)
    polygon_ann.read(Path(f'{poly_annotations_dir}/{poly_annotations_file}.json'))

    not_found = []
    iou_results = []
    results = []
    root = Path(pics_dir)
    for file in sorted(root.glob('*.png')):
        try:
            pic = read_image_as_numpyimage(file, 'rgb')
            pic_copy = pic.copy()

            data = polygon_ann.filter_by_image(file.name)
            points_gt = np.asarray(transform_annotation(pic, data.points))
            gt_left_top, gt_left_bottom, gt_right_top, gt_right_bottom = [(int(point[0]), int(point[1])) for point in get_corners(points_gt)]

            pic_copy = plot_on_image(pic_copy, polygons=[[gt_left_top, gt_left_bottom, gt_right_bottom, gt_right_top]])

            finder = PlayfieldFinder(pic)
            bottom_cushion = finder.find_bottom_internal_cushion()
            top_cushion = finder.find_top_internal_cushion()
            side_lines = finder.find_internal_side_cushions()

            all_lines = [bottom_cushion, top_cushion, *side_lines]
            internal_intersection = compute_intersections(all_lines, pic)
            lt, lb, rt, rb = get_corners(PlayfieldFinder.intersection_to_points_array(internal_intersection))

            pic_copy = plot_on_image(pic_copy, polygons=[[lt, lb, rb, rt]], polygon_color=(255, 0, 0))
            iou_result = iou(points_gt, [lt, lb, rb, rt])

            iou_results.append(iou_result)

            results.append({
                'pic_name': file.stem,
                'iou': iou_result,
            })

            iou_text = f"IOU: {iou_result:.3f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2

            pic_bgr = cv2.cvtColor(pic_copy, cv2.COLOR_RGB2BGR)
            text_color_bgr = (0, 255, 0)
            (text_width, text_height), baseline = cv2.getTextSize(iou_text, font, font_scale, thickness)
            

            cv2.rectangle(pic_bgr, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 10), (0, 0, 0), -1)
            cv2.putText(pic_bgr, iou_text, (15, 10 + text_height), font, font_scale, text_color_bgr, thickness)
            pic_copy = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2RGB)

            cv2.imwrite(f'{test_root_dir}/{file.stem}.png', cv2.cvtColor(pic_copy, cv2.COLOR_BGR2RGB))

        except Exception as e:
            print(file.stem, e)
            not_found.append(file.stem)
            results.append({
                'pic_name': file.stem,
                'iou': None,
            })

    with pd.ExcelWriter(f'{test_root_dir}/_iou_results.xlsx', engine='openpyxl') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='results', index=False)
        pd.DataFrame({
            'median': [np.median(iou_results)],
            'mean': [np.mean(iou_results)],
            'std': [np.std(iou_results)],
            'min': [np.min(iou_results)],
            'max': [np.max(iou_results)],
            'not_found': [len(not_found)],
        }).to_excel(writer, sheet_name='stats', index=False)

    plt.hist(iou_results, edgecolor='black')
    plt.savefig(f'{test_root_dir}/_iou_hist.png')


if __name__ == '__main__':
    tyro.cli(run)

