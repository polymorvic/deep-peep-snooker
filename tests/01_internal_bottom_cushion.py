
import tyro

import src.config
from src.utils.testing import (prepare_test_results_report, save_test_histogram, TestType, 
                               build_output_dir, test_cushion)
from pathlib import Path
import pandas as pd
from src.utils.annotations import PolygonAnnotation
import matplotlib.pyplot as plt



def run(
    pics_dir: str | Path,
    poly_annotations_dir: str,
    poly_annotations_file: str,
    test_type: TestType,
    parent_dir: str | Path = 'tests/results'
    ) -> None:
# uv run python -m tests.01_internal_bottom_cushion --pics-dir pics   --poly-annotations-dir playfield_gt   --poly-annotations-file all_new2   --test-type BOTTOM
    proj_cwd = Path.cwd()
    pics_dir =  proj_cwd / pics_dir
    parent_dir = proj_cwd / parent_dir
    test_out_dir = build_output_dir(parent_dir, test_type)

    polygon_ann = PolygonAnnotation(root_dir=poly_annotations_dir)
    polygon_ann.read(Path(f'{poly_annotations_dir}/{poly_annotations_file}.json'))


    results = []
    not_found = []
    for file in sorted(pics_dir.glob('*.png')):
        try:
            pred_line, y_ref = test_cushion(file, polygon_ann, test_out_dir, test_type.lower())

        except Exception as e:
            print(f'Error processing {file}: {e}')
            not_found.append(file.stem)

        finally:

            results.append(
                {
                    'pic_name': file.name, 
                    'pred_line': pred_line, 
                    'intercept_ref': y_ref,
                    'intercept_pred': pred_line.intercept if pred_line is not None else None,
                }) 
            
    colname = 'diff'
    output_filename = test_out_dir.parent.name
    res_df = prepare_test_results_report(test_out_dir, results, output_filename, colname, not_found)
    save_test_histogram(test_out_dir, res_df, colname, f'{output_filename}-hist')


if __name__ == '__main__':
    tyro.cli(run)