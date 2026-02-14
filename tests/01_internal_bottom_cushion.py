
import tyro

import src.config
from src.utils.testing import (prepare_test_results_report, save_test_histogram, TestType, 
                               build_output_dir, test_bottom)
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
            internal_bottom_cushion, y_ref = test_bottom(file, polygon_ann, test_out_dir)

        except Exception as e:
            print(f'Error processing {file}: {e}')
            not_found.append(file.stem)

        finally:

            results.append(
                {'pic_name': file.name, 
                'internal_bottom_cushion': internal_bottom_cushion, 
                'intercept_ref': y_ref,
                'intercept_pred': internal_bottom_cushion.intercept if internal_bottom_cushion is not None else None,
                }) 
            
    colname = 'diff'
    output_filename = test_out_dir.parent.name
    res_df = prepare_test_results_report(test_out_dir, results, output_filename, colname, not_found)
    save_test_histogram(test_out_dir, res_df, colname, f'{output_filename}-hist')


if __name__ == '__main__':
    tyro.cli(run)