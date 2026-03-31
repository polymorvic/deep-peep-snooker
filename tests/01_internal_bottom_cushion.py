import tyro

import src.config
from src.utils.testing import (prepare_test_results_report, save_test_histogram, TestType, 
                               build_output_dir, test_cushion)
from pathlib import Path
from src.utils.annotations import PlayfieldAnnotationCollection


def run(
    pics_dir: str | Path,
    poly_annotations_file_path: str | Path,
    test_type: TestType,
    parent_dir: str | Path = 'results'
    ) -> None:
    '''
    uv run python -m tests.01_internal_bottom_cushion --pics-dir data/pics --poly_annotations-file-path data/playfield_gt.json --test-type BOTTOM
    '''
    proj_cwd = Path.cwd()
    pics_dir =  proj_cwd / pics_dir
    parent_dir = proj_cwd / parent_dir
    test_out_dir = build_output_dir(parent_dir, test_type)

    polygon_ann = PlayfieldAnnotationCollection.from_clean_file(file_path = poly_annotations_file_path)
    results = []
    not_found = []
    for file in sorted(pics_dir.glob('*.png'))[:5]:
        
        try:
            pred_line, y_ref = test_cushion(file, polygon_ann, test_out_dir, test_type.lower())

        except Exception as e:
            print(f'Error processing {file}: {e}')
            not_found.append(file.stem)
            pred_line, y_ref = None, None

        finally:

            results.append(
                {
                    'pic_name': file.name, 
                    'pred_line_intercept': pred_line, 
                    'intercept_ref': y_ref,
                    'intercept_pred': pred_line.intercept if pred_line is not None else None,
                }) 
            
    output_filename = test_out_dir.parent.name
    res_df = prepare_test_results_report(test_out_dir, results, output_filename, not_found)
    save_test_histogram(test_out_dir, res_df, 'measure', f'{output_filename}-hist')


if __name__ == '__main__':
    tyro.cli(run)