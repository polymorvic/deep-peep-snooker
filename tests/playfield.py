import tyro
from tqdm import tqdm

import deep_peep_snooker.config
from deep_peep_snooker.utils.testing import (prepare_test_results_report, save_test_histogram, PlayfieldTestType, 
                               build_output_dir, test_cushion, test_iou, prepare_single_metric_report)
from pathlib import Path
from deep_peep_snooker.utils.annotations import PlayfieldAnnotationCollection


def run(
    pics_dir: str | Path,
    annotation_file_path: str | Path,
    test_type: PlayfieldTestType,
    output_dir: str | Path = 'results/playfield'
    ) -> None:
    '''
    uv run python -m tests.playfield --pics-dir data/pics --annotation-file-path data/playfield_gt.json --test-type BOTTOM
    '''
    proj_cwd = Path.cwd()
    pics_dir =  proj_cwd / pics_dir
    output_dir = proj_cwd / output_dir
    output_dir = build_output_dir(output_dir, test_type)

    annotations = PlayfieldAnnotationCollection.from_clean_file(file_path = annotation_file_path)
    output_filename = output_dir.parent.name

    results = []
    not_found = []
    for file in tqdm(sorted(pics_dir.glob("*.png"))):
        try:
            if test_type is PlayfieldTestType.IOU:
                iou_result = test_iou(file, annotations, output_dir)
                results.append({"pic_name": file.name, "iou": iou_result})
                
            else:
                pred_line, y_ref = test_cushion(file, annotations, output_dir, test_type.lower())
                results.append(
                    {
                        "pic_name": file.name,
                        "pred_line_intercept": pred_line,
                        "intercept_ref": y_ref,
                        "intercept_pred": pred_line.intercept if pred_line is not None else None,
                    }
                )
        except Exception as e:
            print(f"Error processing {file}: {e}")
            not_found.append(file.stem)
            if test_type is PlayfieldTestType.IOU:
                results.append({"pic_name": file.name, "iou": None})
            else:
                results.append(
                    {
                        "pic_name": file.name,
                        "pred_line_intercept": None,
                        "intercept_ref": None,
                        "intercept_pred": None,
                    }
                )

    if test_type is PlayfieldTestType.IOU:
        res_df = prepare_single_metric_report(
            output_dir, results, output_filename, "iou", not_found
        )
        if res_df["iou"].notna().any():
            save_test_histogram(output_dir, res_df, "iou", f"{output_filename}-hist")
    else:
        res_df = prepare_test_results_report(output_dir, results, output_filename, not_found)
        save_test_histogram(output_dir, res_df, "measure", f"{output_filename}-hist")


if __name__ == '__main__':
    tyro.cli(run)