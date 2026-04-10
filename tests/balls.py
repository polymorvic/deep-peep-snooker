import tyro
from tqdm import tqdm

import deep_peep_snooker.config
from deep_peep_snooker.utils.testing import (prepare_test_results_report, save_test_histogram, TestType, 
                               build_output_dir, test_cushion, test_iou, prepare_single_metric_report)
from pathlib import Path
from deep_peep_snooker.utils.annotations import BallAnnotationCollection

from deep_peep_snooker.utils.playfield_finder import PlayfieldFinder
from deep_peep_snooker.utils.ball_finder import BallFinder
from deep_peep_snooker.utils.func import read_image_as_numpyimage

def run(
    pics_dir: str | Path,
    annotation_file_path: str | Path,
    test_type: TestType,
    parent_dir: str | Path = 'results/balls'
    ) -> None:
    '''
    uv run python -m tests.playfield --pics-dir data/pics --annotation-file-path data/playfield_gt.json --test-type BOTTOM
    '''
    proj_cwd = Path.cwd()
    pics_dir =  proj_cwd / pics_dir
    parent_dir = proj_cwd / parent_dir
    test_out_dir = build_output_dir(parent_dir, test_type)

    annotations = BallAnnotationCollection.from_clean_file(file_path = annotation_file_path)
    output_filename = test_out_dir.parent.name

    results = []
    not_found = []
    for file in tqdm(sorted(pics_dir.glob("*.png"))):

        img = read_image_as_numpyimage(file)

        ann = annotations.filter_by_image(file.stem)

        if ann is None:
            continue


        try:
            finder = PlayfieldFinder(img)
            playfield = finder.detect_inner_playfield()

            finder = BallFinder(img, playfield)
            x, y, w, h = finder.find_blue()

        except Exception as e:
            pass








if __name__ == '__main__':
    tyro.cli(run)