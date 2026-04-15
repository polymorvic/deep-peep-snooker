import tyro
import torch
import numpy as np
from deep_peep_snooker.utils.annotations import transform_bbox, transform_annotation
from tqdm import tqdm

import deep_peep_snooker.config
from deep_peep_snooker.utils.testing import BallTestType, build_output_dir
from pathlib import Path
from deep_peep_snooker.utils.annotations import BallAnnotationCollection, PlayfieldAnnotationCollection
from deep_peep_snooker.utils.lines import Line
from deep_peep_snooker.schemas.ball import BallColor
from deep_peep_snooker.schemas.playfield import Playfield, PlayfieldLines, PlayfieldPoints
from deep_peep_snooker.utils.func import get_corners, read_image_as_numpyimage
from deep_peep_snooker.utils.ball_finder import BallFinder
from torchvision.ops import box_iou

def run(
    pics_dir: str | Path,
    playfield_annotation_file_path: str | Path,
    ball_annotation_file_path: str | Path,
    test_type: BallTestType,
    output_dir: str | Path = 'results/balls'
    ) -> None:
    '''
    uv run python -m tests.balls --pics-dir data/pics --playfield-annotation-file-path data/playfield_gt.json --ball-annotation-file-path data/ball_gt.json --test-type BLUE
    '''
    proj_cwd = Path.cwd()
    pics_dir = proj_cwd / pics_dir
    parent_dir = proj_cwd / output_dir
    test_out_dir = build_output_dir(parent_dir, test_type)


    playfield_annotations = PlayfieldAnnotationCollection.from_clean_file(file_path = playfield_annotation_file_path)
    ball_annotations = BallAnnotationCollection.from_clean_file(file_path = ball_annotation_file_path)
    output_filename = test_out_dir.parent.name

    results = []
    not_found = []

    for file in tqdm(sorted(pics_dir.glob("*.png"))):
        
        ball_not_exists = False
        ball_not_found = False

        if not file.name.startswith('pic_01'): # TODO usunac jak bedzie wiecej zdjec
            continue

        print(file.stem)

        img = read_image_as_numpyimage(file)

        playfield_ann = playfield_annotations.filter_by_image(file.name)
        if playfield_ann is None:
            continue

        points_px = np.asarray(transform_annotation(img, np.asarray(playfield_ann.points))).astype(int)
        lt, lb, rt, rb = get_corners(points_px)
        playfield = Playfield(
            lines=PlayfieldLines(
                top = Line.from_points(lt, rt),
                bottom = Line.from_points(lb, rb),
                left = Line.from_points(lb, lt),
                right = Line.from_points(rb, rt),
            ),
            points=PlayfieldPoints.from_numpy(lt, lb, rt, rb),
        )

        ball_bbox_gt = ball_annotations.get_bboxes_by_color(file.name, test_type.value)
        if not ball_bbox_gt:
            ball_not_exists = True
            print(f'na zdjeciu {file.stem} nie ma bili {test_type.value}')


        finder = BallFinder(img, playfield)
        ball_bbox_result = finder.find_blue()
        if ball_bbox_result is None:
            ball_not_found = True
            print(f'na zdjeciu {file.stem} nie znaleziono bili {test_type.value}')

        if ball_not_exists or ball_not_found:
            continue

        ball_bbox_result = ball_bbox_result.as_tensor.unsqueeze(0)

        ball_bbox_gt = transform_bbox(finder.img, ball_bbox_gt[0].as_numpy)
        ball_bbox_gt = torch.tensor(ball_bbox_gt).unsqueeze(0)

        print(box_iou(ball_bbox_gt, ball_bbox_result, fmt='xywh'))










if __name__ == '__main__':
    tyro.cli(run)