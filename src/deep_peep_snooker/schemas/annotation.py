import numpy as np
import torch

from pydantic import BaseModel

from deep_peep_snooker.utils.common import Hashable, ArrayLike
from deep_peep_snooker.utils.const import BallColor
from deep_peep_snooker.utils.annotations import transform_bbox


class ImageMetaData(BaseModel):
    name: str
    width: int
    height: int
    source_file_name: str


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


    @property
    def as_list(self) -> list:
        return [self.x, self.y, self.width, self.height]


    @property
    def as_numpy(self) -> np.ndarray:
        return np.array(self.as_list)
    

    @property
    def as_tensor(self) -> torch.Tensor:
        return torch.tensor(self.as_list)
    

    @property
    def as_2d_tensor(self) -> torch.Tensor:
        return self.as_tensor.unsqueeze(0)
    
    
    def transform_to_img(self, img: ArrayLike) -> torch.Tensor:
        bbox_gt = transform_bbox(img, self.as_numpy)
        return torch.tensor(bbox_gt).unsqueeze(0)


class Ball(BaseModel):
    color: BallColor
    bboxes: list[BBox]


class ImageAnnotation(Hashable, BaseModel):
    image: ImageMetaData


class ImageBallAnnotation(ImageAnnotation):
    balls: list[Ball]

    def _key_(self) -> tuple:
        image_key = (
            self.image.source_file_name,
            self.image.name,
            self.image.width,
            self.image.height,
        )

        balls_key = tuple(
            (ball.color, tuple((bb.x, bb.y, bb.width, bb.height) for bb in ball.bboxes))
            for ball in self.balls
        )
        return image_key, balls_key

    def key(self) -> tuple:
        return self._key_()


class ImagePlayfieldAnnotation(ImageAnnotation):
    points: list[list[float]]

    def _key_(self) -> tuple:
        image_key = (
            self.image.source_file_name,
            self.image.name,
            self.image.width,
            self.image.height,
        )
        points_key = tuple((point[0], point[1]) for point in self.points)
        return image_key, points_key

    def key(self) -> tuple:
        return self._key_()

