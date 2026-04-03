from pydantic import BaseModel

from deep_peep_snooker.utils.common import Hashable
from deep_peep_snooker.utils.const import BallColor


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

