from pydantic import BaseModel

from .const import BallColor


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


class ImageAnnotation(BaseModel):
    image: ImageMetaData


class ImageBallAnnotation(ImageAnnotation):
    balls: list[Ball]


class ImagePlayfieldAnnotation(ImageAnnotation):
    points: list[list[float]]

