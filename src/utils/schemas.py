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


class ImagePolygonAnnotation(ImageAnnotation):
    points: list[list[float]]






class PolygonAnnotationData(BaseModel):
    """Polygon annotation data model.
    
    Represents a single polygon annotation with its associated image metadata.
    
    Attributes:
        points: List of [x, y] coordinate pairs defining the polygon vertices.
        image: Image metadata containing filename and dimensions.
    """
    points: list[list[float]]
    image: ImageMetaData


class PolygonAnnotationList(BaseModel):
    """Container for a list of polygon annotations.
    
    Attributes:
        annotations: List of polygon annotations.
    """
    annotations: list[PolygonAnnotationData]
