"""Pydantic data models for annotation schemas."""

from pydantic import BaseModel


class ImageData(BaseModel):
    """Image metadata associated with an annotation.
    
    Attributes:
        name: Image filename.
        width: Original image width in pixels.
        height: Original image height in pixels.
    """
    name: str
    width: int
    height: int


class PolygonAnnotationData(BaseModel):
    """Polygon annotation data model.
    
    Represents a single polygon annotation with its associated image metadata.
    
    Attributes:
        points: List of [x, y] coordinate pairs defining the polygon vertices.
        image: Image metadata containing filename and dimensions.
    """
    points: list[list[float]]
    image: ImageData


class PolygonAnnotationList(BaseModel):
    """Container for a list of polygon annotations.
    
    Attributes:
        annotations: List of polygon annotations.
    """
    annotations: list[PolygonAnnotationData]
