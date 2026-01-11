from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from src.utils.common import array_like, Annotation
from src.utils.points import Point
from src.utils.schemas import PolygonAnnotationData


def transform_annotation(
    img: array_like, 
    annotation: list[list[float]] | np.ndarray
    ) -> list[Point] | np.ndarray:
    """Converts percentage-based annotations to pixel coordinates."""
    arr = np.array(annotation) * np.array([img.width, img.height]) / 100
    return [Point(x, y) for x, y in arr] if isinstance(annotation, list) else arr.astype(np.float32)


class PolygonAnnotation(Annotation):
    """Annotation handler for polygon-shaped annotations.
    
    Processes raw annotation data containing polygon coordinates and extracts
    polygon points along with associated image metadata (filename, dimensions).
    
    The cleaned annotations contain:
        - 'points': List of polygon point coordinates
        - 'image': Dictionary with 'name', 'width', and 'height' keys
    """
    
    def __init__(self, root_dir: Path) -> None:
        """Initialize polygon annotation handler.
        
        Args:
            root_dir: Path to directory containing annotation JSON files.
        """
        super().__init__(root_dir)

    @property
    def clean_annotations(self) -> list[PolygonAnnotationData]:
        """Extract and clean polygon annotation data from raw annotations.
        
        Processes raw annotation dictionaries to extract:
        - Polygon point coordinates
        - Image filename (with prefix removed)
        - Original image dimensions
        
        Returns:
            List of validated PolygonAnnotationData models.
                
        Note:
            Invalid or malformed annotations are silently skipped. The result
            is cached after first computation. All returned data is validated
            using Pydantic models.
        """
        if self.cleaned_annotations is not None:
            return self.cleaned_annotations
        
        if not self.raw_annotations:
            self.cleaned_annotations = []
            return []
        
        cleaned_annotations = []
        for ann in self.raw_annotations:
            try:
                result = ann['annotations'][0]['result'][0]
                image_path = ann['data']['image']
                image_name = image_path.replace('\\', '/').split('/')[-1].split('-', 1)[-1]
                
                annotation_data = PolygonAnnotationData(
                    points=result['value']['points'],
                    image={
                        'name': image_name,
                        'width': result['original_width'],
                        'height': result['original_height']
                    }
                )
                cleaned_annotations.append(annotation_data)
            except (KeyError, IndexError, TypeError, ValidationError):
                continue
        
        self.cleaned_annotations = cleaned_annotations
        return cleaned_annotations