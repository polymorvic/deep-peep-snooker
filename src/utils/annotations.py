from abc import ABC, abstractmethod
import json
from typing import Literal, Type
from pathlib import Path

import numpy as np

from src.utils.const import BallColor
from src.utils.common import array_like
from src.utils.points import Point
from src.utils.schemas import ImageMetaData, Ball, ImageAnnotation, ImageBallAnnotation, BBox, ImagePlayfieldAnnotation


def transform_annotation(
    img: array_like, 
    annotation: list[list[float]] | np.ndarray
    ) -> list[Point] | np.ndarray:
    """Converts percentage-based annotations to pixel coordinates."""
    arr = np.array(annotation) * np.array([img.width, img.height]) / 100
    return [Point(x, y) for x, y in arr] if isinstance(annotation, list) else arr.astype(np.float32)


def transform_bbox(
    img: array_like,
    bbox: dict[str, float] | list[float] | np.ndarray,
) -> dict[str, float] | np.ndarray:
    """Converts percentage-based bbox (x,y,w,h) to pixel coordinates."""
    if isinstance(bbox, dict):
        sx, sy = img.width / 100, img.height / 100
        return {
            "x": float(bbox["x"] * sx),
            "y": float(bbox["y"] * sy),
            "width": float(bbox["width"] * sx),
            "height": float(bbox["height"] * sy),
        }
    arr = np.asarray(bbox, dtype=np.float32) * np.array([img.width, img.height, img.width, img.height], dtype=np.float32) / 100
    return arr


_RawAnnotations = list[dict[Literal['filename', 'data'], str | list]]
class AnnotationCollection[AT: ImageAnnotation](ABC):
    # Podklasy muszą ustawić ten atrybut (typ modelu Pydantic dla `cleaned_annotations`).
    annotation_model: Type[AT]

    def __init__(self, root_dir: Path, extension: str = 'json') -> None:
        self._root_dir: Path = Path(root_dir)
        self._raw_annotations: _RawAnnotations = self._concat_files(extension)
        self.cleaned_annotations: list[AT] = self._clean_annotations()

# classmethofd
#     init from raw
#     init from cleans

    @classmethod
    def from_cleans(cls, file_path: Path | str) -> "AnnotationCollection[AT]":
        file_path = Path(file_path)

        with file_path.open(encoding="utf-8") as f:
            raw_cleaned = json.load(f)

        cleaned = [cls.annotation_model.model_validate(item) for item in raw_cleaned]

        obj = cls.__new__(cls)
        obj._root_dir = file_path.parent
        obj._raw_annotations = []
        obj.cleaned_annotations = cleaned
        return obj


    @staticmethod
    @abstractmethod
    def display_on_image(annotation: AT, image: array_like) -> array_like:
        raise NotImplemented


    @abstractmethod
    def _clean_annotations(self) -> list[AT]:
        raise NotImplemented


    def _concat_files(self, extension: str = 'json') -> _RawAnnotations:
        if extension != 'json':
            raise NotImplemented
        
        raw_annotations = []
        for json_file in sorted(self._root_dir.glob(f'*.{extension}')):
            with open(json_file, 'r') as f:
                data = json.load(f)
                raw_annotations.append(
                    {
                        "filename": json_file.stem,
                        "data": data
                    }
                )  
        return raw_annotations


    def filter_by_image(self, image_name: str) -> AT:
        for ann in self.cleaned_annotations:
            if ann.image.name == image_name:
                return ann
            

    def save(self, file_path: Path | str) -> None:
        if not self.cleaned_annotations:
            print('No data to be saved')
            return

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = [item.model_dump() for item in self.cleaned_annotations]

        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)

        print('Data saved successfully')


    @staticmethod
    def _build_image_name(item: dict) -> str:
        image_path = item.get("data", {}).get("image", "")
        name = Path(image_path).name if image_path else item.get("file_upload", "")

        if "-" in name:
            return '-'.join(name.split("-", 1)[1:])
        return name


class BallAnnotationCollection(AnnotationCollection[ImageBallAnnotation]):

    @staticmethod
    def _extract_results(item: dict) -> list[dict]:
        annotations = item.get("annotations") or []
        if not annotations:
            return []
        return annotations[0].get("result") or []
    

    @staticmethod
    def _convert_item(source_filename: str, item: dict) -> ImageBallAnnotation:
        results = BallAnnotationCollection._extract_results(item)
        width = 0
        height = 0
        grouped: dict[str, list[BBox]] = {color: [] for color in BallColor}

        for result in results:
            value = result.get("value", {})
            labels = value.get("rectanglelabels") or []
            if not labels:
                continue

            color = labels[0]
            if color not in grouped:
                continue

            width = result.get("original_width", width)
            height = result.get("original_height", height)

            grouped[color].append(
                BBox(
                    x=value["x"],
                    y=value["y"],
                    width=value["width"],
                    height=value["height"],
                )
            )

        return ImageBallAnnotation(
            image=ImageMetaData(
                source_file_name = source_filename,
                name=BallAnnotationCollection._build_image_name(item),
                width=width,
                height=height,
            ),
            balls=[Ball(color=color, bboxes=grouped[color]) for color in BallColor],
        )


    def _clean_annotations(self) -> list[ImageBallAnnotation]:
        return [self._convert_item(data['filename'], item) for data in self._raw_annotations for item in data['data']]
    

    def validate(self) -> None:
        are_all_good = True
        for item in self.cleaned_annotations:
            for ball in item.balls:

                ball_count = len(ball.bboxes)
                is_invalid = False
                match ball.color:
                    case BallColor.RED:
                        if ball_count > 15:
                            is_invalid = True
                    case _:
                        if ball_count > 1:
                            is_invalid = True

                if is_invalid:
                    are_all_good = False
                    print(f'Uwaga liczba koloru {ball.color} jest równa {ball_count} na zdjeciu {item.image.name}')

        if are_all_good:
            print('Wszystkie oznaczenia są ok!')


    def display_on_image(annotation, image: array_like) -> array_like:
        return


class PlayfieldAnnotationCollection(AnnotationCollection[ImagePlayfieldAnnotation]):
    annotation_model = ImagePlayfieldAnnotation


    def _clean_annotations(self):
        cleaned_annotations = []

        for item in self._raw_annotations:
            ann_data = item['data']

            for subtitem in ann_data:
                result = subtitem['annotations'][0]['result'][0]

                annotation_data = ImagePlayfieldAnnotation(
                    image = ImageMetaData(
                        source_file_name = item['filename'],
                        name=PlayfieldAnnotationCollection._build_image_name(subtitem),
                        width=result['original_width'],
                        height=result['original_height'],
                    ),
                    points=result['value']['points'],
                )
                cleaned_annotations.append(annotation_data)

        return cleaned_annotations
    

    def display_on_image(annotation, image: array_like) -> array_like:
        return
    
