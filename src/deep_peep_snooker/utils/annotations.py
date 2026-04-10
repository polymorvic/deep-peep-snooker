from abc import ABC, abstractmethod
import json
from typing import Literal, Type, Self
from pathlib import Path

import numpy as np
import cv2

from deep_peep_snooker.utils.const import BallColor
from deep_peep_snooker.utils.common import ArrayLike, NumpyImage
from deep_peep_snooker.utils.points import Point
from deep_peep_snooker.schemas.annotation import ImageMetaData, Ball, ImageAnnotation, ImageBallAnnotation, BBox, ImagePlayfieldAnnotation


def transform_annotation(
    img: ArrayLike, 
    annotation: list[list[float]] | np.ndarray
    ) -> list[Point] | np.ndarray:
    """Converts percentage-based annotations to pixel coordinates."""
    arr = np.array(annotation) * np.array([img.width, img.height]) / 100
    return [Point(x, y) for x, y in arr] if isinstance(annotation, list) else arr.astype(np.float32)


def transform_bbox(
    img: ArrayLike,
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
    annotation_model: Type[AT]

    def __init__(self, 
                 root_dir: Path | str, 
                 raw_annotations: _RawAnnotations | None = None,
                 cleaned_annotations: dict[str, AT] | None = None
                ) -> None:
        self._root_dir: Path = Path(root_dir)
        self._raw_annotations = raw_annotations
        self.cleaned_annotations = cleaned_annotations


    @classmethod
    def from_raw_dir(cls, root_dir: Path | str, extension: str = 'json') -> Self:
        obj = cls(root_dir)
        obj._raw_annotations = obj._concat_files(extension)
        obj.cleaned_annotations = obj._clean_annotations()
        return obj


    @classmethod
    def from_clean_file(cls, file_path: Path | str) -> Self:
        file_path = Path(file_path)

        with file_path.open(encoding="utf-8") as f:
            raw_cleaned = json.load(f)

        cleaned = {item['image']['name']: cls.annotation_model.model_validate(item) for item in raw_cleaned}
        return cls(file_path.parent, [], cleaned)


    @staticmethod
    @abstractmethod
    def display_on_image(annotation: AT, image: ArrayLike) -> ArrayLike:
        raise NotImplemented


    @abstractmethod
    def _clean_annotations(self) -> dict[str, AT]:
        raise NotImplemented
    

    @abstractmethod
    def _validate(self) -> None:
        raise NotImplemented
    

    def validate(self) -> None:
        self._validate()
        self._check_duplicates()
    

    def _check_duplicates(self) -> None:
        if not self.cleaned_annotations:
            raise ValueError('Clean annotations not set')
        
        duplicates = set()
        cleaned_annotations = list(self.cleaned_annotations.values())
        for i in range(len(cleaned_annotations)):
            for j in range(len(cleaned_annotations)):

                if i == j:
                    continue

                if cleaned_annotations[i] == cleaned_annotations[j]:
                    duplicates.add(cleaned_annotations[j])

        if not duplicates:
            print('Nie ma duplikatow')

        else:
            print(f'Są duplikaty {len(duplicates)} :')
            for item in duplicates:
                print(item.image.name)


    def remove_duplicates(self) -> None:
        if self.cleaned_annotations is None:
            raise ValueError('Clean annotations not set')

        unique_items: dict[str, AT] = {}
        for img_name, item in self.cleaned_annotations.items():
            unique_items[img_name] = item

        removed_count = len(self.cleaned_annotations) - len(unique_items)
        self.cleaned_annotations = unique_items
        print(f'Usunięto {removed_count} duplikatów')


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
        return self.cleaned_annotations[image_name]
            

    def save(self, file_path: Path | str) -> None:
        if not self.cleaned_annotations:
            print('No data to be saved')
            return

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = [item.model_dump() for item in self.cleaned_annotations.values()]

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
    annotation_model = ImageBallAnnotation


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


    def _clean_annotations(self) -> dict[str, ImageBallAnnotation]:
        return {self._build_image_name(item): self._convert_item(data['filename'], item) for data in self._raw_annotations for item in data['data']}
    

    def _validate(self) -> None:
        are_all_good = True
        for item in self.cleaned_annotations.values():
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


    @staticmethod
    def display_on_image(annotation: ImageBallAnnotation, image: ArrayLike, frame_color: tuple[int] = (255, 0, 0)) -> ArrayLike:
        img = np.asarray(image).copy()
        nim = NumpyImage(img)

        for ball in annotation.balls:
            for bb in ball.bboxes:
                px = transform_bbox(nim, bb.model_dump())
                x1 = int(round(px["x"]))
                y1 = int(round(px["y"]))
                x2 = int(round(px["x"] + px["width"]))
                y2 = int(round(px["y"] + px["height"]))
                cv2.rectangle(img, (x1, y1), (x2, y2), frame_color, 2)

        return img
    

    def get_bboxes_by_color(self, image_name: str, color: BallColor) -> list[BBox]:
        self.filter_by_image(image_name)


class PlayfieldAnnotationCollection(AnnotationCollection[ImagePlayfieldAnnotation]):
    annotation_model = ImagePlayfieldAnnotation


    def _clean_annotations(self) -> dict[str, ImagePlayfieldAnnotation]:
        cleaned_annotations = {}

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
                cleaned_annotations[self._build_image_name(item)] = annotation_data

        return cleaned_annotations
    

    def _validate(self) -> None:
        are_all_good = True

        for item in self.cleaned_annotations.values():
            pts = item.points
            is_invalid = False

            if not isinstance(pts, list) or len(pts) != 4:
                is_invalid = True
            else:
                flat_vals: list[float] = []
                for p in pts:
                    if not isinstance(p, list) or len(p) != 2:
                        is_invalid = True
                        break
                    for v in p:
                        if not isinstance(v, (int, float)):
                            is_invalid = True
                            break
                        fv = float(v)
                        flat_vals.append(fv)
                        if fv < 0.0:
                            is_invalid = True
                            break
                    if is_invalid:
                        break

                if not is_invalid and flat_vals:
                    max_allowed = 1.0 if max(flat_vals) <= 1.0 else 100.0
                    if any(v > max_allowed for v in flat_vals):
                        is_invalid = True

            if is_invalid:
                are_all_good = False
                print(f"Uwaga: adnotacja playfield jest niepoprawna na zdjeciu {item.image.name}")

        if are_all_good:
            print("Wszystkie oznaczenia są ok!")

        # warunek na trapez czyli gorne punkty maja byc bardziej wewnatrz niz dolne
    

    @staticmethod
    def display_on_image(annotation: ImagePlayfieldAnnotation, image: ArrayLike, frame_color: tuple[int] = (255, 0, 0)) -> ArrayLike:
        img = np.asarray(image).copy()
        nim = NumpyImage(img)

        pts = transform_annotation(nim, annotation.points)
        arr = np.array([[int(round(p.x)), int(round(p.y))] for p in pts], dtype=np.int32)

        if len(arr) >= 2:
            cv2.polylines(img, [arr.reshape((-1, 1, 2))], True, frame_color, 2)

        return img
    
