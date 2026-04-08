from pathlib import Path
import yaml
import cv2
import torch

from deep_peep_snooker.utils.common import ArrayLike, Rectangle
from deep_peep_snooker.ops import transform


_DEBUG_MODE = False


def set_debug_mode(mode: bool) -> None:
    global _DEBUG_MODE
    _DEBUG_MODE = mode


def get_debug_mode() -> bool:
    global _DEBUG_MODE
    return _DEBUG_MODE


def create_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def preprare_single_frame_for_inference(img: ArrayLike, device: torch.device) -> tuple[ArrayLike, torch.Tensor]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    prepared_img = transform(img).unsqueeze(0).to(device)
    return img, prepared_img


def put_text_on_img(img: ArrayLike, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    cv2.rectangle(
        img,
        (10, 10),
        (10 + text_width + 10, 10 + text_height + baseline + 10),
        (0, 0, 0),
        -1,
    )
    cv2.putText(img, text, (15, 10 + text_height), font, font_scale, color, thickness)


def load_yaml(file_path: Path | str) -> dict | None:
    with open(file_path, encoding='utf-8') as f:
        try:
            return yaml.safe_load(f)
        except Exception as e:
            print(f'An error occured, {e}')


def roi_rect_to_img_rect(rect: Rectangle, roi_origin: tuple[int, int]) -> Rectangle:
    x, y, w, h = rect
    roi_x, roi_y = roi_origin
    return x + roi_x, y + roi_y, w, h


def contour_to_bbox(bin_img: ArrayLike, ) -> Rectangle | None:
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return

    cnt = contours[0]
    return cv2.boundingRect(cnt)
