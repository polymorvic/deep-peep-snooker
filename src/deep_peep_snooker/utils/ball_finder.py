import cv2
import numpy as np

from deep_peep_snooker.utils.common import ArrayLike, NumpyImage
from deep_peep_snooker.schemas.ball import BallColor
from deep_peep_snooker.schemas.playfield import Playfield
from deep_peep_snooker.utils.const import BALLS



class BallFinder:

    def __init__(self, img: ArrayLike, playfield: Playfield):
        self.img = NumpyImage(img)
        self.playfield = playfield
        self.crop_origin_xy = self._compute_crop_origin_xy()
        self.cropped_img_hsv = self._crop_and_convert()


    def _compute_crop_origin_xy(self) -> tuple[int, int]:
        x0 = int(self.playfield.points.bottom_left.x)
        y0 = int(self.playfield.points.top_left.y)
        return x0, y0


    def _crop_and_convert(self):
        mask = np.zeros((self.img.height, self.img.width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.playfield.points.to_numpy()], 255)
        mask_img = cv2.bitwise_and(self.img, self.img, mask=mask)

        cropped_img = self._crop_img_by_corners(mask_img)
        return cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    

    def _crop_img_by_corners(self, mask_img: np.ndarray) -> ArrayLike:
        return mask_img[
            self.playfield.points.top_left.y: self.playfield.points.bottom_left.y + 1,
            self.playfield.points.bottom_left.x: self.playfield.points.bottom_right.x + 1
            ]
        

    @property
    def to_rgb(self) -> ArrayLike:
        return cv2.cvtColor(self.cropped_img_hsv, cv2.COLOR_HSV2RGB)
    

    def find_blue(self):

        r = BALLS.range_for(BallColor.BLUE)
        bin_img = cv2.inRange(self.cropped_img_hsv, r.lower_bound, r.upper_bound)

        cnt, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if cnt is None:
            return

        cnt = cnt[0]
        x, y, w, h = cv2.boundingRect(cnt)

        h_up = 2*h
        h_low = int(h * 0.25)

        roi_hsv = self.cropped_img_hsv[
            y - h_low: y + h_up + 1,
            x: x+ w + 1
        ]

        med = cv2.medianBlur(roi_hsv[..., 0], 7)
        _, th = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        cnt, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if cnt is None:
            return

        cnt = cnt[0]
        x, y, w, h = cv2.boundingRect(cnt)















