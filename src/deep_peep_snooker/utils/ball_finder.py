import cv2
import numpy as np

from deep_peep_snooker.utils.common import ArrayLike, NumpyImage
from deep_peep_snooker.schemas.ball import BallColor
from deep_peep_snooker.schemas.playfield import Playfield
from deep_peep_snooker.utils.const import BALLS
from deep_peep_snooker.utils.plotting import display_img



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
    

    @staticmethod
    def roi_rect_to_img_rect(rect, roi_origin):
        x, y, w, h = rect
        roi_x, roi_y = roi_origin
        return x + roi_x, y + roi_y, w, h
    

    def find_blue(self, 
                  roi_width_delta: float = 0.25, 
                  roi_height_low_delta: float = 0.25,
                  roi_height_up_delta: int | float = 2):
        r = BALLS.range_for(BallColor.BLUE)
        bin_img = cv2.inRange(self.cropped_img_hsv, r.lower_bound, r.upper_bound)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return

        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)

        h_up = roi_height_up_delta * h
        h_low = int(h * roi_height_low_delta)

        w_delta = int(w * roi_width_delta)  

        img_h, img_w = self.cropped_img_hsv.shape[:2]

        roi_x1 = max(0, x - w_delta)
        roi_y1 = max(0, y - h_low)

        roi_x2 = min(img_w, x + w + w_delta)
        roi_y2 = min(img_h, y + h_up)

        roi_hsv = self.cropped_img_hsv[roi_y1:roi_y2, roi_x1:roi_x2]

        roi_h, _, _ = cv2.split(roi_hsv)

        med = cv2.medianBlur(roi_h, 7)
        _, th = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        display_img(roi_h)
        display_img(th)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return

        cnt = contours[0]
        rect_roi = cv2.boundingRect(cnt)
        rect_img = self.roi_rect_to_img_rect(rect_roi, (roi_x1, roi_y1))

        crop_x, crop_y = self.crop_origin_xy

        x, y, w, h = rect_img
        x += crop_x
        y += crop_y

        return x, y, w, h















