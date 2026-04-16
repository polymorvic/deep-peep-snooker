import cv2
import numpy as np

from deep_peep_snooker.utils.common import ArrayLike, NumpyImage
from deep_peep_snooker.schemas.ball import BallColor
from deep_peep_snooker.schemas.annotation import BBox
from deep_peep_snooker.schemas.playfield import Playfield
from deep_peep_snooker.utils.const import BALLS
from deep_peep_snooker.utils.plotting import display_img
from deep_peep_snooker.utils.helpers import roi_rect_to_img_rect, contour_to_bbox, get_debug_mode
from deep_peep_snooker.utils.func import crop_image_by_points



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
    

    def find_blue(self, 
                  roi_width_delta: float = 0.25, 
                  roi_height_low_delta: float = 0.5,
                  roi_height_up_delta: float = 3.0):
        r = BALLS.range_for(BallColor.BLUE)
        bin_img = cv2.inRange(self.cropped_img_hsv, r.lower_bound, r.upper_bound)

        try:
            x, y, w, h = contour_to_bbox(bin_img)
        except TypeError:
            return
        
        h_up = int(roi_height_up_delta * h)
        h_low = int(h * roi_height_low_delta)
        w_delta = int(w * roi_width_delta) 

        roi_hsv, roi_x1, roi_y1 = crop_image_by_points(self.cropped_img_hsv, 
                                                       np.array([[x - w_delta, y - h_low], [x + w + w_delta, y + h_up]]))

        roi_h, _, _ = cv2.split(roi_hsv)
        med = cv2.medianBlur(roi_h, 7)
        _, th = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if get_debug_mode():
            display_img(roi_h)
            display_img(med)
            display_img(th)

        try:
            rect_roi = contour_to_bbox(th)
        except TypeError:
            return

        rect_img = roi_rect_to_img_rect(rect_roi, (roi_x1, roi_y1))
        x, y, w, h = roi_rect_to_img_rect(rect_img, self.crop_origin_xy)

        return BBox(x=x, y=y, width=w, height=h)















