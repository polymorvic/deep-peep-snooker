import cv2
import numpy as np

from .const import BallColor, BALLS
from .playfield_finder import Playfield


# top_left point, bottom right point

# top_left point H W


# podjescei 1
# sprawdzamy kolor po kolorze

# podejcie 2
# bierzemy wszystki i kazda po kolei patrzymy jaki jest kolor i próbujemy przypisac do zdefiniowanych klas kolorów (grayscale / kanał v w hsv) 

class BallDetector:
    """
    Detects balls in an image.
    """

    def __init__(self, img: np.ndarray, playfield: Playfield) -> None:
        self.img = img
        self.playfield = playfield

        self.img_hsv = self._preprocess_image()


    def _preprocess_image(self) -> np.ndarray:
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        points = self.playfield.points.to_numpy()

        mask = np.zeros(img_hsv.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)

        img_hsv[mask==0] = 0

        return img_hsv

    def get_balls(self, color: BallColor):
        lower_bound = BALLS[color]['lower_bound']
        upper_bound = BALLS[color]['upper_bound']

        


        # mask = (100 < self.img_hsv[..., 0]) & (self.img_hsv[..., 0] < 140)

        mask = cv2.inRange(self.img_hsv, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # dla kazdego poza czerownym
        if color == BallColor.RED:
            pass
        else:
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(self.img, [cnt], -1, (0, 255, 255), 1)
            return self.img



