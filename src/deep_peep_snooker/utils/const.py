import numpy as np

from deep_peep_snooker.schemas.ball import BallHSVRange, BallsConfig, BallColor


GREEN_LOWER_BOUND = np.array([25, 40, 40])


GREEN_UPPER_BOUND = np.array([95, 255, 255])


BALLS = BallsConfig(
    radius_mm=52.5,
    ranges={
        BallColor.BLUE: BallHSVRange(
            lower_bound=np.array([100, 100, 100]),
            upper_bound=np.array([140, 255, 255]),
        ),
        BallColor.YELLOW: BallHSVRange(
            lower_bound=np.array([20, 100, 100]),
            upper_bound=np.array([30, 255, 255]),
        ),
        BallColor.GREEN: BallHSVRange(
            lower_bound=np.array([30, 100, 100]),
            upper_bound=np.array([70, 255, 255]),
        ),
        BallColor.BROWN: BallHSVRange(
            lower_bound=np.array([10, 100, 100]),
            upper_bound=np.array([20, 255, 255]),
        ),
        BallColor.BLACK: BallHSVRange(
            lower_bound=np.array([0, 100, 100]),
            upper_bound=np.array([10, 255, 255]),
        ),
        BallColor.CUE: BallHSVRange(
            lower_bound=np.array([0, 100, 100]),
            upper_bound=np.array([10, 255, 255]),
        ),
        BallColor.RED: BallHSVRange(
            lower_bound=np.array([0, 100, 100]),
            upper_bound=np.array([10, 255, 255]),
        ),
        BallColor.PINK: BallHSVRange(
            lower_bound=np.array([150, 100, 100]),
            upper_bound=np.array([170, 255, 255]),
        ),
    },
)

