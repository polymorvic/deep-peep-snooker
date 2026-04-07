import numpy as np

from deep_peep_snooker.schemas.ball import BallHSVRange, BallsConfig, BallColor, BallRanges


GREEN_LOWER_BOUND = np.array([25, 40, 40])


GREEN_UPPER_BOUND = np.array([95, 255, 255])


BALLS = BallsConfig(
    radius_mm=52.5,
    ranges=BallRanges(
        blue=BallHSVRange(
            lower_bound=np.array([100, 100, 100]),
            upper_bound=np.array([140, 255, 255]),
        ),
        yellow=BallHSVRange(
            lower_bound=np.array([20, 100, 100]),
            upper_bound=np.array([30, 255, 255]),
        ),
        green=BallHSVRange(
            lower_bound=np.array([30, 100, 100]),
            upper_bound=np.array([70, 255, 255]),
        ),
        brown=BallHSVRange(
            lower_bound=np.array([10, 100, 100]),
            upper_bound=np.array([20, 255, 255]),
        ),
        black=BallHSVRange(
            lower_bound=np.array([0, 100, 100]),
            upper_bound=np.array([10, 255, 255]),
        ),
        cue=BallHSVRange(
            lower_bound=np.array([0, 100, 100]),
            upper_bound=np.array([10, 255, 255]),
        ),
        red=BallHSVRange(
            lower_bound=np.array([0, 100, 100]),
            upper_bound=np.array([10, 255, 255]),
        ),
        pink=BallHSVRange(
            lower_bound=np.array([150, 100, 100]),
            upper_bound=np.array([170, 255, 255]),
        ),
    ),
)

