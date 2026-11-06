"""Stage 3: IMM Kalman filter for the ball.

Three motion models run in parallel: rolling (friction), airborne (steady 2D speed), carried
(follows the nearest player). Outputs per frame: smoothed x, y, vx, vy, model probabilities,
touch impulses, and predicted position + uncertainty through gaps.
Owner: person 1.
"""


def run_imm(ball_xy, players, params):
    raise NotImplementedError

