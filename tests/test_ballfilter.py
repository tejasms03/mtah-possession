import numpy as np

from possession import ballfilter
from possession.config import FPS, load_params
from possession.load import Tracking

PARAMS = load_params()
P = PARAMS["ball"]


def pass_play(noise=0.0, seed=0):
    """Barça player at (30, 34) passes at 12 m/s to an Athletic player at (54, 34), who stops it
    dead. Frames 0-24 the ball sits at the passer's feet, 25-49 travels, 50-99 sits at the receiver."""
    n = 100
    x = np.empty(n)
    x[:25] = 30.0
    x[25:50] = 30.0 + 12.0 * (np.arange(25) + 1) / FPS
    x[50:] = x[49]
    ball = np.column_stack([x, np.full(n, 34.0)])
    ball += np.random.default_rng(seed).normal(0, noise, ball.shape)
    px = np.column_stack([np.full(n, 29.5), np.full(n, x[49] + 0.5)])
    py = np.full((n, 2), 34.0)
    tr = Tracking(ball=ball, px=px, py=py, ps=np.zeros((n, 2)), team=np.array(["H", "A"]))
    return tr


def test_kick_and_receive_are_touches_by_the_right_team():
    bt = ballfilter.run(pass_play(), PARAMS)
    frames = np.flatnonzero(bt.touch)
    assert len(frames) == 2
    assert abs(frames[0] - 25) <= 3 and bt.touch_team[frames[0]] == "H"
    assert abs(frames[1] - 50) <= 3 and bt.touch_team[frames[1]] == "A"


def test_jitter_alone_makes_no_touches():
    n = 200
    rng = np.random.default_rng(1)
    ball = np.column_stack([np.full(n, 50.0), np.full(n, 30.0)]) + rng.normal(0, 0.1, (n, 2))
    tr = Tracking(ball=ball, px=np.full((n, 1), 50.5), py=np.full((n, 1), 30.0),
                  ps=np.zeros((n, 1)), team=np.array(["H"]))
    assert ballfilter.run(tr, PARAMS).touch.sum() == 0


def test_rolling_ball_far_from_players_is_not_a_touch():
    tr = pass_play()
    tr.px[:] = 0.0                                   # everyone far away
    assert ballfilter.run(tr, PARAMS).touch.sum() == 0


def test_short_gap_bridged_long_gap_left():
    tr = pass_play()
    tr.ball[30:35] = np.nan                          # 5 frames
    tr.ball[60:90] = np.nan                          # 30 frames
    bt = ballfilter.run(tr, PARAMS)
    assert bt.bridged[30:35].all() and not np.isnan(bt.x[30:35]).any()
    assert np.isnan(bt.x[60:90]).all() and not bt.bridged[60:90].any()
    assert bt.gap_len[32] == 5 and bt.gap_len[70] == 30 and bt.gap_len[10] == 0
    assert np.allclose(bt.x[30:35], 30.0 + 12.0 * (np.arange(5, 10) + 1) / FPS, atol=0.05)


def test_runs():
    assert ballfilter.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
