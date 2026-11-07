import numpy as np
import pytest

from possession.clean import clean_ball, clean_players, player_velocities
from possession.config import FPS, load_params
from possession.load import read_atd_raw, to_tracking

P = load_params()["clean"]


def ball_track(n=100):
    """Ball rolling along x at 10 m/s from the centre spot."""
    t = np.arange(n) / FPS
    return np.column_stack([52.0 + 10 * t, np.full(n, 34.0)])


def test_detour_removed_real_pass_kept():
    ball = ball_track()
    ball[50] = [10.0, 10.0]                  # one-frame jump to the far corner and back
    ball[70:] += [0.0, 0.0]                  # untouched
    out, rep = clean_ball(ball, P)
    assert np.isnan(out[50]).all()
    assert rep["ball_removed_detours"] == 1
    assert (~np.isnan(out[:, 0])).sum() == 99


def test_long_gap_reappearance_kept():
    ball = ball_track()
    ball[20:80] = np.nan                     # 2.4 s gap, ball reappears 40 m away: possible
    ball[80:] = [92.0, 60.0]
    out, rep = clean_ball(ball, P)
    assert rep["ball_removed_detours"] == 0
    assert not np.isnan(out[80:, 0]).any()


def test_ball_far_off_pitch_removed():
    ball = ball_track()
    ball[10] = [52.0, -20.0]
    out, rep = clean_ball(ball, P)
    assert np.isnan(out[10]).all()
    assert rep["ball_removed_offpitch"] == 1


def test_duplicate_detection_removed():
    n = 10
    px = np.column_stack([np.full(n, 30.0), np.full(n, 30.1), np.full(n, 30.1)])
    py = np.full((n, 3), 20.0)
    team = np.array(["H", "H", "A"])
    px2, _, rep = clean_players(px, py, team, P)
    assert np.isnan(px2[:, 1]).all()          # second Barça detection dropped
    assert not np.isnan(px2[:, 2]).any()      # opponent at the same spot kept
    assert rep["player_removed_duplicates"] == n


def test_velocity_survives_slot_swap():
    """Two teammates 10 m apart run at 5 m/s; their slots swap at frame 50."""
    n = 100
    t = np.arange(n) / FPS
    a = 20 + 5 * t
    b = 40 + 5 * t
    px = np.column_stack([a, b])
    px[50:] = px[50:, ::-1]
    py = np.full((n, 2), 30.0)
    team = np.array(["H", "H"])
    vx, vy = player_velocities(px, py, team, P)
    assert np.allclose(vx, 5.0)
    assert np.allclose(vy, 0.0)


def test_velocity_ignores_opponent_slot():
    """An opponent appears where our player's slot would be: links never cross teams."""
    n = 20
    px = np.column_stack([np.full(n, 30.0), np.full(n, np.nan)])
    px[10:, 0] = np.nan
    px[10:, 1] = 30.3                         # if linked across teams, vx[9, 0] would be ~1.25
    py = np.full((n, 2), 30.0)
    vx, _ = player_velocities(px, py, np.array(["H", "A"]), P)
    assert vx[9, 0] == 0.0                    # one-sided from its own past only
    assert vx[10, 1] == 0.0


def test_raw_parser_and_frame_alignment(tmp_path):
    lines = [
        "1:0.5,0.5,1.0;NaN,NaN,NaN:0.5,0.5,NaN",
        "2:0.6,0.5,NaN;0.1,0.2,3.0:NaN,NaN,NaN",
    ]
    f = tmp_path / "atd.txt"
    f.write_text("\n".join(lines) + "\n")
    frame_id, players, ball = read_atd_raw(f)
    assert frame_id.tolist() == [1, 2]
    assert players.shape == (2, 2, 3)
    assert np.isnan(players[0, 1]).all()
    tr = to_tracking(frame_id, players, ball, np.array(["H", "A"]))
    assert tr.ball[0] == pytest.approx([105.3 / 2, 34.0])      # frame_id 1 -> video frame 0
    assert np.isnan(tr.ball[1]).all()
    assert tr.px[1, 1] == pytest.approx(10.53)
