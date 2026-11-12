import numpy as np

from possession import features
from possession.load import Tracking


def test_since_until():
    ev = np.zeros(10, bool)
    ev[3] = ev[7] = True
    since, until = features.since_until(ev)
    assert since[5] * 25 == 2 and until[5] * 25 == 2
    assert since[3] == 0 and until[3] == 0
    assert since[0] == features.CAP_S and until[9] == features.CAP_S


def test_attack_sign_from_keepers():
    """H keeper stays at x=3 in half 1 and x=102 in half 2; A's keeper mirrors it."""
    n = 200
    px = np.full((n, 2), 50.0)
    px[:100, 0], px[100:, 0] = 3.0, 102.0
    px[:100, 1], px[100:, 1] = 102.0, 3.0
    tr = Tracking(ball=np.full((n, 2), np.nan), px=px, py=np.full((n, 2), 34.0),
                  ps=np.zeros((n, 2)), team=np.array(["H", "A"]))
    s = features.attack_sign(tr, [(1, 0, 99), (2, 100, 199)], (105.0, 68.0))
    assert s[50] == 1 and s[150] == -1


def test_team_matrix_is_mirror():
    """From A's side, own and opp columns are H's columns swapped."""
    import pandas as pd
    blocks = {"H": pd.DataFrame({"d1": [1.0], "n2": [2.0]}), "A": pd.DataFrame({"d1": [5.0], "n2": [0.0]}),
              "game": pd.DataFrame({"ball_known": [1.0], "ball_seen": [1.0], "gap_s": [0.0], "ball_speed": [3.0]})}
    h, a = features.team_matrix(blocks, "H"), features.team_matrix(blocks, "A")
    assert h["own_d1"][0] == a["opp_d1"][0] == 1.0
    assert h["diff_d1"][0] == -a["diff_d1"][0]
