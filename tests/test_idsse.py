"""Degrade and bench on a synthetic match: no downloads, no match data."""

import numpy as np

from possession.config import FPS, load_params
from possession.external import bench
from possession.external.idsse import IdsseMatch, degrade, degrade_report
from possession.load import Tracking

P = load_params()["idsse_degrade"]


def synthetic_match(minutes=20, seed=1) -> IdsseMatch:
    """22 players jogging around, a ball passed between them, alternating 20 s possessions."""
    rng = np.random.default_rng(seed)
    half = minutes * 60 * FPS // 2
    gap = 100
    n = 2 * half + gap
    t = np.arange(n) / FPS
    base = rng.uniform([5, 5], [100, 63], (22, 2))
    px = base[:, 0] + 3 * np.sin(t[:, None] / 7 + np.arange(22))
    py = base[:, 1] + 3 * np.cos(t[:, None] / 9 + np.arange(22))
    holder = (t // 2).astype(int) % 22
    ball = np.column_stack([px[np.arange(n), holder], py[np.arange(n), holder]])
    gt = np.where((t // 20).astype(int) % 2 == 0, "H", "A")
    gt[(t % 60) > 50] = "D"
    gt[half:half + gap] = "X"
    px[half:half + gap] = py[half:half + gap] = ball[half:half + gap] = np.nan
    tr = Tracking(ball=ball, px=px, py=py, ps=np.zeros_like(px),
                  team=np.array(["H"] * 11 + ["A"] * 11))
    return IdsseMatch("SYNTH", tr, np.zeros(n), gt, [(1, 0, half - 1), (2, half + gap, n - 1)],
                      (105.0, 68.0))


def test_degrade_is_deterministic():
    m = synthetic_match()
    a, b = degrade(m, P, 0), degrade(m, P, 0)
    np.testing.assert_array_equal(a.tracking.px, b.tracking.px)
    np.testing.assert_array_equal(a.tracking.ball, b.tracking.ball)
    assert not np.array_equal(np.isnan(degrade(m, P, 1).tracking.ball[:, 0]),
                              np.isnan(a.tracking.ball[:, 0]))


def test_degrade_hits_ball_target_and_keeps_labels():
    m = synthetic_match()
    d = degrade(m, P, 0)
    r = degrade_report(d)
    assert abs(r["ball_missing"] - P["ball_missing_share"]) < 0.05
    np.testing.assert_array_equal(d.gt, m.gt)
    assert d.halves == m.halves


def test_degrade_removes_identity_but_keeps_team_positions():
    m = synthetic_match()
    d = degrade(m, P, 0)
    t = d.tracking
    assert set(np.unique(t.team)) == {"H", "A"}
    assert t.n_slots > 22                       # fragments land on reused, extra slots
    vis = ~np.isnan(t.px)
    assert 18 < vis.sum(1)[d.gt != "X"].mean() < 22
    # every visible degraded home point is within noise + duplicate offset of a real home player
    f = 1000
    home = vis[f] & (t.team == "H")
    real = np.column_stack([m.tracking.px[f, :11], m.tracking.py[f, :11]])
    pts = np.column_stack([t.px[f, home], t.py[f, home]])
    dist = np.hypot(*(pts[:, None, :] - real[None]).transpose(2, 0, 1)).min(1)
    assert (dist < 4.0).all()


def test_bench_scores_perfect_labels_as_one():
    m = synthetic_match()
    r = bench.score(m, m.gt.copy())
    assert r["spellF1"] == 1.0 and r["macroF1"] == 1.0
    assert r["poss_pred"] == r["poss_gt"]


def test_baseline_runs_on_degraded_match():
    d = degrade(synthetic_match(), P, 0)
    labels = bench.upstream_baseline(d)
    assert labels.shape == d.gt.shape
    assert set(np.unique(labels)) <= {"H", "A", "D"}
