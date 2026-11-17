"""Sequence model, tree version (off unless model.sequence.use): a second stage that reads the first
stage's per-frame outputs (team, switch and dead log-odds) at many time offsets, -4 s to +4 s,
plus window summaries, so it can learn the shape of a turnover (pass, opponent arrives, play goes
on) that per-frame models cannot see. Two heads: team (H vs A on live frames) and switch (a direct
H<->A change within +-0.5 s). Team-blind: every row is also shown with H and A swapped.

Trained on outside matches only. Inputs come from first-stage models that never saw the match
they are predicting (out-of-fold), so the second stage learns from honest first-stage errors.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

OFFSETS = (-100, -75, -50, -37, -25, -12, -6, 0, 6, 12, 25, 37, 50, 75, 100)    # frames (25 fps)
DEAD_OFFSETS = (-50, -25, 0, 25, 50)
WINDOWS = (12, 25, 50, 100)                                                      # half-widths


def _shift(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _win(a, h, fn):
    n = len(a)
    c = np.r_[0.0, np.cumsum(np.nan_to_num(a))]
    i = np.arange(n)
    lo, hi = np.maximum(0, i - h), np.minimum(n, i + h + 1)
    if fn == "mean":
        return (c[hi] - c[lo]) / (hi - lo)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(np.nan_to_num(a), h, mode="edge")
    v = sliding_window_view(pad, 2 * h + 1)
    return v.max(axis=1) if fn == "max" else v.min(axis=1)


def matrix(team, switch, dead, observed, sign=1.0) -> np.ndarray:
    """Second-stage inputs. sign=-1 swaps H and A (team log-odds change sign; switch and dead
    do not depend on which team is which)."""
    t = sign * np.clip(np.nan_to_num(team), -20, 20)
    s = np.clip(np.nan_to_num(switch), -20, 20)
    d = np.clip(np.nan_to_num(dead), -20, 20)
    cols = [_shift(t, k) for k in OFFSETS] + [_shift(s, k) for k in OFFSETS] + [_shift(d, k) for k in DEAD_OFFSETS]
    flips = (np.sign(t[1:]) != np.sign(t[:-1])).astype(float)
    flips = np.r_[0.0, flips]
    for h in WINDOWS:
        cols += [_win(t, h, "mean"), _win(t, h, "max"), _win(t, h, "min"), _win(s, h, "max"),
                 _win(flips, h, "mean"), _win(observed.astype(float), h, "mean")]
    cols.append(observed.astype(float))
    return np.column_stack(cols).astype(np.float32)


def _model(seed=0):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)


def fit(items, step: int = 5, half_window: int = 12, seed: int = 0):
    """items: list of (expert dict with dead/team/switch, observed (N,), truth labels (N,)).
    Returns (team_model, switch_model)."""
    from possession.model import switch_target

    Xt, yt, Xs, ys = [], [], [], []
    for e, obs, gt in items:
        live = np.flatnonzero(np.isin(gt, ["H", "A"]))[::step]
        sw = switch_target(gt, half_window)
        for sign in (1.0, -1.0):
            X = matrix(e["team"], e["switch"], e["dead"], obs, sign)[live]
            Xt.append(X)
            yt.append((gt[live] == "H") if sign > 0 else (gt[live] == "A"))
            Xs.append(X)
            ys.append(sw[live])
    mt, ms = _model(seed), _model(seed)
    mt.fit(np.concatenate(Xt), np.concatenate(yt))
    ms.fit(np.concatenate(Xs), np.concatenate(ys))
    return mt, ms


def predict(models, e, obs, chunk: int = 50000):
    """(team log-odds for H, switch log-odds); team is averaged over both orientations so the
    result is exactly team-blind."""
    mt, ms = models
    X1, X2 = matrix(e["team"], e["switch"], e["dead"], obs, 1.0), matrix(e["team"], e["switch"], e["dead"], obs, -1.0)
    n = len(X1)
    f = lambda m, X: np.concatenate([m.decision_function(X[a : a + chunk]) for a in range(0, n, chunk)])  # noqa: E731
    team = 0.5 * (f(mt, X1) - f(mt, X2))
    switch = 0.5 * (f(ms, X1) + f(ms, X2))
    return team, switch
