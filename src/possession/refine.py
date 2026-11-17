"""Boundary refiner (off unless decode.refine.use): after decoding, look again at every possession
boundary and move it to the frame where the ball most likely changed hands.

For each predicted boundary t (a turnover H<->A, a restart D->team, or a stoppage team->D), every
frame c within +-W frames (not crossing the neighbouring boundaries) is a candidate. A tree model
scores how likely c is the true boundary, from: the models' team / switch / dead outputs around c,
the ball's speed and direction change, each team's nearest-player distance and detected touches.
Everything is oriented to the team that gains (or keeps) the ball, so the model is team-blind.
The boundary moves to the best candidate if its score clears a threshold.

Trained on outside matches only, from honest (held-out) model outputs.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

H, A, D = "H", "A", "D"
KINDS = {"turnover": 0, "restart": 1, "stoppage": 2}


def _bounds(lab, halves):
    """[(t, kind, team, lo, hi)]: label changes involving a team, with the candidate range lo..hi
    (exclusive of the neighbouring boundaries and halves)."""
    out = []
    for _, a, b in halves:          # (period, first frame, last frame)
        b = min(int(b) + 1, len(lab))
        seg = lab[a:b]
        ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1 + a
        edges = np.r_[a, ch, b]
        for k, t in enumerate(ch):
            prev, new = lab[t - 1], lab[t]
            if prev in (H, A) and new in (H, A):
                kind, team = "turnover", new
            elif prev == D and new in (H, A):
                kind, team = "restart", new
            elif prev in (H, A) and new == D:
                kind, team = "stoppage", prev
            else:
                continue
            # at most halfway to each neighbouring boundary, so two moved boundaries never cross
            out.append((int(t), kind, team, (int(edges[k]) + int(t) + 1) // 2, (int(t) + int(edges[k + 2])) // 2))
    return out


def _cum(a):
    return np.r_[0.0, np.cumsum(np.nan_to_num(a))]


def _wmean(c, lo, hi):
    n = len(c) - 1
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def frame_table(bt, e):
    """Per-frame arrays shared by all candidates of a match."""
    sp = np.nan_to_num(bt.speed, nan=0.0)
    vx, vy = np.nan_to_num(bt.vx), np.nan_to_num(bt.vy)
    n = len(sp)
    sh = lambda a, k: a[np.clip(np.arange(n) + k, 0, n - 1)]  # noqa: E731
    dot = sh(vx, -3) * sh(vx, 3) + sh(vy, -3) * sh(vy, 3)
    nrm = np.hypot(sh(vx, -3), sh(vy, -3)) * np.hypot(sh(vx, 3), sh(vy, 3))
    turn = np.where(nrm > 1e-6, 1 - dot / np.maximum(nrm, 1e-6), 0.0)
    dH, dA = np.clip(np.nan_to_num(bt.d_H, posinf=50), 0, 50), np.clip(np.nan_to_num(bt.d_A, posinf=50), 0, 50)
    team = np.clip(np.nan_to_num(e["team"]), -20, 20)
    return {
        "n": n, "speed": sp, "dspeed": sh(sp, 3) - sh(sp, -3), "turn": turn,
        "obs": bt.observed.astype(float), "gap": np.minimum(bt.gap_len, 500).astype(float),
        "dH": dH, "dA": dA, "tH": (bt.touch_team == H).astype(float), "tA": (bt.touch_team == A).astype(float),
        "team": team, "switch": np.clip(np.nan_to_num(e["switch"]), -20, 20),
        "dead": np.clip(np.nan_to_num(e["dead"]), -20, 20),
        "c_team": _cum(team), "c_dead": _cum(np.clip(np.nan_to_num(e["dead"]), -20, 20)),
        "c_tH": _cum((bt.touch_team == H).astype(float)), "c_tA": _cum((bt.touch_team == A).astype(float)),
        "c_dH": _cum(dH), "c_dA": _cum(dA),
    }


def candidates(lab, halves, F, W):
    """(rows (M, k) float32, boundary index per row, candidate frame per row, boundaries)."""
    bs = _bounds(lab, halves)
    X, bi, cf = [], [], []
    for i, (t, kind, team, lo, hi) in enumerate(bs):
        c = np.arange(max(lo, t - W), min(hi, t + W) + 1)
        if not len(c):
            continue
        s = 1.0 if team == H else -1.0
        dT, dO = (F["dH"], F["dA"]) if team == H else (F["dA"], F["dH"])
        cT, cO = (F["c_tH"], F["c_tA"]) if team == H else (F["c_tA"], F["c_tH"])
        cdT, cdO = (F["c_dH"], F["c_dA"]) if team == H else (F["c_dA"], F["c_dH"])
        cols = [
            c - t, np.full(len(c), KINDS[kind], float),
            s * F["team"][c],
            s * (_wmean(F["c_team"], c, c + 12) - _wmean(F["c_team"], c - 12, c)),
            s * (_wmean(F["c_team"], c, c + 25) - _wmean(F["c_team"], c - 25, c)),
            s * _wmean(F["c_team"], c, c + 25), s * _wmean(F["c_team"], c - 25, c),
            F["switch"][c], F["dead"][c],
            _wmean(F["c_dead"], c, c + 12) - _wmean(F["c_dead"], c - 12, c),
            F["obs"][c], F["gap"][c], F["speed"][c], F["dspeed"][c], F["turn"][c],
            dT[c], dO[c], dT[c] - dO[c],
            _wmean(cdT, c, c + 12) - _wmean(cdT, c - 12, c), _wmean(cdO, c, c + 12) - _wmean(cdO, c - 12, c),
            (cT[np.minimum(c + 4, F["n"])] - cT[np.maximum(c - 3, 0)]),
            (cO[np.minimum(c + 4, F["n"])] - cO[np.maximum(c - 3, 0)]),
            (cT[np.minimum(c + 25, F["n"])] - cT[c]) - (cT[c] - cT[np.maximum(c - 25, 0)]),
            (cO[np.minimum(c + 25, F["n"])] - cO[c]) - (cO[c] - cO[np.maximum(c - 25, 0)]),
        ]
        X.append(np.column_stack(cols))
        bi.append(np.full(len(c), i))
        cf.append(c)
    if not X:
        return np.zeros((0, 24), np.float32), np.zeros(0, int), np.zeros(0, int), bs
    return np.concatenate(X).astype(np.float32), np.concatenate(bi), np.concatenate(cf), bs


def targets(gt, bs, bi, cf, tol=2):
    """1 where the candidate is within tol frames of a true boundary of the same shape near it."""
    y = np.zeros(len(cf), bool)
    n = len(gt)
    prev = np.concatenate([gt[:1], gt[:-1]])
    for i, (t, kind, team, lo, hi) in enumerate(bs):
        if kind == "stoppage":
            true = (prev == team) & (gt != team)
        else:
            true = (prev != team) & (gt == team)
        m = bi == i
        c = cf[m]
        if not len(c):
            continue
        a, b = max(0, c.min() - tol), min(n, c.max() + tol + 1)
        tf = np.flatnonzero(true[a:b]) + a
        if len(tf):
            y[np.flatnonzero(m)] = (np.abs(c[:, None] - tf[None, :]) <= tol).any(axis=1)
    return y


def fit(rows, seed=0):
    X = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                       l2_regularization=1.0, random_state=seed)
    m.fit(X, y)
    return m


def apply(lab, m, X, bi, cf, bs, thr=0.3, kinds=("turnover", "restart", "stoppage"), margin=0.0):
    """Move each boundary to its best-scoring candidate when that score clears thr and beats the
    current position's score by margin. Boundaries are moved one at a time inside their own
    range, so they never cross."""
    out = lab.copy()
    if not len(cf):
        return out
    p = m.predict_proba(X)[:, 1]
    for i, (t, kind, team, lo, hi) in enumerate(bs):
        if kind not in kinds:
            continue
        idx = np.flatnonzero(bi == i)
        if not len(idx):
            continue
        j = idx[np.argmax(p[idx])]
        c, pc = int(cf[j]), p[j]
        here = idx[cf[idx] == t]
        p0 = p[here[0]] if len(here) else 0.0
        if c == t or pc < thr or pc - p0 < margin:
            continue
        old, new = lab[t - 1], lab[t]
        if c > t:
            out[t:c] = old
        else:
            out[c:t] = new
    return out
