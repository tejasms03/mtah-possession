"""The council's chair: a second-level model that reads every expert's output around a frame and
decides H / A / D, in place of the hand-weighted sum in decode.emissions.

Experts (per frame): dead-ball log-odds (model A), H-vs-A log-odds (model B), change-detector
log-odds, touch team (+1 H, -1 A, 0 none), ball seen (0/1). The chair sees each at fixed offsets
(-5 s .. +5 s), as centred means over 1-30 s, and as past/future means over 2 s. It is trained on
expert outputs for matches those experts never saw (out-of-fold), with every row also added
team-swapped, so it is team-blind. Its log-probabilities become the decoder's emissions.
"""

import numpy as np

from possession import model

OFFSETS = (-125, -50, -25, -12, -5, 0, 5, 12, 25, 50, 125)    # frames
WINDOWS = (25, 75, 250, 750)                                  # centred means, frames
SIDE = 50                                                     # past / future means, frames
TEAM_SIGNED = (1, 3)                                          # columns that flip sign with H<->A


def signals(prep: dict, dead: np.ndarray, team: np.ndarray, switch: np.ndarray) -> np.ndarray:
    bt = prep["bt"]
    touch = np.where(bt.touch_team == "H", 1.0, np.where(bt.touch_team == "A", -1.0, 0.0))
    seen = (~np.isnan(bt.x)).astype(float)
    return np.column_stack([np.nan_to_num(dead), np.nan_to_num(team), np.nan_to_num(switch), touch, seen])


def _means(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    i = np.arange(n)
    a, b = np.clip(i - w, 0, n), np.clip(i + w + 1, 0, n)
    return (c[b] - c[a]) / (b - a)


def _side(x: np.ndarray, w: int, past: bool) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    i = np.arange(n)
    a, b = (np.clip(i - w, 0, n), i) if past else (i + 1, np.clip(i + w + 1, 0, n))
    b = np.maximum(b, a + 1)
    b = np.minimum(b, n)
    a = np.minimum(a, b - 1)
    return (c[b] - c[a]) / (b - a)


def features(sig: np.ndarray) -> np.ndarray:
    """(N, F) chair inputs for every frame."""
    n = len(sig)
    cols = []
    for k in range(sig.shape[1]):
        x = sig[:, k]
        for o in OFFSETS:
            cols.append(x[np.clip(np.arange(n) + o, 0, n - 1)])
        for w in WINDOWS:
            cols.append(_means(x, w))
        cols.append(_side(x, SIDE, True))
        cols.append(_side(x, SIDE, False))
    return np.column_stack(cols).astype(np.float32)


def _mirror(F: np.ndarray, n_sig: int) -> np.ndarray:
    per = len(OFFSETS) + len(WINDOWS) + 2
    G = F.copy()
    for k in TEAM_SIGNED:
        G[:, k * per : (k + 1) * per] *= -1
    return G


def fit(examples: list[tuple[np.ndarray, np.ndarray]], params: dict):
    """examples: (signals (N, 5), labels (N,) H/A/D/X) from out-of-fold expert outputs."""
    step = params["model"]["row_step"]
    X, y = [], []
    for sig, gt in examples:
        idx = np.flatnonzero(np.isin(gt, ["H", "A", "D"]))[3::step]
        F = features(sig)[idx]
        lab = np.select([gt[idx] == "H", gt[idx] == "A"], [0, 1], 2)
        X += [F, _mirror(F, sig.shape[1])]
        y += [lab, np.select([lab == 0, lab == 1], [1, 0], 2)]
    m = model.make_model(params)
    m.fit(np.concatenate(X), np.concatenate(y))
    return m


def log_proba(m, sig: np.ndarray, chunk: int = 50000) -> np.ndarray:
    """(N, 3) log-probabilities in decode.LABELS order (H, A, D), symmetrised over the team swap."""
    F = features(sig)
    out = []
    for a in range(0, len(F), chunk):
        f = F[a : a + chunk]
        p = m.predict_proba(f)
        q = m.predict_proba(_mirror(f, sig.shape[1]))[:, [1, 0, 2]]
        out.append(np.log(np.clip((p + q) / 2, 1e-6, 1)))
    return np.concatenate(out)
