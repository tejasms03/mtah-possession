"""Network sequence model (off unless model.tcn.use): a small non-causal dilated temporal
convolution network (TCN) that reads the first stage's per-frame outputs (team, switch, dead
log-odds) and a few ball signals over about +-20 s, and gives per-frame H/A/D log-probabilities
plus a turnover head (none / H->A / A->H within +-5 frames).

Trained offline in PyTorch on the outside matches only (tools/train_tcn.py); the weights are
saved to models/tcn.npz and this file runs them with plain numpy in float64, so the pipeline
needs no torch and the output is reproducible. Team-blind: the network is run on the inputs as
they are and with H and A swapped, and the two answers are averaged.

Design: generic dilated residual TCN (input 1x1 conv, residual blocks
h = h + W2 relu(conv_d(relu(h))), kernel 3, dilations 1..256, 1x1 output heads). Numpy only.
"""

import numpy as np

N_IN = 12
PAD = 512          # zero context added outside each half (receptive field is +-511 frames)


def features(team, switch, dead, observed, speed, d_H, d_A, touch_team) -> np.ndarray:
    """(N, N_IN) float32 inputs for one stretch of frames (one half). Channel order matters:
    swap_ha() relies on it."""
    n = len(team)
    t = np.clip(np.nan_to_num(np.asarray(team, float)), -20, 20) / 4
    s = np.clip(np.nan_to_num(np.asarray(switch, float)), -20, 20) / 4
    d = np.clip(np.nan_to_num(np.asarray(dead, float)), -20, 20) / 4
    obs = np.asarray(observed, bool).astype(float)
    sp = np.asarray(speed, float)
    sp_miss = ~np.isfinite(sp)
    sp = np.log1p(np.clip(np.where(sp_miss, 0.0, sp), 0, 60)) / 3
    dh, da = np.asarray(d_H, float), np.asarray(d_A, float)
    d_miss = ~(np.isfinite(dh) & np.isfinite(da))
    dh = np.log1p(np.clip(np.where(np.isfinite(dh), dh, 40.0), 0, 40)) / 3
    da = np.log1p(np.clip(np.where(np.isfinite(da), da, 40.0), 0, 40)) / 3
    tt = np.asarray(touch_team).astype(str)
    cols = [t, s, d, obs, sp, sp_miss.astype(float), dh, da,
            (tt == "H").astype(float), (tt == "A").astype(float), d_miss.astype(float), np.ones(n)]
    return np.column_stack(cols).astype(np.float32)


SWAP = np.array([0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10, 11])
SIGN = np.array([-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32)


def swap_ha(X: np.ndarray) -> np.ndarray:
    """The same inputs with H and A exchanged."""
    return X[..., SWAP] * SIGN


def _log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _forward(W: dict, X: np.ndarray):
    """X (N, N_IN) -> (emission logits (N, 3) H/A/D, turnover logits (N, 3) none/HA/AH).
    Zero padding of PAD frames on each side, like training."""
    n = len(X)
    x = np.zeros((n + 2 * PAD, X.shape[1]))
    x[PAD : PAD + n] = X
    h = x @ W["in_w"].T + W["in_b"]
    for k, d in enumerate(W["dilations"]):
        d = int(d)
        a = np.maximum(h, 0.0)
        w = W[f"c{k}_w"]                       # (C, C, 3): out, in, tap
        z = a @ w[:, :, 1].T + W[f"c{k}_b"]
        z[d:] += a[:-d] @ w[:, :, 0].T         # tap 0 looks d frames back
        z[:-d] += a[d:] @ w[:, :, 2].T         # tap 2 looks d frames ahead
        h = h + np.maximum(z, 0.0) @ W[f"o{k}_w"].T + W[f"o{k}_b"]
    a = np.maximum(h[PAD : PAD + n], 0.0)
    return a @ W["em_w"].T + W["em_b"], a @ W["to_w"].T + W["to_b"]


def load(path) -> dict:
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) if k != "dilations" else z[k] for k in z.files}


def predict(W: dict, X: np.ndarray):
    """Team-blind outputs for one half: (emission log-probs (N, 3) H/A/D, turnover log-probs
    (N, 3) none/H->A/A->H)."""
    e1, t1 = _forward(W, X)
    e2, t2 = _forward(W, swap_ha(X))
    e = 0.5 * (_log_softmax(e1) + _log_softmax(e2)[:, [1, 0, 2]])
    t = 0.5 * (_log_softmax(t1) + _log_softmax(t2)[:, [0, 2, 1]])
    return _log_softmax(e), _log_softmax(t)


def predict_match(W: dict, team, switch, dead, bt, halves, n: int):
    """Whole match: (N, 3) emission and (N, 3) turnover log-probs; frames outside the halves
    get uniform values."""
    em = np.full((n, 3), -np.log(3.0))
    to = np.full((n, 3), -np.log(3.0))
    for _, a, b in halves:
        sl = slice(a, b + 1)
        X = features(team[sl], switch[sl], dead[sl], bt.observed[sl], bt.speed[sl], bt.d_H[sl], bt.d_A[sl],
                     bt.touch_team[sl])
        em[sl], to[sl] = predict(W, X)
    return em, to
