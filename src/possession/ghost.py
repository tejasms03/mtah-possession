"""Idea D, "ghost ball" (off unless features.ghost_extra): where is the ball, judging only by the
players? A boosted-tree regressor learns ball x, y (as a share of pitch length / width) from player
positions and speeds on frames where the ball is seen in the outside matches. It uses ball positions
only, never possession labels, and never our match. The decoder's dead-ball model then gets notes
about that guessed point while the real ball is missing: how close it is to a line or a restart spot,
how many players are around it, whether players are running towards it, and how fast it moves.
Never compared with the real ball (DFL freezes the ball at stoppages, which would give the answer away).

Models: outputs/cache/ghost/ghost_all.pkl (all outside matches, used for our match) and
ghost_without_{id}.pkl (used when {id} is the held-out bench match). Built by train().
"""

import pickle

import numpy as np
import pandas as pd

from possession import config
from possession.config import FPS

DIR = config.REPO_ROOT / "outputs" / "cache" / "ghost"
_MODELS: dict = {}


def inputs(tr, pitch) -> np.ndarray:
    """Per-frame player summaries in pitch shares (x / L, y / W). Players only, no ball."""
    L, W = pitch
    hs = np.isin(tr.team, ["H", "A"])
    u, v = tr.px[:, hs] / L, tr.py[:, hs] / W
    team = tr.team[hs]
    s = np.hypot(tr.vx[:, hs], tr.vy[:, hs]) if tr.vx is not None else tr.ps[:, hs]
    cols = []
    with np.errstate(invalid="ignore"):
        cols += [np.nanmean(u, 1), np.nanmean(v, 1), np.nanstd(u, 1), np.nanstd(v, 1)]
        for t in ("H", "A"):
            c = team == t
            cols += [np.nanmean(u[:, c], 1), np.nanmean(v[:, c], 1), np.nanmin(u[:, c], 1), np.nanmax(u[:, c], 1)]
        # densest spot: the 4 players with most others within 10 m
        d = np.hypot((u[:, :, None] - u[:, None, :]) * L, (v[:, :, None] - v[:, None, :]) * W)
        crowd = (np.nan_to_num(d, nan=np.inf) <= 10).sum(2).astype(float)
        crowd[np.isnan(u)] = -1
        top = np.argsort(-crowd, 1)[:, :4]
        r = np.arange(len(u))[:, None]
        cols += [np.nanmean(u[r, top], 1), np.nanmean(v[r, top], 1), crowd.max(1)]
        # the 3 fastest players
        fast = np.argsort(-np.nan_to_num(s, nan=-1), 1)[:, :3]
        cols += [np.nanmean(u[r, fast], 1), np.nanmean(v[r, fast], 1), np.nanmean(s[r, fast], 1)]
    return np.column_stack(cols).astype(np.float32)


def train(preps_tracking: dict, pitches: dict, log=print) -> None:
    """preps_tracking: {match id: cleaned degraded Tracking}. Fits one all-match model and one
    leave-one-out model per match; every 5th seen-ball frame."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    data = {}
    for mid, tr in preps_tracking.items():
        L, W = pitches[mid]
        seen = np.flatnonzero(~np.isnan(tr.ball[:, 0]))[::5]
        data[mid] = (inputs(tr, pitches[mid])[seen], tr.ball[seen, 0] / L, tr.ball[seen, 1] / W)
    DIR.mkdir(parents=True, exist_ok=True)

    def fit(keys, name):
        X = np.concatenate([data[k][0] for k in keys])
        mx = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0)
        my = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0)
        mx.fit(X, np.concatenate([data[k][1] for k in keys]))
        my.fit(X, np.concatenate([data[k][2] for k in keys]))
        with open(DIR / f"{name}.pkl", "wb") as fh:
            pickle.dump((mx, my), fh)
        return mx, my

    fit(list(data), "ghost_all")
    for mid in data:
        mx, my = fit([k for k in data if k != mid], f"ghost_without_{mid}")
        X, gx, gy = data[mid]
        L, W = pitches[mid]
        err = np.hypot((mx.predict(X) - gx) * L, (my.predict(X) - gy) * W)
        log(f"ghost held out {mid}: median error {np.median(err):.1f} m, p90 {np.percentile(err, 90):.1f} m")


def _model(name: str):
    if name not in _MODELS:
        with open(DIR / f"{name}.pkl", "rb") as fh:
            _MODELS[name] = pickle.load(fh)
    return _MODELS[name]


def ghost_block(tr, bt, pitch, match_id: str | None = None) -> pd.DataFrame:
    """Notes about the guessed ball point, team-blind. match_id: the held-out bench match (uses the
    model that never saw it); None for our match (all-match model)."""
    L, W = pitch
    mx, my = _model(f"ghost_without_{match_id}" if match_id else "ghost_all")
    X = inputs(tr, pitch)
    gx = np.clip(mx.predict(X), 0, 1) * L
    gy = np.clip(my.predict(X), 0, 1) * W
    hs = np.isin(tr.team, ["H", "A"])
    px, py = tr.px[:, hs], tr.py[:, hs]
    dx, dy = px - gx[:, None], py - gy[:, None]
    dist = np.hypot(dx, dy)
    spots = np.array([[0, 0], [0, W], [L, 0], [L, W], [L / 2, W / 2], [5.5, W / 2], [L - 5.5, W / 2],
                      [11, W / 2], [L - 11, W / 2]])
    restart = np.min(np.hypot(gx[:, None] - spots[None, :, 0], gy[:, None] - spots[None, :, 1]), 1)
    restart = np.minimum(restart, np.minimum(gy, W - gy))
    k = 10                                                     # 0.4 s
    move = np.full(len(gx), np.nan)
    move[k:] = np.hypot(gx[k:] - gx[:-k], gy[k:] - gy[:-k]) / (k / FPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        near = dist <= 20
        if tr.vx is not None:
            radial = -(tr.vx[:, hs] * dx + tr.vy[:, hs] * dy) / np.maximum(dist, 0.5)   # + = running towards
            conv = np.nanmean(np.where(near, radial, np.nan), 1)
        else:
            conv = np.full(len(gx), np.nan)
    return pd.DataFrame({
        "gh_edge": np.minimum.reduce([gx, L - gx, gy, W - gy]),
        "gh_restart_m": restart,
        "gh_n5": (np.nan_to_num(dist, nan=np.inf) <= 5).sum(1).astype(float),
        "gh_conv": conv,
        "gh_move": move,
    })
