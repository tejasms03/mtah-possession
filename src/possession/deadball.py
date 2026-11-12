"""Stage 4a: dead-ball evidence and restarts.

Per-frame signals that the ball is out of play:
- players slow down together (mean speed over the outfield, smoothed over ~2 s)
- the ball is seen beyond a line
- the ball sits still, or has been missing for a long time
- a Smart Tagging set piece starts (our match only; IDSSE has no tags)

`rule_dead_logodds` turns them into one number per frame for the rule-only fallback. The trained
model (model A) uses `frame_features` instead. `set_piece_hints` gives the decoder the restarts
and, where the tag names one, the team that takes them.
Every threshold is in params.yaml under `dead_ball`. Owner: person 2.
"""

import numpy as np
import pandas as pd

from possession.ballfilter import BallTrack, runs
from possession.config import FPS, PITCH_LENGTH_M, PITCH_WIDTH_M
from possession.load import Tracking


def centred_mean(a: np.ndarray, frames: int) -> np.ndarray:
    """NaN-aware moving average over a centred window of `frames`."""
    ok = ~np.isnan(a)
    k = np.ones(frames)
    num = np.convolve(np.where(ok, a, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def team_speed(tr: Tracking) -> np.ndarray:
    """Mean speed of tracked H/A players per frame: our stage-1 velocity, else the ATD speed.
    Ours first, so BAR-ATH and IDSSE are measured the same way."""
    s = tr.ps
    if tr.vx is not None:
        s = np.where(np.isnan(tr.vx), s, np.hypot(tr.vx, tr.vy))
    s = np.where(np.isin(tr.team, ["H", "A"])[None, :], s, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(s, axis=1) if s.size else np.full(len(s), np.nan)


def out_distance(x: np.ndarray, y: np.ndarray, pitch=(PITCH_LENGTH_M, PITCH_WIDTH_M)) -> np.ndarray:
    """Metres beyond the nearest line (negative = inside the pitch), NaN where the ball is unknown."""
    L, W = pitch
    return np.maximum.reduce([-x, x - L, -y, y - W])


def still_seconds(bt: BallTrack, p: dict) -> np.ndarray:
    """How long the ball has been seen and nearly still, up to this frame."""
    still = bt.observed & (bt.speed < p["still_speed_mps"])
    out = np.zeros(len(still))
    for s, e in runs(still):
        out[s:e] = np.arange(1, e - s + 1) / FPS
    return out


def frame_features(tr: Tracking, bt: BallTrack, params: dict, pitch=None) -> pd.DataFrame:
    p = params["dead_ball"]
    spd = team_speed(tr)
    out = out_distance(bt.x, bt.y, pitch or (PITCH_LENGTH_M, PITCH_WIDTH_M))
    return pd.DataFrame({
        "spd": spd,
        "spd_2s": centred_mean(spd, 2 * FPS),
        "spd_6s": centred_mean(spd, 6 * FPS),
        "spd_drop": centred_mean(spd, 2 * FPS) - centred_mean(spd, 10 * FPS),
        "ball_out_m": out,
        "ball_out": (out > p["out_margin_m"]).astype(np.int8),
        "ball_still_s": still_seconds(bt, p),
        "gap_s": bt.gap_len / FPS,
        "ball_speed": bt.speed,
    })


def rule_dead_logodds(f: pd.DataFrame, params: dict) -> np.ndarray:
    """Hand-set log-odds that the ball is dead; positive = dead. For the rule-only fallback."""
    p = params["dead_ball"]
    lo = p["w_speed"] * (p["speed_ref_mps"] - f["spd_2s"].fillna(p["speed_ref_mps"]).to_numpy())
    lo = lo + p["w_out"] * f["ball_out"].to_numpy()
    lo = lo + p["w_still"] * np.minimum(f["ball_still_s"].to_numpy(), 3.0) / 3.0
    lo = lo - p["w_fast_ball"] * (f["ball_speed"].fillna(0).to_numpy() > p["fast_ball_mps"])
    return lo + p["bias"]


def set_piece_hints(tags: pd.DataFrame) -> pd.DataFrame:
    """Restarts from SET PIECES tags: start_frame, type, team ("" when the tag names none)."""
    sp = tags[tags["code"] == "SET PIECES"]
    return sp[["start_frame", "end_frame", "type", "team"]].reset_index(drop=True)
