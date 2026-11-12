"""Stage 2: smooth the ball, bridge short gaps, find touches.

- smooth: Savitzky-Golay on each run of observed frames (position and velocity)
- bridge: gaps up to max_bridge_frames are filled by straight-line interpolation between the
  last and next observation (offline, so both ends are known). Longer gaps stay NaN.
- touches: the ball's velocity changes sharply (speed or direction) while a player is close.
  Each touch carries the team of the nearest player.
Every threshold is in params.yaml under `ball`. Owner: person 1.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from possession.config import FPS
from possession.load import Tracking


@dataclass
class BallTrack:
    x: np.ndarray            # (N,) smoothed, short gaps bridged, NaN in long gaps
    y: np.ndarray
    vx: np.ndarray           # (N,) m/s, NaN where unknown
    vy: np.ndarray
    observed: np.ndarray     # (N,) bool, ball in the cleaned ATD
    bridged: np.ndarray      # (N,) bool, filled by interpolation
    gap_len: np.ndarray      # (N,) frames in the gap this frame sits in, 0 when observed
    d_H: np.ndarray          # (N,) distance to the nearest Barcelona player, inf if none
    d_A: np.ndarray          # (N,) distance to the nearest Athletic player
    touch: np.ndarray        # (N,) bool
    touch_team: np.ndarray   # (N,) "H" / "A" / "" at touches, "" elsewhere
    report: dict

    @property
    def speed(self) -> np.ndarray:
        return np.hypot(self.vx, self.vy)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) half-open index pairs of the True runs in mask."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def smooth(ball: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Savitzky-Golay position and velocity on each observed run; runs shorter than the window
    keep raw positions and get a plain difference for velocity.
    Also returns `steady`: velocity is trustworthy (at least half a window inside a run; the
    filter's estimate at a run's ends is several times noisier)."""
    pos = ball.copy()
    vel = np.full_like(ball, np.nan)
    steady = np.zeros(len(ball), dtype=bool)
    w, k = p["savgol_window"], p["savgol_order"]
    for s, e in runs(~np.isnan(ball[:, 0])):
        seg = ball[s:e]
        if e - s >= w:
            pos[s:e] = savgol_filter(seg, w, k, axis=0)
            vel[s:e] = savgol_filter(seg, w, k, deriv=1, delta=1 / FPS, axis=0)
            steady[s + w // 2 : e - w // 2] = True
        elif e - s >= 2:
            vel[s:e] = np.gradient(seg, 1 / FPS, axis=0)
    return pos, vel, steady


def bridge(pos: np.ndarray, vel: np.ndarray, p: dict):
    """Interpolate gaps of at most max_bridge_frames. Returns pos, vel, bridged, gap_len."""
    pos, vel = pos.copy(), vel.copy()
    n = len(pos)
    missing = np.isnan(pos[:, 0])
    bridged = np.zeros(n, dtype=bool)
    gap_len = np.zeros(n, dtype=np.int64)
    for s, e in runs(missing):
        gap_len[s:e] = e - s
        if s == 0 or e == n or e - s > p["max_bridge_frames"]:
            continue
        a, b = pos[s - 1], pos[e]
        frac = (np.arange(s, e) - (s - 1)) / (e - (s - 1))
        pos[s:e] = a + frac[:, None] * (b - a)
        vel[s:e] = (b - a) * FPS / (e - (s - 1))
        bridged[s:e] = True
    return pos, vel, bridged, gap_len


def nearest_by_team(bx, by, px, py, team) -> tuple[np.ndarray, np.ndarray]:
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    d_H = d[:, team == "H"].min(axis=1, initial=np.inf)
    d_A = d[:, team == "A"].min(axis=1, initial=np.inf)
    return d_H, d_A


def detect_touches(vel, steady, d_H, d_A, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """A touch = the velocity just after differs sharply from the velocity just before
    (by touch_dv_mps, or by touch_angle_deg when the ball is moving), with a player within
    touch_radius_m. Candidates closer than touch_min_separation_frames keep only the strongest."""
    n = len(vel)
    h = p["touch_half_window_frames"]
    before = np.full_like(vel, np.nan)
    after = np.full_like(vel, np.nan)
    before[h:] = np.where(steady[:-h, None], vel[:-h], np.nan)
    after[:-h] = np.where(steady[h:, None], vel[h:], np.nan)
    dv = np.hypot(*(after - before).T)
    sb, sa = np.hypot(*before.T), np.hypot(*after.T)
    cos = np.einsum("ij,ij->i", before, after) / np.maximum(sb * sa, 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    moving = np.minimum(sb, sa) >= p["touch_min_speed_mps"]

    near = np.minimum(d_H, d_A) <= p["touch_radius_m"]
    cand = near & ~np.isnan(dv) & ((dv >= p["touch_dv_mps"]) | (moving & (angle >= p["touch_angle_deg"])))
    strength = np.where(cand, np.nan_to_num(dv), -1.0)

    touch = np.zeros(n, dtype=bool)
    sep = p["touch_min_separation_frames"]
    for i in np.flatnonzero(cand)[np.argsort(-strength[cand], kind="stable")]:
        if not touch[max(0, i - sep) : i + sep + 1].any():
            touch[i] = True
    touch_team = np.where(touch, np.where(d_H <= d_A, "H", "A"), "")
    return touch, touch_team


def run(tr: Tracking, params: dict) -> BallTrack:
    p = params["ball"]
    pos, vel, steady = smooth(tr.ball, p)
    pos, vel, bridged, gap_len = bridge(pos, vel, p)
    steady |= bridged
    d_H, d_A = nearest_by_team(pos[:, 0], pos[:, 1], tr.px, tr.py, tr.team)
    touch, touch_team = detect_touches(vel, steady, d_H, d_A, p)
    observed = ~np.isnan(tr.ball[:, 0])
    report = {
        "ball_bridged_frames": int(bridged.sum()),
        "ball_known_frames": int((~np.isnan(pos[:, 0])).sum()),
        "touches": int(touch.sum()),
        "touches_H": int((touch_team == "H").sum()),
        "touches_A": int((touch_team == "A").sum()),
    }
    return BallTrack(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], observed, bridged, gap_len,
                     d_H, d_A, touch, touch_team, report)
