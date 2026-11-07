"""Stage 1: clean the ATD before anything reads it.

- ball: delete points far off the pitch and short detours that would need an impossible speed
- players: delete points far off the pitch and duplicate detections of the same player
- players: velocity from positions by following each player frame to frame through nearest
  same-team matches, never through slot ids (slots swap identity)
Every threshold is in params.yaml under `clean`. Owner: person 1.
"""

import numpy as np

from possession.config import FPS, PITCH_LENGTH_M, PITCH_WIDTH_M
from possession.load import Tracking


def clean_atd(tr: Tracking, params: dict) -> Tracking:
    p = params["clean"]
    ball, ball_rep = clean_ball(tr.ball, p)
    px, py, player_rep = clean_players(tr.px, tr.py, tr.team, p)
    vx, vy = player_velocities(px, py, tr.team, p)

    valid = ~np.isnan(px)
    ours = np.hypot(vx, vy)[valid & ~np.isnan(vx) & ~np.isnan(tr.ps)]
    theirs = tr.ps[valid & ~np.isnan(vx) & ~np.isnan(tr.ps)]
    report = ball_rep | player_rep | {
        "velocity_coverage": float((~np.isnan(vx[valid])).mean()) if valid.any() else 0.0,
        "speed_vs_atd_median_abs_diff_mps": float(np.median(np.abs(ours - theirs))) if ours.size else None,
    }
    return Tracking(
        ball=ball, px=px, py=py, ps=tr.ps, team=tr.team, vx=vx, vy=vy,
        ball_observed=~np.isnan(ball[:, 0]), report=report,
    )


def off_pitch(x: np.ndarray, y: np.ndarray, margin: float) -> np.ndarray:
    return (x < -margin) | (x > PITCH_LENGTH_M + margin) | (y < -margin) | (y > PITCH_WIDTH_M + margin)


def clean_ball(ball: np.ndarray, p: dict) -> tuple[np.ndarray, dict]:
    """Remove off-pitch points and short detours that are impossible on both sides.

    The observed points are split into runs wherever two consecutive observations (gaps allowed)
    would need more than max_ball_speed. A run shorter than ball_min_run_frames is deleted when the
    runs either side of it are consistent with each other, i.e. it is a detour, not a real jump.
    """
    ball = ball.copy()
    n_raw = int((~np.isnan(ball[:, 0])).sum())
    far = off_pitch(ball[:, 0], ball[:, 1], p["offpitch_margin_m"])
    ball[far] = np.nan

    n_spikes = 0
    for _ in range(3):
        removed = _remove_detours(ball, p)
        n_spikes += removed
        if removed == 0:
            break
    n_kept = int((~np.isnan(ball[:, 0])).sum())
    return ball, {
        "ball_observed_raw": n_raw,
        "ball_removed_offpitch": int(far.sum()),
        "ball_removed_detours": n_spikes,
        "ball_observed_clean": n_kept,
    }


def _feasible(a: np.ndarray, b: np.ndarray, frames: np.ndarray, p: dict) -> np.ndarray:
    reach = p["max_ball_speed_mps"] * frames / FPS + p["position_tolerance_m"]
    return np.hypot(*(b - a).T) <= reach


def _remove_detours(ball: np.ndarray, p: dict) -> int:
    obs = np.flatnonzero(~np.isnan(ball[:, 0]))
    if obs.size < 3:
        return 0
    ok = _feasible(ball[obs[:-1]], ball[obs[1:]], np.diff(obs), p)
    run_id = np.concatenate([[0], np.cumsum(~ok)])
    starts = np.flatnonzero(np.diff(run_id, prepend=-1))
    ends = np.append(starts[1:], obs.size) - 1          # inclusive, positions in obs
    drop = np.zeros(obs.size, dtype=bool)
    for r in range(1, len(starts) - 1):
        s, e = starts[r], ends[r]
        if obs[e] - obs[s] + 1 >= p["ball_min_run_frames"]:
            continue
        before, after = obs[s - 1], obs[e + 1]
        if _feasible(ball[[before]], ball[[after]], np.array([after - before]), p)[0]:
            drop[s : e + 1] = True
    ball[obs[drop]] = np.nan
    return int(drop.sum())


def clean_players(px: np.ndarray, py: np.ndarray, team: np.ndarray, p: dict):
    """Delete far off-pitch points and the second of two same-team detections closer than
    duplicate_radius_m in the same frame."""
    px, py = px.copy(), py.copy()
    far = off_pitch(px, py, p["offpitch_margin_m"])
    px[far] = np.nan
    py[far] = np.nan

    n_dup = 0
    for label in ("H", "A"):
        cols = np.flatnonzero(team == label)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                dup = np.hypot(px[:, a] - px[:, b], py[:, a] - py[:, b]) < p["duplicate_radius_m"]
                px[dup, b] = np.nan
                py[dup, b] = np.nan
                n_dup += int(dup.sum())
    return px, py, {"player_removed_offpitch": int(far.sum()), "player_removed_duplicates": n_dup}


def nearest_link(px: np.ndarray, py: np.ndarray, team: np.ndarray, step: int, gate: float) -> np.ndarray:
    """link[t, j] = slot at frame t+step holding the same-team player nearest to slot j at frame t,
    within `gate` metres, else -1. step is +1 or -1."""
    n, s = px.shape
    link = np.full((n, s), -1, dtype=np.int16)
    same_team = (team[:, None] == team[None, :]) & (team[:, None] != "")
    t = np.arange(max(0, -step), n - max(0, step))
    for c in range(0, t.size, 2000):
        tt = t[c : c + 2000]
        dx = px[tt][:, :, None] - px[tt + step][:, None, :]
        dy = py[tt][:, :, None] - py[tt + step][:, None, :]
        d = np.hypot(dx, dy)
        d = np.where(same_team[None] & ~np.isnan(d), d, np.inf)
        best = d.argmin(axis=2)
        found = np.take_along_axis(d, best[:, :, None], axis=2)[:, :, 0] <= gate
        link[tt] = np.where(found, best, -1)
    return link


def player_velocities(px: np.ndarray, py: np.ndarray, team: np.ndarray, p: dict):
    """Velocity over ±k frames, following each player through nearest same-team links.
    Falls back to one-sided when the chain breaks on one side; NaN when it breaks on both."""
    k = p["velocity_halfwindow_frames"]
    gate = p["max_player_speed_mps"] / FPS + p["position_tolerance_m"]
    n, s = px.shape
    rows = np.arange(n)[:, None]

    def follow(step):
        link = nearest_link(px, py, team, step, gate)
        cur = np.tile(np.arange(s, dtype=np.int64), (n, 1))
        cur[np.isnan(px)] = -1
        for i in range(k):
            r = np.clip(rows + i * step, 0, n - 1)
            nxt = np.where(cur >= 0, link[r, np.maximum(cur, 0)], -1)
            edge = (rows + i * step < 0) | (rows + i * step >= n) | (rows + (i + 1) * step < 0) | (rows + (i + 1) * step >= n)
            cur = np.where(edge, -1, nxt)
        r = np.clip(rows + k * step, 0, n - 1)
        ok = cur >= 0
        x = np.where(ok, px[r, np.maximum(cur, 0)], np.nan)
        y = np.where(ok, py[r, np.maximum(cur, 0)], np.nan)
        return x, y

    x_next, y_next = follow(+1)
    x_prev, y_prev = follow(-1)
    span = k / FPS
    both = ~np.isnan(x_next) & ~np.isnan(x_prev)
    vx = np.where(both, (x_next - x_prev) / (2 * span), np.nan)
    vy = np.where(both, (y_next - y_prev) / (2 * span), np.nan)
    only_next = ~both & ~np.isnan(x_next)
    only_prev = ~both & ~np.isnan(x_prev)
    vx = np.where(only_next, (x_next - px) / span, vx)
    vy = np.where(only_next, (y_next - py) / span, vy)
    vx = np.where(only_prev, (px - x_prev) / span, vx)
    vy = np.where(only_prev, (py - y_prev) / span, vy)
    return vx, vy
