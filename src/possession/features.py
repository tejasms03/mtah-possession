"""Stage 3: per-frame notes for the models, identical for our ATD and degraded IDSSE.

Two blocks per frame:
- game block (model A, dead or alive): ball state, how far out / still / missing it is, how fast
  everyone is moving and whether that is dropping.
- team block, built once per team (model B, whose ball): distances from the ball to that team's
  5 nearest players, players near the ball, time since / until the team's touches, team shape
  in that team's attacking direction (x always grows towards the goal it attacks).

Model B sees "own" vs "opp" columns and never which club is which, so it cannot learn that home
teams have the ball more; `team_matrix(..., own="A")` gives the same frame from Athletic's side.
Every column is then averaged over centred windows (and before/after windows for a few) so the
model sees seconds of play, not one frozen frame.
Owner: person 1.
"""

import numpy as np
import pandas as pd

from possession import deadball
from possession.ballfilter import BallTrack
from possession.config import FPS
from possession.load import Tracking

CAP_S = 30.0          # "seconds since / until" is capped here
CAP_M = 60.0          # distances to absent players are capped here


def attack_sign(tr: Tracking, halves, pitch) -> np.ndarray:
    """+1 on frames where H attacks towards +x, -1 where towards -x, 0 outside the halves.

    Over a half, the team defending the left goal nearly always has someone (its keeper) deep on
    the left and rarely anyone deep on the right. Score each team by
    share(frames with a player at x < 12% of L) - share(frames with one at x > 88% of L);
    H defends the left (attacks +x) when its score beats A's. Kick-off positions were tried
    first and failed: the tactical camera often sees only a few players at kick-off.
    BAR-ATH: matches the metadata (H left_to_right in the 1st half) in both halves."""
    L = pitch[0]
    sign = np.zeros(len(tr.px), dtype=np.int8)
    for _, a, b in halves:
        score = {}
        for team in ("H", "A"):
            x = tr.px[a : b + 1][:, tr.team == team]
            deep_left = np.min(np.where(np.isnan(x), np.inf, x), axis=1, initial=np.inf) < 0.12 * L
            deep_right = np.max(np.where(np.isnan(x), -np.inf, x), axis=1, initial=-np.inf) > 0.88 * L
            score[team] = deep_left.mean() - deep_right.mean()
        sign[a : b + 1] = 1 if score["H"] > score["A"] else -1
    return sign


def since_until(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seconds since the last and until the next True in events (capped at CAP_S)."""
    n = len(events)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(events, idx, -10**9))
    nxt = np.minimum.accumulate(np.where(events, idx, 10**9)[::-1])[::-1]
    return np.minimum((idx - last) / FPS, CAP_S), np.minimum((nxt - idx) / FPS, CAP_S)


def team_block(tr: Tracking, bt: BallTrack, team: str, sign: np.ndarray, pitch) -> pd.DataFrame:
    L, W = pitch
    cols = tr.team == team
    px, py = tr.px[:, cols], tr.py[:, cols]
    d = np.hypot(px - bt.x[:, None], py - bt.y[:, None])
    d = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
    k = min(5, d.shape[1])
    near = np.full((len(d), 5), CAP_M)
    near[:, :k] = np.minimum(d[:, :k], CAP_M)
    known = ~np.isnan(bt.x)
    near[~known] = np.nan

    to_att = lambda x: np.where(sign > 0, x, L - x)               # noqa: E731
    with np.errstate(invalid="ignore"):
        cx = np.nanmean(px, axis=1)
        out = {f"d{i + 1}": near[:, i] for i in range(5)}
        out |= {
            "n2": np.where(known, (d <= 2).sum(axis=1), np.nan),
            "n5": np.where(known, (d <= 5).sum(axis=1), np.nan),
            "cx": to_att(cx),
            "cy_abs": np.abs(np.nanmean(py, axis=1) - W / 2),
            "sx": np.nanstd(px, axis=1),
            "sy": np.nanstd(py, axis=1),
            "line": to_att(np.where(sign > 0, np.nanmin(px, axis=1), np.nanmax(px, axis=1))),
            "front": to_att(np.where(sign > 0, np.nanmax(px, axis=1), np.nanmin(px, axis=1))),
            "vx": np.nanmean(tr.vx[:, cols], axis=1) * np.where(sign > 0, 1, -1) if tr.vx is not None else np.nan,
            "speed": np.nanmean(np.hypot(tr.vx[:, cols], tr.vy[:, cols]), axis=1) if tr.vx is not None else np.nan,
            "n_tracked": (~np.isnan(px)).sum(axis=1).astype(float),
            "ball_x": to_att(bt.x),
        }
    since, until = since_until(bt.touch_team == team)
    out |= {"touch_since": since, "touch_until": until}
    return pd.DataFrame(out)


def game_block(tr: Tracking, bt: BallTrack, params: dict, pitch) -> pd.DataFrame:
    f = deadball.frame_features(tr, bt, params, pitch)
    L, W = pitch
    f["ball_known"] = (~np.isnan(bt.x)).astype(float)
    f["ball_seen"] = bt.observed.astype(float)
    f["ball_centre_dx"] = np.abs(bt.x - L / 2)
    f["ball_side_dy"] = np.abs(bt.y - W / 2)
    since, until = since_until(bt.touch)
    f["touch_since"], f["touch_until"] = since, until
    f["n_tracked"] = (~np.isnan(tr.px[:, np.isin(tr.team, ["H", "A"])])).sum(axis=1).astype(float)
    return f.drop(columns=["ball_out"])


def windowed(df: pd.DataFrame, windows_s, edge_cols=(), edge_s=1.0) -> pd.DataFrame:
    """Add centred means over each window, and for edge_cols the mean of the edge_s seconds just
    before and just after the frame (where a change shows up)."""
    out = {c: df[c].to_numpy(dtype=float) for c in df.columns}
    for c in df.columns:
        a = df[c].to_numpy(dtype=float)
        for w in windows_s:
            out[f"{c}_m{w:g}"] = deadball.centred_mean(a, max(1, int(round(w * FPS))))
    k = int(round(edge_s * FPS))
    for c in edge_cols:
        a = df[c].to_numpy(dtype=float)
        m = deadball.centred_mean(a, k)
        past, fut = np.full_like(a, np.nan), np.full_like(a, np.nan)
        past[k // 2 + 1 :] = m[: -(k // 2 + 1)]
        fut[: -(k // 2 + 1)] = m[k // 2 + 1 :]
        out[f"{c}_before"], out[f"{c}_after"] = past, fut
    return pd.DataFrame(out).astype(np.float32)


def extra_team(tr: Tracking, bt: BallTrack, team: str) -> pd.DataFrame:
    """Idea 2 (off unless features.extra_team): who is the ball going to, who is under pressure,
    and touch runs. All from the ball track and both teams' positions; no slot identities."""
    cols, opp = tr.team == team, np.isin(tr.team, ["H", "A"]) & (tr.team != team)
    px, py = tr.px[:, cols], tr.py[:, cols]
    d = np.hypot(px - bt.x[:, None], py - bt.y[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    j = d.argmin(axis=1)
    rows = np.arange(len(d))
    has = np.isfinite(d[rows, j]) & ~np.isnan(bt.x)
    cx, cy = np.where(has, px[rows, j], np.nan), np.where(has, py[rows, j], np.nan)

    to_x, to_y = cx - bt.x, cy - bt.y
    dist = np.maximum(np.hypot(to_x, to_y), 0.1)
    speed = np.nan_to_num(bt.speed)
    approach = (np.nan_to_num(bt.vx) * to_x + np.nan_to_num(bt.vy) * to_y) / dist     # m/s towards carrier
    heading = np.where(speed > 2.0, approach / np.maximum(speed, 1e-6), 0.0)           # cosine

    od = np.hypot(tr.px[:, opp] - cx[:, None], tr.py[:, opp] - cy[:, None])
    pressure = np.where(has, (np.nan_to_num(od, nan=np.inf) <= 3.0).sum(axis=1), np.nan)

    own_touch = (bt.touch_team == team).astype(float)
    k, n = 5 * 25, len(own_touch)
    c = np.r_[0.0, np.cumsum(own_touch)]
    i = np.arange(n)
    past = c[i + 1] - c[np.maximum(0, i - k)]                 # own touches in the last 5 s
    future = c[np.minimum(n, i + k + 1)] - c[i]               # ... and in the next 5 s
    since_own, _ = since_until(bt.touch_team == team)
    since_any, _ = since_until(bt.touch)
    return pd.DataFrame({
        "heading": np.where(has, heading, np.nan), "approach": np.where(has, approach, np.nan),
        "pressure": pressure, "touches_past5": past, "touches_next5": future,
        "last_toucher": (since_own <= since_any).astype(float),
    })


def build(tr: Tracking, bt: BallTrack, params: dict, halves, pitch) -> dict:
    """{'game': game block, 'H': H team block, 'A': A team block}, each already windowed."""
    p = params["features"]
    sign = attack_sign(tr, halves, pitch)
    game = windowed(game_block(tr, bt, params, pitch), p["game_windows_s"], ("spd", "ball_speed"))
    blocks = {"game": game}
    for team in ("H", "A"):
        tb = team_block(tr, bt, team, sign, pitch)
        if p.get("extra_team"):
            tb = pd.concat([tb, extra_team(tr, bt, team)], axis=1)
        blocks[team] = windowed(tb, p["team_windows_s"], ("d1", "n2", "cx", "vx"))
    return blocks


def dead_matrix(blocks: dict) -> pd.DataFrame:
    """Model A input: the game block plus team-order-free summaries of both teams."""
    h, a = blocks["H"], blocks["A"]
    both = pd.DataFrame({
        "d1_min": np.fmin(h["d1"], a["d1"]), "d1_min_m2": np.fmin(h["d1_m2"], a["d1_m2"]),
        "n5_sum": h["n5"] + a["n5"], "speed_mean": (h["speed"] + a["speed"]) / 2,
        "speed_mean_m4": (h["speed_m4"] + a["speed_m4"]) / 2,
        "sx_sum": h["sx"] + a["sx"], "line_gap": np.abs(h["line"] + a["line"] - 2 * h["line"].mean()),
    })
    return pd.concat([blocks["game"], both.astype(np.float32)], axis=1)


def team_matrix(blocks: dict, own: str = "H") -> pd.DataFrame:
    """Model B input from `own`'s side: own_*, opp_* and own-minus-opp for the key columns."""
    opp = "A" if own == "H" else "H"
    o, p = blocks[own], blocks[opp]
    diff = {f"diff_{c}": o[c] - p[c] for c in o.columns if c.split("_")[0] in ("d1", "d2", "n2", "n5", "touch")}
    return pd.concat([o.add_prefix("own_"), p.add_prefix("opp_"), pd.DataFrame(diff),
                      blocks["game"][["ball_known", "ball_seen", "gap_s", "ball_speed"]]], axis=1)
