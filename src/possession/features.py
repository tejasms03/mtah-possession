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
    if params["features"].get("gap_extra"):
        f = pd.concat([f, gap_block(tr, pitch)], axis=1)
    if params["features"].get("motion_extra"):
        f = pd.concat([f, motion_block(tr)], axis=1)
    if params["features"].get("turnover_extra"):
        f = pd.concat([f, turnover_block(bt)], axis=1)
    if params["features"].get("restart_extra"):
        f = pd.concat([f, restart_block(tr, pitch)], axis=1)
    if params["features"].get("ghost_extra"):
        from possession import ghost
        f = pd.concat([f, ghost.ghost_block(tr, bt, pitch, params["features"].get("ghost_for"))], axis=1)
    return f.drop(columns=["ball_out"])


def motion_block(tr: Tracking) -> pd.DataFrame:
    """Team-blind player motion that an average speed hides (off unless features.motion_extra):
    the 3 fastest players, the spread of speeds, the share running (> 4 m/s), the 3 hardest
    speed changes over 0.2 s, how much each team moves as one block (|mean velocity| / mean
    speed, and the block's own speed), and opponent pairs within 2 m (duels). All from player
    positions and velocities only, so it works whether or not the ball is seen."""
    hs = np.isin(tr.team, ["H", "A"])
    if tr.vx is not None:
        vx, vy = tr.vx[:, hs], tr.vy[:, hs]
        s = np.hypot(vx, vy)
    else:
        vx = vy = None
        s = tr.ps[:, hs]
    n = len(s)
    with np.errstate(invalid="ignore"):
        srt = -np.sort(-np.nan_to_num(s, nan=-1.0), axis=1)
        top = np.where(srt[:, :3] < 0, np.nan, srt[:, :3])
        k = 5                                                   # 0.2 s at 25 fps
        dv = np.full_like(s, np.nan)
        if vx is not None:
            dv[k:] = np.hypot(vx[k:] - vx[:-k], vy[k:] - vy[:-k]) / (k / FPS)
        else:
            dv[k:] = np.abs(s[k:] - s[:-k]) / (k / FPS)
        dsrt = -np.sort(-np.nan_to_num(dv, nan=-1.0), axis=1)
        dtop = np.where(dsrt[:, :3] < 0, np.nan, dsrt[:, :3])
        out = {
            "mx_top3": np.nanmean(top, axis=1),
            "mx_sd": np.nanstd(s, axis=1),
            "mx_run": np.nanmean(np.where(np.isnan(s), np.nan, s > 4.0), axis=1),
            "mx_acc3": np.nanmean(dtop, axis=1),
        }
        align, block = [], []
        for team in ("H", "A"):
            c = tr.team[hs] == team
            if vx is not None:
                mvx, mvy = np.nanmean(vx[:, c], axis=1), np.nanmean(vy[:, c], axis=1)
                bs = np.hypot(mvx, mvy)
            else:
                cx = np.nanmean(tr.px[:, hs][:, c], axis=1)
                cy = np.nanmean(tr.py[:, hs][:, c], axis=1)
                bs = np.full(n, np.nan)
                bs[k:] = np.hypot(cx[k:] - cx[:-k], cy[k:] - cy[:-k]) / (k / FPS)
            ms = np.nanmean(s[:, c], axis=1)
            align.append(bs / np.maximum(ms, 0.3))
            block.append(bs)
        out["mx_align"] = np.nanmean(align, axis=0)
        out["mx_block"] = np.nanmean(block, axis=0)
        px, py = tr.px[:, hs], tr.py[:, hs]
        h, a = tr.team[hs] == "H", tr.team[hs] == "A"
        d = np.hypot(px[:, h][:, :, None] - px[:, a][:, None, :], py[:, h][:, :, None] - py[:, a][:, None, :])
        out["mx_duels"] = (np.nan_to_num(d, nan=np.inf) <= 2.0).sum(axis=(1, 2)).astype(float)
    return pd.DataFrame(out)


def turnover_block(bt: BallTrack) -> pd.DataFrame:
    """Turnover signs, team-blind (off unless features.turnover_extra; change detector only, since
    they use ball motion, which DFL freezes at stoppages): how sharply the ball turns and changes
    speed over 0.2 s, both teams within 2 m of the ball (a contest), how close the two nearest
    distances are, whether the nearer team flips and stays flipped (0.5 s either side), and
    whether this touch is by the other team than the previous touch."""
    n = len(bt.x)
    k = 5
    vx, vy = bt.vx, bt.vy
    bx, by, ax, ay = (np.full(n, np.nan) for _ in range(4))
    bx[k:], by[k:] = vx[:-k], vy[:-k]
    ax[:-k], ay[:-k] = vx[k:], vy[k:]
    sb, sa = np.hypot(bx, by), np.hypot(ax, ay)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (bx * ax + by * ay) / np.maximum(sb * sa, 1e-9)
        angle = np.where(np.minimum(sb, sa) >= 1.0, np.degrees(np.arccos(np.clip(cos, -1, 1))), 0.0)
    known = ~np.isnan(bt.x)
    dh, da = np.minimum(bt.d_H, CAP_M), np.minimum(bt.d_A, CAP_M)
    near = np.where(known, np.sign(da - dh), 0.0)              # +1 H nearer, -1 A nearer
    h = 12
    c = np.r_[0.0, np.cumsum(near)]
    i = np.arange(n)
    before = (c[i] - c[np.maximum(0, i - h)]) / h
    after = (c[np.minimum(n, i + h)] - c[i]) / h
    flip = np.where(before * after < 0, np.minimum(np.abs(before), np.abs(after)), 0.0)
    tt = bt.touch_team
    idx = np.flatnonzero(bt.touch)
    alt = np.zeros(n)
    if len(idx) > 1:
        sw = idx[1:][tt[idx[1:]] != tt[idx[:-1]]]
        for f in sw:
            alt[max(0, f - h) : f + h] = 1.0
    return pd.DataFrame({
        "to_angle": np.nan_to_num(angle), "to_dv": np.hypot(ax - bx, ay - by),
        "to_contest": (known & (dh <= 2.0) & (da <= 2.0)).astype(float),
        "to_dgap": np.where(known, np.abs(dh - da), np.nan), "to_flip": flip, "to_alt": alt,
    })


def restart_block(tr: Tracking, pitch, chunk: int = 20000) -> pd.DataFrame:
    """restart_layout in chunks of frames (the pairwise distances are frames x players x players)."""
    n = len(tr.px)
    parts = []
    for a in range(0, n, chunk):
        sl = slice(a, min(n, a + chunk))
        sub = Tracking(ball=tr.ball[sl], px=tr.px[sl], py=tr.py[sl], ps=tr.ps[sl], team=tr.team,
                       vx=None if tr.vx is None else tr.vx[sl], vy=None if tr.vy is None else tr.vy[sl])
        parts.append(restart_layout(sub, pitch))
    return pd.concat(parts, ignore_index=True)


def restart_layout(tr: Tracking, pitch) -> pd.DataFrame:
    """Restart layout from players only (off unless features.restart_extra; for model A): the
    player nearest a touchline (a thrower stands on it, negative = outside) and nearest a corner
    flag, the share of players standing still (< 0.5 m/s), the biggest group of still teammates
    within 2 m of each other (a wall), the players inside either penalty box, and the space
    around the most isolated outfield player (opponents keep 9.15 m from a free kick)."""
    L, W = pitch
    hs = np.isin(tr.team, ["H", "A"])
    px, py = tr.px[:, hs], tr.py[:, hs]
    team = tr.team[hs]
    s = np.hypot(tr.vx[:, hs], tr.vy[:, hs]) if tr.vx is not None else tr.ps[:, hs]
    with np.errstate(invalid="ignore"):
        side = np.nanmin(np.minimum(py, W - py), axis=1)
        corners = np.array([[0, 0], [0, W], [L, 0], [L, W]])
        corner = np.nanmin(np.min(np.hypot(px[:, :, None] - corners[None, None, :, 0],
                                           py[:, :, None] - corners[None, None, :, 1]), axis=2), axis=1)
        still = s < 0.5
        still_share = np.nanmean(np.where(np.isnan(s), np.nan, still), axis=1)
        d = np.hypot(px[:, :, None] - px[:, None, :], py[:, :, None] - py[:, None, :])
        d = np.nan_to_num(d, nan=np.inf)
        same = team[:, None] == team[None, :]
        both_still = still[:, :, None] & still[:, None, :]
        wall = ((d <= 2.0) & same[None] & both_still).sum(axis=2).max(axis=1).astype(float)
        inbox = ((py >= W / 2 - 20.16) & (py <= W / 2 + 20.16) & ((px <= 16.5) | (px >= L - 16.5))).sum(axis=1)
        np.einsum("nii->ni", d)[:] = np.inf
        nearest = d.min(axis=2)
        nearest[np.isinf(nearest)] = np.nan
        iso = np.nanmax(np.where((px > 16.5) & (px < L - 16.5), nearest, np.nan), axis=1)
    return pd.DataFrame({
        "rl_side": side, "rl_corner": corner, "rl_still": still_share, "rl_wall": wall,
        "rl_inbox": inbox.astype(float), "rl_iso": np.minimum(iso, CAP_M),
    })


def gap_block(tr: Tracking, pitch) -> pd.DataFrame:
    """Whole-gap ball context (offline: the gap's end is known). For each frame inside a run of
    missing ball: where the ball vanished and where it came back (distance to the nearest line,
    negative outside), how far apart those are, whether it came back near a restart spot, the
    ball's speed just before it vanished, and how far through the gap this frame is. 0 when seen."""
    L, W = pitch
    b = tr.ball
    n = len(b)
    miss = np.isnan(b[:, 0])
    e = np.flatnonzero(np.diff(np.r_[0, miss.astype(np.int8), 0]))
    cols = ["gx_vanish_edge", "gx_back_edge", "gx_jump_m", "gx_back_restart_m", "gx_vanish_speed",
            "gx_frac", "gx_back_in_s"]
    out = {c: np.zeros(n, dtype=float) for c in cols}

    def edge(x, y):
        return min(x, L - x, y, W - y)

    spots = np.array([[0, 0], [0, W], [L, 0], [L, W], [L / 2, W / 2], [5.5, W / 2], [L - 5.5, W / 2],
                      [11, W / 2], [L - 11, W / 2]])

    def restart(x, y):
        return min(float(np.min(np.hypot(spots[:, 0] - x, spots[:, 1] - y))), abs(y), abs(W - y))

    for s, t in zip(e[::2], e[1::2]):
        before = b[s - 1] if s > 0 else None
        after = b[t] if t < n else None
        if before is None and after is None:
            continue
        before = after if before is None else before
        after = before if after is None else after
        k = max(0, s - 10)
        seg = b[k:s]
        seg = seg[~np.isnan(seg[:, 0])]
        vs = float(np.hypot(*(seg[-1] - seg[0])) * 25 / max(1, len(seg) - 1)) if len(seg) > 1 else 0.0
        m = t - s
        out["gx_vanish_edge"][s:t] = edge(*before)
        out["gx_back_edge"][s:t] = edge(*after)
        out["gx_jump_m"][s:t] = np.hypot(*(after - before))
        out["gx_back_restart_m"][s:t] = restart(*after)
        out["gx_vanish_speed"][s:t] = vs
        out["gx_frac"][s:t] = (np.arange(m) + 0.5) / m
        out["gx_back_in_s"][s:t] = np.arange(m)[::-1] / 25
    return pd.DataFrame(out)


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


def carrier_block(tr: Tracking, bt: BallTrack, team: str) -> pd.DataFrame:
    """Ball-carrier notes for one team (off unless features.carrier_extra), not windowed (they
    already look across time). The true labels change hands only when an opponent gains control,
    so: is one of this team's players moving with the ball (close, same velocity, same heading),
    for how long before and after this frame, did the ball slow down on arriving at them, could
    they reach it first, and is this team the last / next team in clear control of the ball."""
    n = len(bt.x)
    cols = tr.team == team
    px, py = tr.px[:, cols], tr.py[:, cols]
    dxy = np.hypot(px - bt.x[:, None], py - bt.y[:, None])
    d = np.where(np.isnan(dxy), np.inf, dxy)
    j = d.argmin(axis=1)
    r = np.arange(n)
    dmin = d[r, j]
    known = ~np.isnan(bt.x) & np.isfinite(dmin)
    bvx, bvy = np.nan_to_num(bt.vx), np.nan_to_num(bt.vy)
    if tr.vx is not None:
        vx, vy = tr.vx[:, cols], tr.vy[:, cols]
    else:
        vx = vy = np.zeros_like(px)
    pvx, pvy = np.nan_to_num(vx[r, j]), np.nan_to_num(vy[r, j])
    relv = np.hypot(bvx - pvx, bvy - pvy)
    bs, ps = np.hypot(bvx, bvy), np.hypot(pvx, pvy)
    with np.errstate(invalid="ignore", divide="ignore"):
        align = np.where((bs > 1) & (ps > 1), (bvx * pvx + bvy * pvy) / np.maximum(bs * ps, 1e-6), 0.0)
        rv_all = np.hypot(bvx[:, None] - np.nan_to_num(vx), bvy[:, None] - np.nan_to_num(vy))
        score = np.nanmax(np.where(np.isfinite(d), np.exp(-(d / 1.2) ** 2) * np.exp(-(rv_all / 3.0) ** 2), 0.0), axis=1)
        # time to reach the ball: 0.3 s reaction, then 7 m/s, helped by the speed already towards it
        rad = -(np.nan_to_num(vx) * (px - bt.x[:, None]) + np.nan_to_num(vy) * (py - bt.y[:, None])) / np.maximum(d, 0.5)
        tti = np.nanmin(np.where(np.isfinite(d), 0.3 + np.maximum(d - np.nan_to_num(rad) * 0.3, 0) / 7.0, np.inf), axis=1)
    other = bt.d_A if team == "H" else bt.d_H
    own_d = bt.d_H if team == "H" else bt.d_A
    close = known & (own_d <= 1.5)
    excl = close & (other >= own_d + 1.0)
    excl_o = known & (other <= 1.5) & (own_d >= other + 1.0)
    c = np.r_[0.0, np.cumsum(close.astype(float))]
    i = np.arange(n)
    dwell = {}
    for k, tag in ((10, "04"), (25, "1")):
        dwell[f"ca_dwell_b{tag}"] = (c[i + 1] - c[np.maximum(0, i + 1 - k)]) / k
        dwell[f"ca_dwell_f{tag}"] = (c[np.minimum(n, i + k)] - c[i]) / k
    # last / next team in clear control (forward and backward fill of exclusive control)
    state = np.where(excl, 1.0, np.where(excl_o, 0.0, np.nan))
    last = pd.Series(state).ffill().to_numpy()
    nxt = pd.Series(state).bfill().to_numpy()
    since_own, until_own = since_until(excl)
    since_opp, until_opp = since_until(excl_o)
    sp = np.where(known, bs, np.nan)
    drop = np.full(n, np.nan)
    drop[5:-5] = sp[:-10] - sp[10:]
    arrive = np.where(close, np.nan_to_num(drop), 0.0)
    out = {
        "ca_relv": np.where(known, np.minimum(relv, 30), np.nan), "ca_align": np.where(known, align, np.nan),
        "ca_score": np.where(known, score, np.nan), "ca_tti": np.where(known, np.minimum(tti, 10), np.nan),
        "ca_excl": excl.astype(float), "ca_last": last, "ca_next": nxt,
        "ca_since": since_own, "ca_until": until_own,
        "ca_lead": np.clip(since_opp - since_own, -CAP_S, CAP_S),       # + = own control more recent
        "ca_lead_next": np.clip(until_opp - until_own, -CAP_S, CAP_S),  # + = own control comes sooner
        "ca_arrive": arrive,
    } | dwell
    return pd.DataFrame(out).astype(np.float32)


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
        if p.get("carrier_extra"):
            blocks[team] = pd.concat([blocks[team], carrier_block(tr, bt, team)], axis=1)
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
    diff = {f"diff_{c}": o[c] - p[c] for c in o.columns if c.split("_")[0] in ("d1", "d2", "n2", "n5", "touch", "ca")}
    return pd.concat([o.add_prefix("own_"), p.add_prefix("opp_"), pd.DataFrame(diff),
                      blocks["game"][["ball_known", "ball_seen", "gap_s", "ball_speed"]]], axis=1)
