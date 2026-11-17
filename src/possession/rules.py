"""Bench-test rules applied to the decoded labels, one half at a time (all off unless their
decode.* switch is set). Each takes the full label array `out`, the half's first and last frame
a, b, the ball track, a context dict and the decode params, and edits `out` in place.

ctx (built by `context`): per-frame "spd" (mean player speed), "block" (team-block speed),
"duels" (opponent pairs within 2 m), from model A's columns; "tags": Smart Tagging rows
(code, type, team, start_frame, end_frame) or None; "pitch".
"""

import numpy as np
import pandas as pd

from possession.config import FPS

OUT_TYPES = ("throw", "corner", "goal kick")
FOUL_TYPES = ("free kick",)
# Smart Tagging phases whose team is the team on the ball (DEFENDING* name the team without it)
ATTACK_CODES = ("BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES", "LONG BALL",
                "ATTACKING TRANSITION", "BALL IN THE BOX")


def context(prep: dict, params: dict, tags: pd.DataFrame | None = None) -> dict:
    from possession import model

    cols = model.dead_columns(params)
    pick = lambda c: prep["dead"][:, cols.index(c)] if c in cols else None  # noqa: E731
    t = None
    if tags is not None and len(tags):
        t = tags.copy()
        t["type"] = t["type"].fillna("").str.lower().str.replace("-", " ")
        t["code"] = t["code"].fillna("").str.upper()
    return {"spd": pick("spd"), "block": pick("mx_block"), "duels": pick("mx_duels"), "tags": t,
            "pitch": prep["pitch"]}


def _spells(out, a, b):
    seg = out[a : b + 1]
    e = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    s, t = np.r_[0, e] + a, np.r_[e, len(seg)] + a
    return [(int(x), int(y), out[x]) for x, y in zip(s, t)]


def _dead_spells(out, a, b):
    """(d0, d1, before, after) for D spells inside the half with H/A on both sides."""
    sp = _spells(out, a, b)
    res = []
    for k in range(1, len(sp) - 1):
        d0, d1, lab = sp[k]
        if lab == "D" and sp[k - 1][2] in "HA" and sp[k + 1][2] in "HA":
            res.append((d0, d1, sp[k - 1][2], sp[k + 1][2]))
    return res


def _other(t):
    return "A" if t == "H" else "H"


def _line_dist(x, y, pitch):
    L, W = pitch
    return np.minimum.reduce([x, L - x, y, W - y])


def _nearest(bt, f, max_m, gap_m=0.0):
    dh, da = bt.d_H[f], bt.d_A[f]
    if min(dh, da) > max_m or abs(dh - da) < gap_m:
        return ""
    return "H" if dh < da else "A"


def _tags_in(ctx, lo, hi, codes=("SET PIECES",)):
    t = ctx.get("tags")
    if t is None:
        return t
    return t[t["code"].isin(codes) & (t["start_frame"] >= lo) & (t["start_frame"] < hi)]


def restart_kind(out_d0, d1, bt, ctx, p):
    """'out' (throw-in, corner, goal kick), 'foul' (free kick) or '' for the restart at d1."""
    if p.get("rt_tags", 0):
        t = _tags_in(ctx, out_d0, d1 + FPS)
        if t is not None and len(t):
            typ = t["type"].iloc[-1]
            if any(k in typ for k in OUT_TYPES):
                return "out", t["team"].iloc[-1]
            if any(k in typ for k in FOUL_TYPES):
                return "foul", t["team"].iloc[-1]
    L, W = ctx["pitch"]
    w = FPS
    idx = np.flatnonzero(bt.observed[d1 - w : d1 + w]) + d1 - w
    if not len(idx):
        return "", ""
    f = idx[np.argmin(abs(idx - d1))]
    x, y = bt.x[f], bt.y[f]
    side = min(y, W - y)
    end = min(x, L - x)
    if side <= p.get("rt_line_m", 2.0) or (end <= 6.5 and abs(y - W / 2) <= 9.2) or (end <= 2 and side <= 2):
        return "out", ""
    return "foul", ""


def restart_team(out, a, b, bt, ctx, p):
    """Rules 1-2: the restart says who touched last. Ball out with the same team labelled on both
    sides -> the other team touched it last: it owns the ball from its last touch (within rt_s) to
    the stoppage. rt_foul: a free kick to the team that did not have it before -> the fouled team
    owns the ball from its last touch to the stoppage."""
    look = int(p["rt_s"] * FPS)
    for d0, d1, before, after in _dead_spells(out, a, b):
        kind, tag_team = restart_kind(d0, d1, bt, ctx, p)
        q = tag_team or after
        if kind == "out" and before == q:
            o = _other(q)
        elif kind == "foul" and p.get("rt_foul", 0) and before != q:
            o = q
        else:
            continue
        lo = max(a, d0 - look)
        idx = np.flatnonzero(bt.touch_team[lo:d0] == o)
        if len(idx):
            f = lo + idx[-1]
        elif p.get("rt_fallback_s", 0):
            f = max(a, d0 - int(p["rt_fallback_s"] * FPS))
        else:
            continue
        if np.all(out[f:d0] != "D"):
            out[f:d0] = o


def last_near(out, a, b, bt, ctx, p):
    """Rule 3: at each stoppage, the team clearly nearest the ball at its last sighting (within
    ln_s) owns the ball from that sighting to the stoppage."""
    look = int(p["ln_s"] * FPS)
    starts = np.flatnonzero((out[a + 1 : b + 1] == "D") & (out[a:b] != "D")) + a + 1
    for d0 in starts:
        lo = max(a, d0 - look)
        idx = np.flatnonzero(bt.observed[lo:d0])
        if not len(idx):
            continue
        f = lo + idx[-1]
        t = _nearest(bt, f, p.get("ln_m", 2.0), p.get("ln_gap_m", 1.0))
        if t and t != out[d0 - 1] and np.all(out[f:d0] != "D"):
            out[f:d0] = t


def team_flip(out, a, b, bt, ctx, p):
    """Rule 4: the ball seen and one team clearly controlling it (within tf_m, the other team
    tf_gap_m further) for at least tf_s -> that team owns those frames."""
    seg = slice(a, b + 1)
    dh, da = bt.d_H[seg], bt.d_A[seg]
    obs = bt.observed[seg]
    m, g = p.get("tf_m", 1.5), p.get("tf_gap_m", 2.0)
    ctrl = np.where(obs & (dh <= m) & (da - dh >= g), "H", np.where(obs & (da <= m) & (dh - da >= g), "A", ""))
    need = int(p["tf_s"] * FPS)
    for s, t, lab in _spells(ctrl, 0, len(ctrl) - 1):
        if lab in ("H", "A") and t - s >= need:
            cur = out[a + s : a + t]
            if np.all(np.isin(cur, ["H", "A"])) and np.mean(cur != lab) > 0.5:
                out[a + s : a + t] = lab


def dead_check(out, a, b, bt, ctx, p):
    """Rule 5: a stoppage with the same team on both sides, the ball mostly missing and the players
    still moving (mean speed >= dc_spd, team blocks >= dc_block) -> live, that team's."""
    for d0, d1, before, after in _dead_spells(out, a, b):
        if before != after:
            continue
        if p.get("dc_max_s", 0) and d1 - d0 > p["dc_max_s"] * FPS:
            continue
        if np.mean(~bt.observed[d0:d1]) < p.get("dc_miss", 0.8):
            continue
        if p.get("dc_spd", 0) and (ctx["spd"] is None or np.nanmean(ctx["spd"][d0:d1]) < p["dc_spd"]):
            continue
        if p.get("dc_block", 0) and (ctx["block"] is None or np.nanmean(ctx["block"][d0:d1]) < p["dc_block"]):
            continue
        t = _tags_in(ctx, d0 - 2 * FPS, d1 + FPS)
        if t is not None and len(t):
            continue
        out[d0:d1] = before


def dead_reason(out, a, b, bt, ctx, p):
    """Rule 6: a stoppage with the same team on both sides needs a reason: the ball seen within
    nr_m of a line around its start (nr_s either side), or a set-piece tag. None -> live."""
    pitch, w, m = ctx["pitch"], int(p["nr_s"] * FPS), p.get("nr_m", 3.0)
    for d0, d1, before, after in _dead_spells(out, a, b):
        if before != after:
            continue
        lo, hi = max(a, d0 - w), min(d1, d0 + w)
        idx = np.flatnonzero(bt.observed[lo:hi]) + lo
        if len(idx) and np.nanmin(_line_dist(bt.x[idx], bt.y[idx], pitch)) <= m:
            continue
        t = _tags_in(ctx, d0 - 2 * FPS, d1 + FPS)
        if t is not None and len(t):
            continue
        out[d0:d1] = before


def gap_ends(out, a, b, bt, ctx, p):
    """Rule 7: inside a run of missing raw ball (>= gr_min_s), mode 1: the team clearly nearest
    when the ball vanished and when it came back, if the same, owns every live frame of the run;
    mode 2: the vanishing end alone decides."""
    obs = bt.observed
    miss = ~obs[a : b + 1]
    e = np.flatnonzero(np.diff(np.r_[0, miss.astype(np.int8), 0]))
    need, m = int(p.get("gr_min_s", 1.0) * FPS), p.get("gr_m", 2.0)
    for s, t in zip(e[::2] + a, e[1::2] + a):
        if t - s < need or s == a or t > b:
            continue
        ta = _nearest(bt, s - 1, m, p.get("gr_gap_m", 1.0))
        tb = _nearest(bt, t, m, p.get("gr_gap_m", 1.0))
        team = ta if (p["gr_mode"] == 2 or ta == tb) else ""
        if team:
            live = np.isin(out[s:t], ["H", "A"])
            out[s:t][live] = team


def snap_speed(out, a, b, bt, ctx, p):
    """Rule 8: restart (D -> H/A) moves to the nearest frame within fs_s where the seen ball
    speeds up past fs_v; fs_stop: stoppage start moves to the nearest frame where it slows below."""
    v, w = p["fs_v"], int(p.get("fs_s", 1.0) * FPS)
    sp = np.where(bt.observed, bt.speed, np.nan)
    up = np.r_[False, (sp[1:] >= v) & (sp[:-1] < v)]
    down = np.r_[False, (sp[1:] < v) & (sp[:-1] >= v)]
    spl = _spells(out, a, b)
    for k in range(1, len(spl)):
        t, prev, cur = spl[k][0], spl[k - 1], spl[k]
        lo, hi = prev[0] + 1, cur[1] - 1
        if prev[2] == "D" and cur[2] in "HA":
            cand = np.flatnonzero(up[max(lo, t - w) : min(hi, t + w) + 1]) + max(lo, t - w)
        elif cur[2] == "D" and prev[2] in "HA" and p.get("fs_stop", 0):
            cand = np.flatnonzero(down[max(lo, t - w) : min(hi, t + w) + 1]) + max(lo, t - w)
        else:
            continue
        if len(cand):
            n = int(cand[np.argmin(abs(cand - t))])
            if n < t:
                out[n:t] = cur[2]
            elif n > t:
                out[t:n] = prev[2]


def snap_tag(out, a, b, bt, ctx, p):
    """Rule 9: restart moves to a set-piece tag's start + ts_off_s, if within ts_w_s."""
    t = ctx.get("tags")
    if t is None:
        return
    w, off = int(p.get("ts_w_s", 3.0) * FPS), int(p["ts_off_s"] * FPS)
    spl = _spells(out, a, b)
    for k in range(1, len(spl)):
        prev, cur = spl[k - 1], spl[k]
        if not (prev[2] == "D" and cur[2] in "HA"):
            continue
        r = cur[0]
        tg = _tags_in(ctx, prev[0] - off, cur[1] - off)
        if tg is None or not len(tg):
            continue
        n = tg["start_frame"].to_numpy() + off
        n = n[(abs(n - r) <= w) & (n > prev[0]) & (n < cur[1])]
        if len(n):
            n = int(n[np.argmin(abs(n - r))])
            if n < r:
                out[n:r] = cur[2]
            elif n > r:
                out[r:n] = "D"


def phase_tags(out, a, b, bt, ctx, p):
    """Rule 10: an attacking-phase tag covering most of a stoppage with the same team on both
    sides (and that team) -> live (pt_live); inside a missing-ball run, frames labelled the other
    team than an attacking-phase tag's team -> the tag's team (pt_gap)."""
    t = ctx.get("tags")
    if t is None:
        return
    ph = t[t["code"].isin(ATTACK_CODES) & (t["team"] != "")]
    if not len(ph):
        return
    cover = np.full(len(out), "", dtype="<U1")
    for s, e, tm in zip(ph["start_frame"], ph["end_frame"], ph["team"]):
        cover[max(a, s) : min(b + 1, e)] = tm
    if p.get("pt_live", 0):
        for d0, d1, before, after in _dead_spells(out, a, b):
            if before == after and np.mean(cover[d0:d1] == before) > 0.8:
                out[d0:d1] = before
    if p.get("pt_gap", 0):
        sel = (~bt.observed) & np.isin(out, ["H", "A"]) & (cover != "") & (out != cover)
        sel[: a] = False
        sel[b + 1 :] = False
        out[sel] = cover[sel]


# order matters: possession-end rules after the dead/live fixes
ORDER = [("dc_on", dead_check), ("nr_s", dead_reason), ("pt_on", phase_tags), ("gr_mode", gap_ends),
         ("tf_s", team_flip), ("rt_s", restart_team), ("ln_s", last_near), ("fs_v", snap_speed),
         ("ts_off_s", snap_tag)]


def apply(out, a, b, bt, ctx, p):
    if ctx is None:
        return
    for key, fn in ORDER:
        if p.get(key, 0):
            fn(out, a, b, bt, ctx, p)
