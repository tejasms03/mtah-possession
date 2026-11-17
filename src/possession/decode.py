"""Stages 5-6: turn per-frame evidence into H / A / D possessions.

Viterbi over three labels. Each label is a chain of sub-states, so a stretch of it lasts at least
`min_len` frames (no one-frame flicker). Switching costs a penalty: H<->A costs `lambda_ha`, cheaper
(`lambda_ha_touch`) within a few frames of a detected touch; entering or leaving D costs
`lambda_d`. Frames with no evidence (ball missing) score the same for every label, so the label
holds through them.
Every value is in params.yaml under `decode`. Owner: person 3.
"""

import numpy as np
from numba import njit

from possession import rules
from possession.ballfilter import BallTrack

LABELS = np.array(["H", "A", "D"])


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -z)


def team_evidence(bt: BallTrack, params: dict) -> np.ndarray:
    """Log-odds per frame that Barcelona (H) rather than Athletic (A) has the ball.
    Touches count most; a ball close to one team and clear of the other counts a little."""
    p = params["decode"]
    llr = np.zeros(len(bt.x))
    sign = np.where(bt.touch_team == "H", 1.0, np.where(bt.touch_team == "A", -1.0, 0.0))
    llr += p["w_touch"] * sign
    known = ~np.isnan(bt.x)
    h_ctrl = known & (bt.d_H <= p["control_radius_m"]) & (bt.d_A > p["free_radius_m"])
    a_ctrl = known & (bt.d_A <= p["control_radius_m"]) & (bt.d_H > p["free_radius_m"])
    llr += p["w_control"] * (h_ctrl.astype(float) - a_ctrl.astype(float))
    return llr


def emissions(dead_logodds: np.ndarray, team_llr: np.ndarray, params: dict) -> np.ndarray:
    """(N, 3) log-scores for H, A, D, scaled by `temperature` (neighbouring frames are not
    independent, so raw log-likelihoods would be far too confident)."""
    live = log_sigmoid(-dead_logodds)
    e = np.column_stack([live + log_sigmoid(team_llr), live + log_sigmoid(-team_llr),
                         log_sigmoid(dead_logodds)])
    return e * params["decode"]["temperature"]


def viterbi(e: np.ndarray, min_len: tuple[int, int, int], switch_cost: np.ndarray) -> np.ndarray:
    """Best label path. e: (T, 3) scores; min_len: frames per label; switch_cost: (T, 3, 3)
    cost of moving from label i at t-1 to label j at t (diagonal ignored). Returns (T,) 0/1/2.

    Each label is a chain of min_len sub-states: a stretch enters at the first, must walk to the
    last, and only the last can stay or switch. The first stretch may start at its last sub-state.
    Ties go to the lower state index, so the result is deterministic."""
    if len(e) == 0:
        return np.zeros(0, dtype=np.int64)
    return _viterbi(np.ascontiguousarray(e, dtype=np.float64),
                    np.asarray(min_len, dtype=np.int64),
                    np.ascontiguousarray(switch_cost, dtype=np.float64))


@njit(cache=True)
def _viterbi(e, min_len, cost):
    T = e.shape[0]
    n = min_len.sum()
    first = np.zeros(3, np.int64)
    last = np.zeros(3, np.int64)
    cls = np.zeros(n, np.int64)
    k = 0
    for c in range(3):
        first[c] = k
        for _ in range(min_len[c]):
            cls[k] = c
            k += 1
        last[c] = k - 1

    score = np.full(n, -np.inf)
    for c in range(3):
        score[last[c]] = e[0, c]
    back = np.zeros((T, n), np.int32)
    new = np.empty(n)
    for t in range(1, T):
        for s in range(n):
            new[s] = -np.inf
            back[t, s] = s
        for c in range(3):
            for s in range(first[c] + 1, last[c] + 1):      # walk along the chain
                new[s] = score[s - 1]
                back[t, s] = s - 1
            if score[last[c]] > new[last[c]]:                # stay
                new[last[c]] = score[last[c]]
                back[t, last[c]] = last[c]
        for j in range(3):                                   # switch into label j
            best, bi = -np.inf, -1
            for i in range(3):
                if i != j:
                    v = score[last[i]] - cost[t, i, j]
                    if v > best:
                        best, bi = v, i
            if best > new[first[j]]:
                new[first[j]] = best
                back[t, first[j]] = last[bi]
        for s in range(n):
            score[s] = new[s] + e[t, cls[s]]

    path = np.zeros(T, np.int64)
    s, best = last[0], score[last[0]]
    for c in range(1, 3):
        if score[last[c]] > best:
            s, best = last[c], score[last[c]]
    for t in range(T - 1, -1, -1):
        path[t] = cls[s]
        s = back[t, s]
    return path


def switch_costs(bt: BallTrack, n: int, params: dict, switch_logodds=None) -> np.ndarray:
    p = params["decode"]
    cost = np.full((n, 3, 3), float(p["lambda_d"]))
    near_touch = np.convolve(bt.touch.astype(float), np.ones(2 * p["touch_window_frames"] + 1), "same") > 0
    ha = np.where(near_touch, p["lambda_ha_touch"], p["lambda_ha"]).astype(float)
    if switch_logodds is not None and p.get("w_switch", 0):
        # change detector: a likely switch makes H<->A cheaper, an unlikely one dearer
        ha = np.clip(ha - p["w_switch"] * (switch_logodds - p["switch_center"]), 0.0, p["switch_max_cost"])
    cost[:, 0, 1] = ha
    cost[:, 1, 0] = ha
    # play restarts with a kick: leaving D is cheap only near a detected touch
    exit_d = np.where(near_touch, p["lambda_d_exit_touch"], p["lambda_d_exit"])
    cost[:, 2, 0] = exit_d
    cost[:, 2, 1] = exit_d
    return cost


def add_set_piece_hints(dead_logodds, team_llr, hints, params):
    """A set piece tag: the ball is dead around its start, and the named team (if any) restarts."""
    p = params["decode"]
    dead, llr = dead_logodds.copy(), team_llr.copy()
    n = len(dead)
    for f, team in zip(hints["start_frame"], hints["team"]):
        dead[max(0, f - p["tag_dead_before"]) : min(n, f + p["tag_dead_after"])] += p["w_tag_dead"]
        if team:
            sign = 1.0 if team == "H" else -1.0
            llr[f : min(n, f + p["tag_team_frames"])] += sign * p["w_tag_team"]
    return dead, llr


def refine_boundaries(lab: np.ndarray, bt: BallTrack, a: int, b: int, params: dict, pitch) -> np.ndarray:
    """Stage 6: move each decoded boundary in [a, b] onto the event that causes it.

    - switch to team q: the first touch by q that follows a touch by the other team, nearest the
      decoded switch within +-snap_s
    - stop (into D): the frame the ball is first seen beyond a line, within -snap_s .. +snap_stop_after_s
    - restart (out of D into q): the nearest touch, by q if there is one, within +-snap_s
    Boundaries with no such event nearby stay where the decoder put them. A move never crosses
    the neighbouring boundaries."""
    from possession.deadball import out_distance

    p = params["decode"]
    w, w_after = int(p["snap_s"] * 25), int(p["snap_stop_after_s"] * 25)
    out = lab.copy()
    seg = lab[a : b + 1]
    edges = list(np.flatnonzero(seg[1:] != seg[:-1]) + 1 + a)
    if not edges:
        return out
    touch_idx = np.flatnonzero(bt.touch[a : b + 1]) + a
    touch_team = bt.touch_team[touch_idx]
    prev_team = np.r_[[""], touch_team[:-1]]
    seen_out = np.flatnonzero(out_distance(np.where(bt.observed, bt.x, np.nan),
                                           np.where(bt.observed, bt.y, np.nan), pitch)[a : b + 1] > 0.5) + a
    bounds = [a] + edges + [b + 1]
    for k in range(1, len(bounds) - 1):
        t, lo, hi = bounds[k], bounds[k - 1] + 1, bounds[k + 1] - 1
        before, after = lab[t - 1], lab[t]
        cand = np.array([], dtype=np.int64)
        if before in "HA" and after in "HA":
            m = (touch_team == after) & (prev_team != after) & (prev_team != "")
            cand = touch_idx[m & (abs(touch_idx - t) <= w)]
        elif after == "D":
            cand = seen_out[(seen_out >= t - w) & (seen_out <= t + w_after)]
            cand = cand[:1] if len(cand) else cand
        elif before == "D":
            near = abs(touch_idx - t) <= w
            own = near & (touch_team == after)
            cand = touch_idx[own] if own.any() else touch_idx[near]
        cand = cand[(cand >= lo) & (cand <= hi)]
        if len(cand):
            n = int(cand[np.argmin(abs(cand - t))])
            if n < t:
                out[n:t] = after
            elif n > t:
                out[t:n] = before
            bounds[k] = n
    return out


def missing_ball_adjust(dead_logodds: np.ndarray, bt: BallTrack, p: dict) -> np.ndarray:
    """Bench tests (both 0 = unchanged). gap_pool: inside each run of frames where the raw ball is
    missing, blend each frame's dead log-odds with that run's mean (judge the gap as a whole).
    dead_bias_miss: added to the dead log-odds only where the raw ball is missing."""
    pool, bias = p.get("gap_pool", 0.0), p.get("dead_bias_miss", 0.0)
    if not pool and not bias:
        return dead_logodds
    miss = ~np.asarray(bt.observed, bool)
    out = np.asarray(dead_logodds, float).copy()
    if pool:
        e = np.flatnonzero(np.diff(np.r_[0, miss.astype(np.int8), 0]))
        for s, t in zip(e[::2], e[1::2]):
            out[s:t] = (1 - pool) * out[s:t] + pool * np.nanmean(out[s:t])
    return out + bias * miss


def decode(bt: BallTrack, dead_logodds: np.ndarray, halves, params: dict, hints=None,
           model_team_llr=None, pitch=(105.3, 68.0), switch_logodds=None, chair_logp=None,
           ctx=None, tcn_logp=None) -> np.ndarray:
    """Labels (N,) "H"/"A"/"D" inside each half, "X" outside. model_team_llr (model B's H-vs-A
    log-odds) is added to the touch evidence when given."""
    p = params["decode"]
    dead_logodds = missing_ball_adjust(dead_logodds, bt, p)
    team_llr = team_evidence(bt, params)
    if model_team_llr is not None:
        team_llr = team_llr + params["model"]["w_team"] * np.nan_to_num(model_team_llr)
    if p.get("gap_anchor", 0):
        team_llr = gap_anchor(team_llr, bt, p)
    if hints is not None and len(hints):
        dead_logodds, team_llr = add_set_piece_hints(dead_logodds, team_llr, hints, params)
    e = emissions(dead_logodds, team_llr, params)
    if chair_logp is not None:
        # the council's chair has already weighed every expert: use its verdict as the evidence
        e = chair_logp * p["chair_temperature"]
    if hints is not None and len(hints) and p["dead_before_kickoff"]:
        # the ball is dead from the start of a half until its kick-off (Smart Tagging "Kick Off
        # (Start)"), so only D is allowed there. A fixed rule on an inference input, no frame numbers.
        starts = hints.loc[hints["type"].str.contains("Kick Off \\(Start\\)", case=False, na=False), "start_frame"]
        for _, a, b in halves:
            inside = starts[(starts > a) & (starts <= b)]
            if len(inside):
                e[a : int(inside.min()), :2] = -1e9
    if tcn_logp is not None:
        # network sequence model (src/possession/tcn.py): blend its H/A/D log-probabilities in,
        # on every frame or only where the ball is not seen (decode.tcn_hidden_only)
        w = np.full(len(e), float(p["tcn_weight"]))
        if p.get("tcn_hidden_only", False):
            w[np.asarray(bt.observed, bool)] = 0.0
        e = (1 - w)[:, None] * e + w[:, None] * p["temperature"] * tcn_logp
    cost = switch_costs(bt, len(e), params, switch_logodds)
    min_len = (p["min_possession_frames"], p["min_possession_frames"], p["min_dead_frames"])
    out = np.full(len(e), "X", dtype="<U1")
    for _, a, b in halves:
        out[a : b + 1] = LABELS[viterbi(e[a : b + 1], min_len, cost[a : b + 1])]
        if p["refine"]:
            out = refine_boundaries(out, bt, a, b, params, pitch)
        if p.get("peak_insert", 0) and switch_logodds is not None:
            out[a : b + 1] = insert_at_peaks(out[a : b + 1], team_llr[a : b + 1], switch_logodds[a : b + 1], p)
        if p.get("merge_sandwich_s", 0):
            out[a : b + 1] = merge_sandwiches(out[a : b + 1], int(round(p["merge_sandwich_s"] * 25)))
        if p.get("last_touch_s", 0):
            out[a : b + 1] = last_touch_before_dead(out[a : b + 1], bt.touch_team[a : b + 1], p)
        rules.apply(out, a, b, bt, ctx, p)      # bench-test rules (src/possession/rules.py), all off by default
    return out


def insert_at_peaks(lab: np.ndarray, team_llr: np.ndarray, sw: np.ndarray, p: dict) -> np.ndarray:
    """Off unless decode.peak_insert > 0 (ASRF-style boundary insertion). Peaks of the change
    detector's log-odds above peak_insert (local maxima, one per +-0.5 s) inside one of our H/A
    runs cut it. Each piece between cuts that is at least peak_min_s long and whose mean team
    evidence favours the other team by more than peak_min_llr takes that team."""
    out = lab.copy()
    sw = np.nan_to_num(np.asarray(sw, float), nan=-99.0)
    n, w = len(sw), 12
    pad = np.r_[np.full(w, -99.0), sw, np.full(w, -99.0)]
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)
    peaks = np.flatnonzero((sw >= win.max(axis=1)) & (sw > p["peak_insert"]))
    min_len, thr = int(round(p.get("peak_min_s", 1.5) * 25)), p.get("peak_min_llr", 1.0)
    e = np.flatnonzero(np.r_[True, out[1:] != out[:-1], True])
    for s, t in zip(e[:-1], e[1:]):
        if out[s] not in ("H", "A"):
            continue
        cuts = [s] + [int(c) for c in peaks if s < c < t] + [t]
        if len(cuts) == 2:
            continue
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            if c1 - c0 < min_len:
                continue
            m = np.nanmean(team_llr[c0:c1])
            if out[s] == "H" and m < -thr:
                out[c0:c1] = "A"
            elif out[s] == "A" and m > thr:
                out[c0:c1] = "H"
    return out


def gap_anchor(team_llr: np.ndarray, bt: BallTrack, p: dict) -> np.ndarray:
    """Off unless decode.gap_anchor > 0. For each run of frames where the raw ball is missing
    (at least gap_anchor_min_s long), read the team evidence over gap_anchor_edge_s just before the
    ball vanished and just after it came back. If both ends favour the same team (each beyond
    gap_anchor_min_llr), add gap_anchor to that team's evidence across the gap: the same team most
    likely kept the ball. If the ends disagree, nothing changes and the decoder places the switch."""
    out = np.asarray(team_llr, float).copy()
    miss = ~np.asarray(bt.observed, bool)
    k = max(1, int(round(p.get("gap_anchor_edge_s", 0.4) * 25)))
    lo_len = int(round(p.get("gap_anchor_min_s", 1.0) * 25))
    thr = p.get("gap_anchor_min_llr", 0.5)
    e = np.flatnonzero(np.diff(np.r_[0, miss.astype(np.int8), 0]))
    n = len(out)
    for s, t in zip(e[::2], e[1::2]):
        if t - s < lo_len or s < k or t + k > n:
            continue
        a, b = np.nanmean(out[s - k : s]), np.nanmean(out[t : t + k])
        if a > thr and b > thr:
            out[s:t] += p["gap_anchor"]
        elif a < -thr and b < -thr:
            out[s:t] -= p["gap_anchor"]
    return out


def merge_sandwiches(lab: np.ndarray, max_len: int) -> np.ndarray:
    """Off unless decode.merge_sandwich_s > 0. An H run shorter than max_len frames with A on both
    sides (or the reverse) becomes the surrounding team: the same rule the outside training labels
    use (idsse_bench.merge_blip_s), so we never output a possession the labels cannot contain."""
    out = lab.copy()
    e = np.flatnonzero(np.r_[True, out[1:] != out[:-1], True])
    for i in range(1, len(e) - 2):
        s, t = e[i], e[i + 1]
        if t - s < max_len and out[s] in ("H", "A") and out[s - 1] == out[t] and out[t] in ("H", "A") and out[t] != out[s]:
            out[s:t] = out[t]
    return out


def last_touch_before_dead(lab: np.ndarray, touch_team: np.ndarray, p: dict) -> np.ndarray:
    """Off unless decode.last_touch_s > 0. At each start of a dead spell, look back up to
    last_touch_s seconds for the last detected touch. If the other team made it (a clearance or
    block that goes out), that team owns the ball from its touch to the stoppage, as event-based
    possession labels do. last_touch_extend_s: also let a touch just after our stoppage start
    (within that many seconds) pull the stoppage start back to the touch."""
    out = lab.copy()
    look = int(round(p["last_touch_s"] * 25))
    ext = int(round(p.get("last_touch_extend_s", 0) * 25))
    starts = np.flatnonzero((out[1:] == "D") & (out[:-1] != "D")) + 1
    for d0 in starts:
        lo, hi = max(0, d0 - look), min(len(out), d0 + ext)
        idx = np.flatnonzero(np.isin(touch_team[lo:hi], ["H", "A"]))
        if not len(idx):
            continue
        f = lo + idx[-1]
        team = touch_team[f]
        if f < d0:
            if out[d0 - 1] != team and np.all(out[f:d0] != "D"):
                out[f:d0] = team
        elif ext:
            out[d0 : f + 1] = team          # touch after our stoppage start: stoppage begins later
    return out
