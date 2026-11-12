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


def switch_costs(bt: BallTrack, n: int, params: dict) -> np.ndarray:
    p = params["decode"]
    cost = np.full((n, 3, 3), float(p["lambda_d"]))
    near_touch = np.convolve(bt.touch.astype(float), np.ones(2 * p["touch_window_frames"] + 1), "same") > 0
    ha = np.where(near_touch, p["lambda_ha_touch"], p["lambda_ha"])
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


def decode(bt: BallTrack, dead_logodds: np.ndarray, halves, params: dict, hints=None,
           model_team_llr=None, pitch=(105.3, 68.0)) -> np.ndarray:
    """Labels (N,) "H"/"A"/"D" inside each half, "X" outside. model_team_llr (model B's H-vs-A
    log-odds) is added to the touch evidence when given."""
    p = params["decode"]
    team_llr = team_evidence(bt, params)
    if model_team_llr is not None:
        team_llr = team_llr + params["model"]["w_team"] * np.nan_to_num(model_team_llr)
    if hints is not None and len(hints):
        dead_logodds, team_llr = add_set_piece_hints(dead_logodds, team_llr, hints, params)
    e = emissions(dead_logodds, team_llr, params)
    cost = switch_costs(bt, len(e), params)
    min_len = (p["min_possession_frames"], p["min_possession_frames"], p["min_dead_frames"])
    out = np.full(len(e), "X", dtype="<U1")
    for _, a, b in halves:
        out[a : b + 1] = LABELS[viterbi(e[a : b + 1], min_len, cost[a : b + 1])]
        if p["refine"]:
            out = refine_boundaries(out, bt, a, b, params, pitch)
    return out
