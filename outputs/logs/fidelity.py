"""Label-free fidelity audit: our match's tracking vs each degraded outside match, same measures.

Only tracking is used (no labels, no tags) so the comparison can calibrate the damage like the
existing idsse_degrade calibration. Prints one row per match and a gap column per measure.
"""
import sys

import numpy as np

from possession import load, model
from possession.config import load_params

FPS = 25


def runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return e[::2], e[1::2]


def measures(ball, px, py, L, W, live_mask):
    """live_mask: frames in a half (we have no labels; 'graded' = inside a half)."""
    b, x, y = ball[live_mask], px[live_mask], py[live_mask]
    out = {}
    miss = np.isnan(b[:, 0])
    out["ball missing"] = miss.mean()
    s, t = runs(miss)
    d = (t - s) / FPS
    out["ball gaps/90min"] = len(s) / (len(b) / FPS / 5400)
    out["gap median s"] = np.median(d) if len(d) else 0
    out["gap p90 s"] = np.percentile(d, 90) if len(d) else 0
    out["gaps <1 s share"] = np.mean(d < 1) if len(d) else 0
    k = (s > 0) & (d >= 1)
    xs, ys = b[s[k] - 1, 0], b[s[k] - 1, 1]
    edge = np.minimum.reduce([xs, L - xs, ys, W - ys])
    out["gap starts >5 m in"] = np.mean(edge > 5)
    jump = np.hypot(b[s[k] - 1, 0] - b[np.minimum(t[k], len(b) - 1), 0], b[s[k] - 1, 1] - b[np.minimum(t[k], len(b) - 1), 1])
    out["ball moved over gap m"] = np.nanmedian(jump)
    v = np.hypot(np.diff(b[:, 0]), np.diff(b[:, 1])) * FPS
    out["ball speed median"] = np.nanmedian(v)
    out["ball speed p95"] = np.nanpercentile(v, 95)
    j2 = np.hypot(np.diff(b[:, 0], 2), np.diff(b[:, 1], 2))
    out["ball jitter m"] = np.nanmedian(j2)
    out["ball still share"] = np.nanmean(v < 0.5)
    out["ball off pitch"] = np.nanmean((b[:, 0] < 0) | (b[:, 0] > L) | (b[:, 1] < 0) | (b[:, 1] > W))
    vis = ~np.isnan(x)
    nv = vis.sum(1)
    out["players visible"] = nv.mean()
    out["frames >22"] = np.mean(nv > 22)
    out["frames <18"] = np.mean(nv < 18)
    rl = [(t2 - s2) / FPS for j in range(vis.shape[1]) for s2, t2 in zip(*runs(vis[:, j]))]
    out["track run median s"] = np.median(rl)
    out["tracks/90min"] = len(rl) / (len(b) / FPS / 5400)
    pv = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)) * FPS
    out["player speed median"] = np.nanmedian(pv)
    out["player speed p95"] = np.nanpercentile(pv, 95)
    out["player jitter m"] = np.nanmedian(np.hypot(np.diff(x, 2, axis=0), np.diff(y, 2, axis=0)))
    # nearest-other-player distance (duplicates show as very small)
    sub = np.arange(0, len(x), 50)
    dmin = []
    for i in sub:
        ok = vis[i]
        if ok.sum() < 2:
            continue
        P = np.c_[x[i, ok], y[i, ok]]
        D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1)); np.fill_diagonal(D, np.inf)
        dmin.append(D.min(1))
    dmin = np.concatenate(dmin)
    out["pairs <1 m share"] = np.mean(dmin < 1)
    bb = ~np.isnan(b[sub, 0])
    nd = []
    for i in sub[bb]:
        ok = vis[i]
        if ok.any():
            nd.append(np.nanmin(np.hypot(x[i, ok] - b[i, 0], y[i, ok] - b[i, 1])))
    out["nearest player to ball m"] = np.median(nd)
    out["x spread players"] = np.nanmedian(np.nanmax(x, 1) - np.nanmin(x, 1))
    return out


def ours():
    t = load.load_atd()
    live = ~(np.isnan(t.ball[:, 0]) & np.all(np.isnan(t.px), 1))  # frames with any data
    return measures(t.ball, t.px, t.py, 105.3, 68.0, live)


def outside(mid, hide=0.0):
    m = model.load_external(mid, hide)
    live = np.zeros(len(m.gt), bool)
    for _, a, b in m.halves:
        live[a : b + 1] = True
    return measures(m.tracking.ball, m.tracking.px, m.tracking.py, *m.pitch, live)


if __name__ == "__main__":
    hide = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    rows = {"OURS": ours()}
    ids = model.external_matches(load_params())
    for mid in ids:
        rows[mid] = outside(mid, hide if mid.startswith("MET") else 0.0)
        print(f"  measured {mid}", flush=True)
    keys = list(rows["OURS"])
    names = list(rows)
    print(f"\n{'measure':26}" + "".join(f"{n[:7]:>9}" for n in names))
    for k in keys:
        print(f"{k:26}" + "".join(f"{rows[n][k]:9.3f}" for n in names))
