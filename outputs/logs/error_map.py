"""Error map on the 10 held-out outside matches (v17): where do wrong frames happen?
Reads outputs/logs/holdout_preds/{mid}.npz (truth + ours) and the faithful-degraded tracking.
Outside matches only; nothing from our match."""
import numpy as np, pandas as pd
from possession import config, model

FPS = 25
base = config.load_params()
dp = model.degrade_params(base, "faithful")


def runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return e[::2], e[1::2]


def ffill(a):
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def frame_table(mid):
    z = np.load(f"outputs/logs/{__import__('os').environ.get('PREDS', 'holdout_preds')}/{mid}.npz", allow_pickle=True)
    gt, lab = z["gt"], z["lab"]
    m = model.load_external(mid, dp)
    t = m.tracking
    L, W = m.pitch
    n = len(gt)
    bx, by = ffill(t.ball[:, 0].copy()), ffill(t.ball[:, 1].copy())
    seen = ~np.isnan(t.ball[:, 0])
    k = FPS
    bsp = np.full(n, np.nan); bsp[k:] = np.hypot(bx[k:] - bx[:-k], by[k:] - by[:-k])  # m per second
    psp = np.nanmean(t.ps, axis=1) if t.ps is not None else np.full(n, np.nan)
    # which side each team defends, per half: mean x of team H early in the half
    own_goal_x = np.full((n, 2), np.nan)  # col0 = H's goal x, col1 = A's goal x
    for _, s, e in m.halves:
        hx = np.nanmean(t.px[s:s + 10 * FPS, t.team == "H"])
        gh = 0.0 if hx < L / 2 else L
        own_goal_x[s:e + 1] = [gh, L - gh]
    df = pd.DataFrame(dict(match=mid, truth=gt, ours=lab, seen=seen, ball_speed=bsp, player_speed=psp,
                           goal_dist=np.minimum(bx, L - bx), touch_dist=np.minimum(by, W - by)))
    pos = np.where(gt == "H", own_goal_x[:, 0], np.where(gt == "A", own_goal_x[:, 1], np.nan))
    df["from_own_goal"] = np.abs(bx - pos)
    # time since / until truth live spell start / end
    live = np.isin(gt, ["H", "A"])
    since, until = np.full(n, np.nan), np.full(n, np.nan)
    for s, e in zip(*runs(live)):
        since[s:e] = np.arange(e - s) / FPS
        until[s:e] = np.arange(e - s)[::-1] / FPS
    df["since_start"], df["until_end"] = since, until
    return df[gt != "X"]


def spells(mid, gt, lab):
    """Restart timing and false stoppages inside truth live spells."""
    tl, ol = np.isin(gt, ["H", "A"]), np.isin(lab, ["H", "A"])
    ts, te = runs(tl); os_, oe = runs(ol)
    late_start = [ (os_[np.argmin(np.abs(os_ - s))] - s) / FPS for s in ts if len(os_) and np.min(np.abs(os_ - s)) < 10 * FPS]
    late_end = [ (oe[np.argmin(np.abs(oe - e))] - e) / FPS for e in te if len(oe) and np.min(np.abs(oe - e)) < 10 * FPS]
    fs = []  # our dead runs fully inside a truth live spell
    ds, de = runs(~ol)
    for s, e in zip(ts, te):
        k = (ds > s) & (de < e)
        fs += list((de[k] - ds[k]) / FPS)
    # possessions
    tp = sum(len(runs(gt == c)[0]) for c in "HA"); op = sum(len(runs(lab == c)[0]) for c in "HA")
    return dict(match=mid, truth_live_spells=len(ts), our_live_spells=len(os_),
                start_offset_med_s=np.median(late_start), end_offset_med_s=np.median(late_end),
                false_stoppages=len(fs), false_stop_med_s=np.median(fs) if fs else 0,
                false_stop_frames_share=sum(fs) * FPS / len(gt), truth_poss=tp, our_poss=op)


def table(df, col, bins, labels):
    live = df[df.truth.isin(["H", "A"])]
    cat = pd.cut(live[col], bins, labels=labels, right=False)
    ld = live.ours == "D"
    both = live.ours != "D"
    wrong_team = both & (live.ours != live.truth)
    out = pd.DataFrame({
        "share of live frames": cat.value_counts(normalize=True),
        "called dead (rate)": ld.groupby(cat, observed=False).mean(),
        "share of all live->dead errors": ld.groupby(cat, observed=False).sum() / ld.sum(),
        "wrong team (rate, both live)": wrong_team.groupby(cat, observed=False).sum() / both.groupby(cat, observed=False).sum(),
    }).reindex(labels)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None); pd.set_option("display.precision", 3)
    ids = model.external_matches(base)
    frames = pd.concat([frame_table(m) for m in ids])
    frames["group"] = np.where(frames.match.str.startswith("MET"), "Metrica", "DFL")
    sp = pd.DataFrame([spells(m, *[g.to_numpy() for g in (frames[frames.match == m].truth, frames[frames.match == m].ours)]) for m in ids]).set_index("match")
    print("== spell level (offset >0 = we are late)\n", sp.round(2))
    for grp, d in [("ALL", frames)] + list(frames.groupby("group")):
        live = d[d.truth.isin(["H", "A"])]
        print(f"\n######## {grp}: live frames called dead {np.mean(live.ours=='D'):.3f}; dead frames called live {np.mean(d[d.truth=='D'].ours!='D'):.3f}")
        print("\n-- time since the truth live spell started (s)")
        print(table(d, "since_start", [0, 1, 2, 4, 8, 15, 1e9], ["0-1", "1-2", "2-4", "4-8", "8-15", "15+"]))
        print("\n-- time until the truth live spell ends (s)")
        print(table(d, "until_end", [0, 1, 2, 4, 8, 1e9], ["0-1", "1-2", "2-4", "4-8", "8+"]))
        print("\n-- ball seen this frame")
        d2 = d.assign(seen_i=d.seen.astype(float))
        print(table(d2, "seen_i", [0, 0.5, 2], ["missing", "seen"]))
        print("\n-- ball moved in the last 1 s (m)")
        print(table(d, "ball_speed", [0, 1, 3, 8, 15, 1e9], ["<1", "1-3", "3-8", "8-15", "15+"]))
        print("\n-- mean player speed (m/s)")
        print(table(d, "player_speed", [0, 1, 1.5, 2, 3, 1e9], ["<1", "1-1.5", "1.5-2", "2-3", "3+"]))
        print("\n-- ball distance from the possessing team's own goal line (m)")
        print(table(d, "from_own_goal", [0, 17, 35, 70, 88, 1e9], ["own box", "own third", "middle", "att third", "att box"]))
        print("\n-- ball distance from touchline (m)")
        print(table(d, "touch_dist", [-1e9, 2, 5, 15, 1e9], ["<2", "2-5", "5-15", "15+"]))
