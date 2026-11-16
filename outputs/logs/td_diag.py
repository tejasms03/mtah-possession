"""Turnover-then-stoppage possessions (truth: other team > X > dead), outside matches, v18 hold-out."""
import numpy as np, pandas as pd
from possession import config, model
base = config.load_params()
dp = model.degrade_params(base, "faithful")
rows = []
for mid in model.external_matches(base):
    z = np.load(f"outputs/logs/holdout_preds_v18/{mid}.npz", allow_pickle=True)
    gt, lab = z["gt"], z["lab"]
    ball = model.load_external(mid, dp).tracking.ball[:, 0]
    e = np.flatnonzero(np.r_[True, gt[1:] != gt[:-1], True])
    runs = [(e[i], e[i + 1], gt[e[i]]) for i in range(len(e) - 1)]
    for k in range(1, len(runs) - 1):
        s, t, g = runs[k]
        if g in ("H", "A") and runs[k - 1][2] in ("H", "A") and runs[k - 1][2] != g and runs[k + 1][2] == "D":
            prev = runs[k - 1][2]
            # where does our stoppage start relative to the truth stoppage start t?
            d = np.flatnonzero(lab[max(0, t - 250):t + 250] == "D")
            ours_d0 = (d[0] + max(0, t - 250) - t) / 25 if len(d) else np.nan
            rows.append(dict(match=mid, dur=(t - s) / 25, same=np.mean(lab[s:t] == g), prev=np.mean(lab[s:t] == prev),
                             dead=np.mean(lab[s:t] == "D"), miss=np.mean(np.isnan(ball[s:t])), d0=ours_d0,
                             prev_tail=np.mean(lab[max(s, t - 25):t] == prev)))
d = pd.DataFrame(rows); d["found"] = d.same >= 0.5
print(f"turnover-then-stoppage possessions: {len(d)}, found {d.found.mean():.1%}")
m = d[~d.found]
print(f"missed {len(m)}: median length {m.dur.median():.1f} s (found ones {d[d.found].dur.median():.1f} s)")
print(f"  labelled mostly the previous team: {np.mean(m.prev > 0.5):.1%}; mostly dead: {np.mean(m.dead > 0.5):.1%}")
print(f"  ball missing during the possession: missed {m.miss.mean():.1%} vs found {d[d.found].miss.mean():.1%}")
print(f"  our stoppage starts (s, + = after truth): missed median {m.d0.median():+.2f}, found {d[d.found].d0.median():+.2f}")
print("\nfound share by length:")
print(d.groupby(pd.cut(d.dur, [0, 1, 2, 4, 8, 1e9]), observed=True).found.agg(["size", "mean"]).round(3).to_string())
print("\nfound share by ball-missing share:")
print(d.groupby(pd.cut(d.miss, [-0.01, 0.1, 0.5, 0.9, 1.0]), observed=True).found.agg(["size", "mean"]).round(3).to_string())
