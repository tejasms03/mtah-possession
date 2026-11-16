"""Which truth possessions does v18 miss? Outside matches only (holdout_preds_v18).
A truth possession (run of H or A) counts as found if our labels give the same team on >= 50% of its frames."""
import numpy as np, pandas as pd
from possession import config, model
rows = []
for mid in model.external_matches(config.load_params()):
    z = np.load(f"outputs/logs/holdout_preds_v18/{mid}.npz", allow_pickle=True)
    gt, lab = z["gt"], z["lab"]
    e = np.flatnonzero(np.r_[True, gt[1:] != gt[:-1], True])
    runs = [(e[i], e[i + 1], gt[e[i]]) for i in range(len(e) - 1)]
    for k, (s, t, g) in enumerate(runs):
        if g not in ("H", "A"):
            continue
        prev = runs[k - 1][2] if k else "X"
        nxt = runs[k + 1][2] if k + 1 < len(runs) else "X"
        same = np.mean(lab[s:t] == g)
        dead = np.mean(lab[s:t] == "D")
        rows.append(dict(match=mid, dur=(t - s) / 25, found=same >= 0.5, same=same, ours_dead=dead,
                         ctx=("D" if prev == "D" else "T") + ">" + ("D" if nxt == "D" else "T")))
d = pd.DataFrame(rows)
d["len"] = pd.cut(d.dur, [0, 1, 2, 4, 8, 15, 1e9], labels=["<1s", "1-2s", "2-4s", "4-8s", "8-15s", "15s+"])
print(f"truth possessions {len(d)}, found {d.found.mean():.1%}")
g = d.groupby("len", observed=True).agg(n=("found", "size"), found=("found", "mean"))
g["share_of_missed"] = d[~d.found].groupby("len", observed=True).size() / (~d.found).sum()
print("\nby length (T = bordered by the other team, D = by dead ball)\n" + g.round(3).to_string())
c = d.groupby("ctx").agg(n=("found", "size"), found=("found", "mean"))
c["share_of_missed"] = d[~d.found].groupby("ctx").size() / (~d.found).sum()
print("\nby what comes before > after\n" + c.round(3).to_string())
m = d[~d.found]
print(f"\nmissed possessions: we called them mostly dead in {np.mean(m.ours_dead > 0.5):.1%}, "
      f"mostly the other team in {np.mean((1 - m.same - m.ours_dead) > 0.5):.1%}, mixed in the rest")
print("\nmissed <2 s that sit between the same other team on both sides (A-H-A blips):",
      int(((m.dur < 2) & (m.ctx == "T>T")).sum()))
