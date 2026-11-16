"""Direct H<->A switches: our switch time minus the nearest truth switch (same direction), v18 hold-out."""
import numpy as np
from possession import config, model
lags, unmatched, n_truth = [], 0, 0
per = {}
for mid in model.external_matches(config.load_params()):
    z = np.load(f"outputs/logs/holdout_preds_v18/{mid}.npz", allow_pickle=True)
    gt, lab = z["gt"], z["lab"]
    def sw(a):
        i = np.flatnonzero(np.isin(a[1:], ["H", "A"]) & np.isin(a[:-1], ["H", "A"]) & (a[1:] != a[:-1])) + 1
        return i, a[i]
    ti, tt = sw(gt); oi, ot = sw(lab)
    n_truth += len(ti); l = []
    for i, t in zip(ti, tt):
        c = oi[(ot == t) & (np.abs(oi - i) <= 75)]
        if len(c): l.append((c[np.argmin(np.abs(c - i))] - i) / 25)
        else: unmatched += 1
    per[mid] = np.median(l); lags += l
lags = np.array(lags)
print(f"truth direct switches {n_truth}; matched within 3 s {len(lags)} ({len(lags)/n_truth:.0%})")
print(f"our switch minus truth (s): median {np.median(lags):+.2f}, mean {lags.mean():+.2f}, p25 {np.percentile(lags,25):+.2f}, p75 {np.percentile(lags,75):+.2f}")
print("share late >0.2 s: %.0f%%, early < -0.2 s: %.0f%%" % (100*np.mean(lags > 0.2), 100*np.mean(lags < -0.2)))
print("per match median:", {k[-4:]: round(v, 2) for k, v in per.items()})
