"""Probe: does 'last team in clear control of the ball' predict the team better than v19 where
v19 is weak (short possessions, ball seen)? Bench truth, faithful damage, aggregates only."""
import pickle
import numpy as np, pandas as pd
from possession import config, decode, model, selection
base = config.load_params(); ids = model.external_matches(base)
P = selection.version_params(base, selection.VERSIONS["v6"])
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
res = []
for m in ids:
    pr = model.prepare_idsse(m, base, variant=base["idsse_bench"]["train_degrade"])
    bt, gt = pr["bt"], pr["gt"]
    e = pickle.load(open(f"outputs/logs/experts_mot/{m}.pkl", "rb"))
    lab = decode.decode(bt, P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                        model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"])
    s0 = np.flatnonzero(np.r_[True, gt[1:] != gt[:-1]]); s1 = np.r_[s0[1:], len(gt)]
    short = np.zeros(len(gt), bool)
    for a, b in zip(s0, s1):
        if gt[a] in "HA" and b - a < 100: short[a:b] = True
    both = np.isin(gt, ["H", "A"]) & np.isin(lab, ["H", "A"])
    seen = bt.observed
    for r, g in ((1.0, 1.0), (1.5, 1.0), (2.0, 1.5)):
        dh, da = bt.d_H, bt.d_A
        ctrl = np.where((dh <= r) & (da >= dh + g), "H", np.where((da <= r) & (dh >= da + g), "A", ""))
        st = pd.Series(np.where(ctrl == "", None, ctrl)).ffill().fillna("H").to_numpy()
        for name, mask in (("all live", both), ("seen", both & seen), ("short+seen", both & seen & short), ("short", both & short)):
            res.append(dict(match=m, r=r, g=g, where=name, n=mask.sum(), v19=np.mean(lab[mask] == gt[mask]),
                            last_ctrl=np.mean(st[mask] == gt[mask]),
                            agree_wrong_v19=np.mean((st[mask] == gt[mask])[lab[mask] != gt[mask]])))
df = pd.DataFrame(res)
w = df.groupby(["r", "g", "where"]).apply(lambda x: pd.Series({k: np.average(x[k], weights=x.n) for k in ("v19", "last_ctrl", "agree_wrong_v19")}))
print(w.round(3).to_string())
