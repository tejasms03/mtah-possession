"""Headroom on the 10-match hold-out (v19): fix one error type perfectly with the truth, rescore."""
import pickle
import numpy as np, pandas as pd
from possession import config, decode, model, selection
from possession.external.bench import _vendor
pm = _vendor("possession_metrics")
base = config.load_params(); ids = model.external_matches(base)
P = selection.version_params(base, selection.VERSIONS["v6"])
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
sc = lambda gt, lab: pm.evaluate(pm.label_codes(gt), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"]  # noqa
def ffill_team(lab):
    out = lab.copy(); last = "H"
    for i, x in enumerate(out):
        if x in "HA": last = x
        elif x == "D": out[i] = last
    return out
rows = []
for m in ids:
    pr = model.prepare_idsse(m, base, variant=base["idsse_bench"]["train_degrade"])
    e = pickle.load(open(f"outputs/logs/experts_mot/{m}.pkl", "rb"))
    lab = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                        model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"])
    gt = pr["gt"]; g = gt != "X"; live = np.isin(gt, ["H", "A"])
    # A: perfect dead/live (truth D where dead; where truth live and we said D, our nearest earlier team)
    a = ffill_team(lab); a[gt == "D"] = "D"; a[~g] = "X"
    # B: perfect team on frames both call live
    b = lab.copy(); both = live & np.isin(lab, ["H", "A"]); b[both] = gt[both]
    # C: perfect dead/live + perfect team only where the ball is seen
    c = a.copy(); seen = pr["bt"].observed & live; c[seen] = gt[seen]
    # D: perfect everything where the ball is seen (dead/live and team), ours where missing
    d = lab.copy(); s = pr["bt"].observed & g; d[s] = gt[s]
    # E: perfect team on live frames + perfect dead/live = all
    obs = pr["bt"].observed
    bs = lab.copy(); k = both & obs; bs[k] = gt[k]
    bm = lab.copy(); k = both & ~obs; bm[k] = gt[k]
    # team perfect only inside short true possessions (< 4 s)
    e2 = np.flatnonzero(np.r_[True, gt[1:] != gt[:-1]]); ends = np.r_[e2[1:], len(gt)]
    short = np.zeros(len(gt), bool)
    for s0, t0 in zip(e2, ends):
        if gt[s0] in "HA" and t0 - s0 < 100: short[s0:t0] = True
    bsh = lab.copy(); k = both & short; bsh[k] = gt[k]
    rows.append(dict(match=m[-4:], v19=sc(gt, lab), team_seen=sc(gt, bs), team_missing=sc(gt, bm),
                     team_short_poss=sc(gt, bsh), share_short=short[live].mean()))
    print(rows[-1], flush=True)
t = pd.DataFrame(rows).set_index("match"); print(t.round(3).to_string()); print("MEAN", t.mean().round(4).to_dict())
