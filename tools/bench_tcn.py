"""Bench the network sequence model (src/possession/tcn.py) on the 10 outside matches,
leave-one-match-out, against exact v21. For held-out match m the TCN weights come from a fold
trained on the other 9 (tools/train_tcn.py); its inputs are v21's out-of-fold first-stage
outputs (experts_to/). v21 = experts_seqT_to/ decoded with switch_center -1, w_team 1.

Mixing: decoder emissions e = (1 - a) * e_v21 + a * temperature * TCN log-probs (a on every
frame, or only where the ball is not seen); turnover head as switch signal:
switch = (1 - b) * switch_v21 + b * log(P(turnover) / P(none)).

Usage: python tools/bench_tcn.py EXPERTS_DIR FOLD_DIR [BOOT=name,name]
"""

import copy
import os
import pickle
import sys

import numpy as np
import pandas as pd

from possession import config, decode, model, selection, tcn
from possession.external.bench import _vendor

pm = _vendor("possession_metrics")
EXP, FOLDS = sys.argv[1], sys.argv[2]
base = config.load_params()
ids = model.external_matches(base)
_pb = copy.deepcopy(base)
_pb["features"].update(turnover_extra=False, restart_extra=False, carrier_extra=False)
preps = {m: model.prepare_idsse(m, _pb, variant=base["idsse_bench"]["train_degrade"]) for m in ids}
P = selection.version_params(base, selection.VERSIONS["v6"])
P["decode"]["switch_center"], P["model"]["w_team"] = -1.0, 1.0
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
E21 = {m: pickle.load(open(f"{EXP}/experts_seqT_to/{m}.pkl", "rb")) for m in ids}
E1 = {m: pickle.load(open(f"{EXP}/experts_to/{m}.pkl", "rb")) for m in ids}

TC = {}
for m in ids:
    f = f"{FOLDS}/tcn_out_{m}.npz"
    if not os.path.exists(f):
        W = tcn.load(f"{FOLDS}/fold_{m}.npz")
        pr, e = preps[m], E1[m]
        em, to = tcn.predict_match(W, e["team"], e["switch"], e["dead"], pr["bt"], pr["halves"], len(pr["gt"]))
        np.savez(f, em=em, to=to)
    z = np.load(f)
    TC[m] = (z["em"], z["to"])


def label(m, a=0.0, hidden_only=False, b=0.0, ref=False):
    pr, e = preps[m], E21[m]
    sw = e["switch"]
    Q, kw = P, {}
    if not ref:
        em, to = TC[m]
        Q = copy.deepcopy(P)
        Q["decode"].update(tcn_weight=a, tcn_hidden_only=hidden_only)
        kw["tcn_logp"] = em
        if b:
            to_lo = np.logaddexp(to[:, 1], to[:, 2]) - to[:, 0]
            sw = (1 - b) * np.nan_to_num(sw) + b * to_lo
    return decode.decode(pr["bt"], Q["model"]["w_dead"] * e["dead"] + Q["model"]["dead_bias"], pr["halves"], Q, empty,
                         model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=sw, **kw)


def spell(gt, lab):
    return pm.evaluate(pm.label_codes(gt), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"]


def rate(lab):
    live = np.isin(lab, ["H", "A"])
    starts = np.sum(live[1:] & (lab[1:] != lab[:-1])) + live[0]
    return starts / (np.sum(lab != "X") / 25 / 60)


ref = {m: label(m, ref=True) for m in ids}
s_ref = np.array([spell(preps[m]["gt"], ref[m]) for m in ids])
chk = {m: label(m, a=0.0) for m in ids}          # a = 0 must give v21 exactly
assert all((chk[m] == ref[m]).all() for m in ids), "mixing path does not reproduce v21 at a=0"
dfl = np.array([not m.startswith("MET") for m in ids])
print(f"v21 bench {s_ref.mean():.4f}  DFL {s_ref[dfl].mean():.4f}  Metrica {s_ref[~dfl].mean():.4f}  "
      f"poss/min ours {np.mean([rate(ref[m]) for m in ids]):.2f} true {np.mean([rate(preps[m]['gt']) for m in ids]):.2f}", flush=True)

cands = {}
for a in (0.25, 0.5, 0.75):
    cands[f"mix a{a}"] = dict(a=a)
    cands[f"hidden a{a}"] = dict(a=a, hidden_only=True)
for b in (0.5, 1.0):
    cands[f"turn b{b}"] = dict(b=b)
for a in (0.25, 0.5):
    for b in (0.5, 1.0):
        cands[f"mix a{a} turn b{b}"] = dict(a=a, b=b)
if os.environ.get("ONLY"):
    cands = {k: v for k, v in cands.items() if k in os.environ["ONLY"].split(",")}
boot = sys.argv[3].split(",") if len(sys.argv) > 3 else []
rng = np.random.default_rng(0)
labs = {}
for name, kw in cands.items():
    lab = {m: label(m, **kw) for m in ids}
    labs[name] = lab
    d = np.array([spell(preps[m]["gt"], lab[m]) for m in ids]) - s_ref
    line = (f"{name:22} mean {d.mean():+.4f}  up {int((d > 0).sum())}/10  worst {d.min():+.4f}  "
            f"DFL {d[dfl].mean():+.4f} MET {d[~dfl].mean():+.4f}  poss/min {np.mean([rate(lab[m]) for m in ids]):.2f}  "
            f"per match {np.round(d, 3).tolist()}")
    if name in boot:
        B = []
        for _ in range(200):
            ds = []
            for m in ids:
                gt = preps[m]["gt"]
                n, blk = len(gt), 1500
                st = rng.integers(0, n - blk, n // blk)
                idx = np.concatenate([np.arange(s, s + blk) for s in st])
                ds.append(spell(gt[idx], lab[m][idx]) - spell(gt[idx], ref[m][idx]))
            B.append(np.mean(ds))
        lo, hi = np.percentile(B, [5, 95])
        line += f"  90% range {lo:+.4f} .. {hi:+.4f}"
    print(line, flush=True)
