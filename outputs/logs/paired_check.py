"""Is a candidate's bench gain real? Per-match paired deltas, sign count, and a bootstrap over
possessions (resample whole true possessions in every match together) for candidate minus v19."""
import os, pickle, sys, copy
import numpy as np, pandas as pd
from possession import config, decode, model, selection
from possession.external.bench import _vendor
pm = _vendor("possession_metrics")
base = config.load_params(); ids = model.external_matches(base)
preps = {m: model.prepare_idsse(m, base, variant=base["idsse_bench"]["train_degrade"]) for m in ids}
P0 = selection.version_params(base, selection.VERSIONS["v6"])
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
def labels(P, d):
    out = {}
    for m in ids:
        e, pr = pickle.load(open(f"outputs/logs/experts_{d}/{m}.pkl", "rb")), preps[m]
        out[m] = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                               model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"])
    return out
def setp(P, kv):
    P = copy.deepcopy(P)
    for k, v in kv.items():
        node = P; *path, last = k.split(".")
        for p in path: node = node[p]
        node[last] = v
    return P
cands = {
    "seqT sc-1 w1": ("seqT_mot", {"decode.switch_center": -1.0, "model.w_team": 1.0}),
    "seqT sc-1 w1.5": ("seqT_mot", {"decode.switch_center": -1.0, "model.w_team": 1.5}),
    "seqT sc-0.5 w1.5": ("seqT_mot", {"decode.switch_center": -0.5, "model.w_team": 1.5}),
    "torl sc-1": ("torl", {"decode.switch_center": -1.0}),
}
if os.environ.get("CANDS"):
    cands = {k: v for k, v in {
        "seqT_to sc-1 w1": ("seqT_to", {"decode.switch_center": -1.0, "model.w_team": 1.0}),
        "seqT_to sc-1 w1.5": ("seqT_to", {"decode.switch_center": -1.0, "model.w_team": 1.5}),
        "seqT_mot sc-1 w1 (v20)": ("seqT_mot", {"decode.switch_center": -1.0, "model.w_team": 1.0}),
        "seqT_ca sc-0.5 w1.5": ("seqT_ca", {"decode.switch_center": -0.5, "model.w_team": 1.5}),
    }.items()}
P0["decode"]["switch_center"] = -0.5          # reference = v19 exactly
ref = labels(P0, "mot")
rng = np.random.default_rng(0)
def spell(gt, lab): return pm.evaluate(pm.label_codes(gt), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"]
for name, (d, kv) in cands.items():
    lab = labels(setp(P0, kv), d)
    delta = np.array([spell(preps[m]["gt"], lab[m]) - spell(preps[m]["gt"], ref[m]) for m in ids])
    # bootstrap: resample 1-minute blocks of each match (keeps possessions mostly whole), 300 draws
    B = []
    for _ in range(200):
        ds = []
        for m in ids:
            gt = preps[m]["gt"]; n = len(gt); blk = 1500
            starts = rng.integers(0, n - blk, n // blk)
            idx = np.concatenate([np.arange(s, s + blk) for s in starts])
            ds.append(spell(gt[idx], lab[m][idx]) - spell(gt[idx], ref[m][idx]))
        B.append(np.mean(ds))
    lo, hi = np.percentile(B, [5, 95])
    print(f"{name:18} mean {delta.mean():+.4f}  up {int((delta > 0).sum())}/10  worst {delta.min():+.4f}  "
          f"90% range {lo:+.4f} .. {hi:+.4f}   per match {np.round(delta, 3).tolist()}", flush=True)
