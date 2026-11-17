"""Boundary refiner on the 10-match bench (leave one match out). Reference = v21 (seqT_to experts,
switch_center -1, w_team 1). Usage: python refine_bench.py  [W list] [thr list]"""
import copy, os, pickle, sys
import numpy as np, pandas as pd
from possession import config, decode, model, refine, rules, selection
from possession.external.bench import _vendor
pm = _vendor("possession_metrics")
base = config.load_params(); ids = model.external_matches(base); v = base["idsse_bench"]["train_degrade"]
P = copy.deepcopy(selection.version_params(base, selection.VERSIONS["v6"]))
P["decode"]["switch_center"] = -1.0; P["model"]["w_team"] = 1.0
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
EXP = os.environ.get("EXP", "seqT_to")
LC = "outputs/logs/refine_cache"; os.makedirs(LC, exist_ok=True)
D = {}
for mid in ids:
    f = f"{LC}/{mid}_{EXP}.pkl"
    if not os.path.exists(f):
        pb = copy.deepcopy(base); pb["features"].update(turnover_extra=False, restart_extra=False, carrier_extra=False)
        pr = model.prepare_idsse(mid, pb, variant=v)
        e = pickle.load(open(f"outputs/logs/experts_{EXP}/{mid}.pkl", "rb"))
        lab = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                            model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"], ctx=rules.context(pr, base, None))
        pickle.dump({"bt": pr["bt"], "gt": pr["gt"], "halves": pr["halves"], "e": e, "lab": np.asarray(lab)}, open(f, "wb"))
        del pr
    D[mid] = pickle.load(open(f, "rb"))
Ws = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "25,38").split(",")]
thrs = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.2,0.3,0.4,0.5").split(",")]
score = lambda g, p: pm.evaluate(pm.label_codes(g), pm.label_codes(p), bootstrap_n=0)["spell_f1_mean"]  # noqa: E731
ref = np.array([score(D[m]["gt"], D[m]["lab"]) for m in ids])
met = np.array([m.startswith("MET") for m in ids])
print("match".ljust(26) + " ".join(f"{m[-4:]:>6}" for m in ids) + "    mean   gain  up  worst  dDFL   dMET")
print("v21".ljust(26) + " ".join(f"{x:6.3f}" for x in ref) + f"  {ref.mean():.4f}")
for W in Ws:
    C = {}
    for m in ids:
        d = D[m]; F = refine.frame_table(d["bt"], d["e"])
        X, bi, cf, bs = refine.candidates(d["lab"], d["halves"], F, W)
        C[m] = (X, bi, cf, bs, refine.targets(np.asarray(d["gt"]), bs, bi, cf))
    models = {m: refine.fit([(C[o][0], C[o][4]) for o in ids if o != m]) for m in ids}
    for kinds in (("turnover",), ("turnover", "restart", "stoppage")):
        for thr in thrs:
            s = np.array([score(D[m]["gt"], refine.apply(D[m]["lab"], models[m], *C[m][:4], thr=thr, kinds=kinds)) for m in ids])
            dd = s - ref
            nm = f"W{W} {'turn' if len(kinds) == 1 else 'all'} thr{thr:g}"
            print(nm.ljust(26) + " ".join(f"{x:6.3f}" for x in s) +
                  f"  {s.mean():.4f} {dd.mean():+.4f} {int((dd > 0).sum()):2d}  {dd.min():+.3f} {dd[~met].mean():+.3f} {dd[met].mean():+.3f}", flush=True)
