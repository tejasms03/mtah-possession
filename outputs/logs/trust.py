"""Evidence to trust the bench: v17/v18/v19 on the 10-match hold-out (paired per match),
public-window bootstrap width, and our match vs bench summary stats (aggregates only)."""
import copy, pickle, sys
import numpy as np, pandas as pd
from possession import config, decode, model, selection
from possession.external.bench import _vendor
pm = _vendor("possession_metrics")
base = config.load_params(); ids = model.external_matches(base)
preps = {m: model.prepare_idsse(m, base, variant=base["idsse_bench"]["train_degrade"]) for m in ids}
P19 = selection.version_params(base, selection.VERSIONS["v6"])
P18 = copy.deepcopy(P19); P18["decode"].update(lambda_ha=2, switch_center=0.0, merge_sandwich_s=0, last_touch_s=0)
P17 = copy.deepcopy(P18); P17["decode"].update(dead_bias_miss=0.0)
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
def run(P, d):
    s = []
    for m in ids:
        e, pr = pickle.load(open(f"outputs/logs/experts{d}/{m}.pkl", "rb")), preps[m]
        lab = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                            model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"])
        s.append(pm.evaluate(pm.label_codes(pr["gt"]), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"])
    return np.array(s)
t = pd.DataFrame({"v17": run(P17, ""), "v18": run(P18, "_mot"), "v19": run(P19, "_mot")}, index=[m[-4:] for m in ids])
t["v19-v17"] = t.v19 - t.v17
print(t.round(3).to_string()); print("mean", t.mean().round(4).to_dict())
print("matches where v19 >= v17:", int((t["v19-v17"] >= 0).sum()), "/ 10")
sys.path.insert(0, "vendor/mtah/evaluation")
gt = pm.label_codes(pm.read_label_csv("vendor/mtah/evaluation/gt/BAR-ATH_possession_gt_public.csv"))
for v in ("v15", "v19"):
    r = pm.evaluate(gt, pm.label_codes(pm.read_label_csv(f"outputs/saved/models_{v}.csv")), bootstrap_n=1000)
    ci = {k: r[k] for k in r if "ci" in k.lower() or "lo" in k.lower() or "hi" in k.lower()}
    print("public", v, round(r["spell_f1_mean"], 3), {k: (np.round(x, 3) if isinstance(x, (float, tuple, list, np.ndarray)) else x) for k, x in ci.items()})
lab = pm.read_label_csv("outputs/saved/models_v19.csv")
lab = np.asarray(lab["label"] if hasattr(lab, "columns") else lab)
g = np.r_[11185:90652, 110808:190198]; L = lab[g]
poss = int(np.sum((np.isin(L[1:], ["H", "A"])) & (L[1:] != L[:-1]))) + int(L[0] in "HA")
mins = len(g) / 25 / 60
print(f"our match v19: possessions {poss} ({poss/mins:.2f}/min), dead {np.mean(L=='D'):.3f}")
