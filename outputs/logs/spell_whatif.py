"""v21 bench: fix one kind of spell error with the truth (only on those spans), rescore. Upper bounds."""
import pickle, copy
import numpy as np, pandas as pd
from possession import config, decode, model, rules, selection
from possession.external.bench import _vendor
pm = _vendor("possession_metrics")
base = config.load_params(); ids = model.external_matches(base); v = base["idsse_bench"]["train_degrade"]
P = copy.deepcopy(selection.version_params(base, selection.VERSIONS["v6"]))
P["decode"]["switch_center"] = -1.0; P["model"]["w_team"] = 1.0
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
KEYS = ["v21", "missed: wrong team/dead over it", "missed: merged into longer", "missed: split", "missed: boundary off",
        "all missed true spells", "spurious: inside other team/D", "spurious: fragments", "all spurious ours",
        "missed <4s only", "missed >=4s only", "boundaries of matched spells", "D blips <2s inside a true spell"]
res = {k: [] for k in KEYS}
f1 = lambda g, p, r: pm.spell_scores(g, p, r)["mean_f1"]
for mid in ids:
    pb = copy.deepcopy(base); pb['features'].update(turnover_extra=False, restart_extra=False, carrier_extra=False)
    pr = model.prepare_idsse(mid, pb, variant=v)  # cached v18 features; decode needs only ball, labels, halves
    e = pickle.load(open(f"outputs/logs/experts_seqT_to/{mid}.pkl", "rb"))
    lab = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                        model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"], ctx=rules.context(pr, base, None))
    g = pm.label_codes(pr["gt"]); runs = pm.valid_runs(g); p = pm.fill_abstentions(pm.label_codes(lab), runs)
    G, Q = pm.possession_spells(g, runs), pm.possession_spells(p, runs); ov = pm.spell_overlaps(G, Q)
    m5 = pm.matched_gt_flags(ov, len(G), 0.5)
    nq, ng, best = {}, {}, {}
    for iou, i, j in ov:
        nq[i] = nq.get(i, 0) + 1; ng[j] = ng.get(j, 0) + 1
    seen, mq, pair = set(), np.zeros(len(Q), bool), {}
    for iou, i, j in ov:
        if iou < 0.5: break
        if i in seen or mq[j]: continue
        seen.add(i); mq[j] = True; pair[i] = j
    def cause(i):
        if i not in nq: return "missed: wrong team/dead over it"
        if nq[i] > 1: return "missed: split"
        if any(ng[j] > 1 for _, ii, j in ov if ii == i): return "missed: merged into longer"
        return "missed: boundary off"
    def paste(spans):
        q = p.copy()
        for a, b in spans: q[a:b] = g[a:b]
        return f1(g, q, runs)
    miss = [i for i in range(len(G)) if not m5[i]]
    res["v21"].append(f1(g, p, runs))
    for k in ["missed: wrong team/dead over it", "missed: merged into longer", "missed: split", "missed: boundary off"]:
        res[k].append(paste([G[i][:2] for i in miss if cause(i) == k]))
    res["all missed true spells"].append(paste([G[i][:2] for i in miss]))
    res["missed <4s only"].append(paste([G[i][:2] for i in miss if G[i][1] - G[i][0] < 100]))
    res["missed >=4s only"].append(paste([G[i][:2] for i in miss if G[i][1] - G[i][0] >= 100]))
    spur = [j for j in range(len(Q)) if not mq[j]]
    inside = [Q[j][:2] for j in spur if j not in ng]; frag = [Q[j][:2] for j in spur if j in ng]
    res["spurious: inside other team/D"].append(paste(inside)); res["spurious: fragments"].append(paste(frag))
    res["all spurious ours"].append(paste(inside + frag))
    sp = []
    for i, j in pair.items():
        (a, b, _), (c, d, _) = G[i], Q[j]; sp.append((min(a, c), max(b, d)))
    res["boundaries of matched spells"].append(paste(sp))
    q = p.copy()
    for a, b, k in pm.segments(p, runs):
        if k == pm.D and b - a < 50 and np.isin(g[a:b], [pm.H, pm.A]).all(): q[a:b] = g[a:b]
    res["D blips <2s inside a true spell"].append(f1(g, q, runs))
b0 = np.array(res["v21"])
print(f"{'fix with truth (only these spans)':42} mean   gain   DFL    MET")
met = np.array([m.startswith("MET") for m in ids])
for k in KEYS:
    x = np.array(res[k]); d = x - b0
    print(f"{k:42} {x.mean():.3f} {d.mean():+.3f} {d[~met].mean():+.3f} {d[met].mean():+.3f}")
