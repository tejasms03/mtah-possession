"""v21 on the 10-match bench: where spellF1 loses, in the scorer's own terms."""
import os, pickle, copy
import numpy as np, pandas as pd
from possession import config, decode, model, rules, selection
from possession.external.bench import _vendor
pm = _vendor("possession_metrics")
base = config.load_params(); ids = model.external_matches(base)
v = base["idsse_bench"]["train_degrade"]
P = selection.version_params(base, selection.VERSIONS["v6"]); P = copy.deepcopy(P)
P["decode"]["switch_center"] = -1.0; P["model"]["w_team"] = 1.0
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
rows, gl, pl, tot = [], [], [], {t: [0, 0, 0] for t in pm.SPELL_IOUS}
cat = {"no overlap (wrong team / dead)": 0, "merged into a longer one": 0, "split by flip or D": 0, "boundary off (best IoU <0.5)": 0}
sp = {"inside a true spell of the other team / D": 0, "fragment of a true spell": 0, "other": 0}
for mid in ids:
    pb = copy.deepcopy(base); pb['features'].update(turnover_extra=False, restart_extra=False, carrier_extra=False)
    pr = model.prepare_idsse(mid, pb, variant=v)  # cached v18 features; decode needs only ball, labels, halves
    e = pickle.load(open(f"outputs/logs/experts_seqT_to/{mid}.pkl", "rb"))
    lab = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                        model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"], ctx=rules.context(pr, base, None))
    g, p = pm.label_codes(pr["gt"]), pm.label_codes(lab); runs = pm.valid_runs(g)
    G, Q = pm.possession_spells(g, runs), pm.possession_spells(pm.fill_abstentions(p, runs), runs)
    ov = pm.spell_overlaps(G, Q); s = pm.spell_scores(g, p, runs)
    for t in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(ov, len(G), len(Q), t); tot[t][0] += tp; tot[t][1] += fp; tot[t][2] += fn
    m5 = pm.matched_gt_flags(ov, len(G), 0.5)
    best = {}; ng = {}; nq = {}
    for iou, i, j in ov:
        best[i] = max(best.get(i, 0), iou); ng[j] = ng.get(j, 0) + 1; nq[i] = nq.get(i, 0) + 1
    for i, (a, b, k) in enumerate(G):
        gl.append((b - a, m5[i]))
        if m5[i]: continue
        if i not in nq: cat["no overlap (wrong team / dead)"] += 1
        elif nq[i] > 1: cat["split by flip or D"] += 1
        elif any(ng.get(j) and ng[j] > 1 for _, ii, j in ov if ii == i): cat["merged into a longer one"] += 1
        else: cat["boundary off (best IoU <0.5)"] += 1
    used = set(j for _, i, j in ov if m5[i] and False)
    mq = np.zeros(len(Q), bool)
    seen = set()
    for iou, i, j in ov:
        if iou < 0.5: break
        if i in seen or mq[j]: continue
        seen.add(i); mq[j] = True
    for j, (a, b, k) in enumerate(Q):
        pl.append((b - a, mq[j]))
        if mq[j]: continue
        if j not in ng: sp["inside a true spell of the other team / D"] += 1
        elif any(i in seen for _, i, jj in ov if jj == j): sp["fragment of a true spell"] += 1
        else: sp["other"] += 1
    rows.append((mid, len(G), len(Q), s["mean_f1"], s["per_iou"][0.3]["f1"], s["per_iou"][0.7]["f1"]))
print("match   true  ours  spellF1  F1@0.3  F1@0.7")
for r in rows: print(f"{r[0][-6:]:>6} {r[1]:5d} {r[2]:5d}  {r[3]:.3f}   {r[4]:.3f}   {r[5]:.3f}")
print("\nall matches, per IoU threshold: precision recall F1")
for t, (tp, fp, fn) in tot.items():
    pr_, rc = tp / (tp + fp), tp / (tp + fn); print(f"  {t}: P {pr_:.3f} R {rc:.3f} F1 {2*pr_*rc/(pr_+rc):.3f}  matched {tp} spurious {fp} missed {fn}")
print("\ntrue spells missed at IoU 0.5, by cause:", cat)
print("our spells unmatched at IoU 0.5, by cause:", sp)
bins = [0, 25, 50, 100, 250, 500, 1500, 10**9]; names = ["<1s", "1-2s", "2-4s", "4-10s", "10-20s", "20-60s", ">60s"]
for nm, L in (("TRUE spells", gl), ("OUR spells", pl)):
    L = np.array(L, float); print(f"\n{nm} by length: count / share matched at 0.5")
    for lo, hi, n in zip(bins, bins[1:], names):
        x = L[(L[:, 0] >= lo) & (L[:, 0] < hi)]; print(f"  {n:>7}: {len(x):5d}  {x[:,1].mean() if len(x) else 0:.2f}")
