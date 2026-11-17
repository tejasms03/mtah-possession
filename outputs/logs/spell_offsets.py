import pickle, copy
import numpy as np, pandas as pd
from possession import config, decode, model, rules, selection
from possession.external.bench import _vendor
pm = _vendor("possession_metrics")
base = config.load_params(); ids = model.external_matches(base); v = base["idsse_bench"]["train_degrade"]
P = copy.deepcopy(selection.version_params(base, selection.VERSIONS["v6"]))
P["decode"]["switch_center"] = -1.0; P["model"]["w_team"] = 1.0
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
S, E, KS, KE = [], [], [], []
for mid in ids:
    pb = copy.deepcopy(base); pb['features'].update(turnover_extra=False, restart_extra=False, carrier_extra=False)
    pr = model.prepare_idsse(mid, pb, variant=v)
    e = pickle.load(open(f"outputs/logs/experts_seqT_to/{mid}.pkl", "rb"))
    lab = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                        model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"], ctx=rules.context(pr, base, None))
    g = pm.label_codes(pr["gt"]); runs = pm.valid_runs(g); p = pm.fill_abstentions(pm.label_codes(lab), runs)
    G, Q = pm.possession_spells(g, runs), pm.possession_spells(p, runs); ov = pm.spell_overlaps(G, Q)
    seen, used = set(), set()
    for iou, i, j in ov:
        if iou < 0.3: break
        if i in seen or j in used: continue
        seen.add(i); used.add(j)
        S.append(Q[j][0] - G[i][0]); E.append(Q[j][1] - G[i][1])
        KS.append(g[G[i][0] - 1] if G[i][0] > 0 else -1); KE.append(g[G[i][1]] if G[i][1] < len(g) else -1)
S, E, KS, KE = map(np.array, (S, E, KS, KE))
def show(nm, x):
    print(f"{nm:34} n={len(x):5d} median {np.median(x):+5.0f}  mean {x.mean():+6.1f}  |err| median {np.median(abs(x)):4.0f}  within 12f {np.mean(abs(x)<=12):.2f}  within 25f {np.mean(abs(x)<=25):.2f}")
print("frames, + = ours later than truth (25 fps)")
show("start, after D (restart)", S[KS == pm.D]); show("start, after other team (turnover)", S[np.isin(KS, [pm.H, pm.A])])
show("end, into D (stoppage)", E[KE == pm.D]); show("end, into other team (turnover)", E[np.isin(KE, [pm.H, pm.A])])
