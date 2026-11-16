"""Sweep decoder settings on saved held-out expert outputs (10 outside matches). Usage:
EXP=name python decode_sweep.py 'decode.gap_pool=0,0.5' [...]   (grid; EXP '' = v17 experts; GAP=1 = EXP=gap)
Always prints the v17 row first; every delta (dMET, dDFL, worst) is against v17."""
import copy, itertools, os, pickle, sys
import numpy as np, pandas as pd
from possession import config, decode, model, selection
from possession.external.bench import _vendor

pm = _vendor("possession_metrics")
EXP = os.environ.get("EXP", "gap" if os.environ.get("GAP") == "1" else "")
base = config.load_params()
ids = model.external_matches(base)
v = base["idsse_bench"]["train_degrade"]
# ball track, labels and halves do not depend on the extra feature flags: v17 caches serve every variant
preps = {m: model.prepare_idsse(m, base, variant=v) for m in ids}
load = lambda d: {m: pickle.load(open(f"outputs/logs/experts{d}/{m}.pkl", "rb")) for m in ids}  # noqa: E731
BASE = os.environ.get("BASE", "")            # experts used for the reference row ("mot" = v18)
ex17 = load(f"_{BASE}" if BASE else "")
ex = load(f"_{EXP}") if EXP else ex17
P0 = selection.version_params(base, selection.VERSIONS["v6"])
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
met = np.array([m.startswith("MET") for m in ids])


def run(P, E):
    out = []
    for mid in ids:
        e, pr = E[mid], preps[mid]
        lab = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P,
                            empty, model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"])
        out.append(pm.evaluate(pm.label_codes(pr["gt"]), pm.label_codes(lab), bootstrap_n=0)["spell_f1_mean"])
    return np.array(out)


def row(name, s):
    d = s - ref
    print(name.ljust(30)[:30], " ".join(f"{x:6.3f}" for x in s),
          f"  {s.mean():.4f}  {d[met].mean():+.3f} {d[~met].mean():+.3f} {d.min():+.3f}", flush=True)


axes = [(a.split("=")[0], [float(x) for x in a.split("=")[1].split(",")]) for a in sys.argv[1:]]
ref = run(P0, ex17)
print("experts:", EXP or "v17")
print("match".ljust(30) + " " + " ".join(f"{m[-4:]:>6}" for m in ids) + "    mean   dMET  dDFL  worst")
row(BASE or "v17", ref)
if EXP:
    row(EXP, run(P0, ex))
for combo in itertools.product(*[vals for _, vals in axes]):
    P = copy.deepcopy(P0)
    for (k, _), val in zip(axes, combo):
        node = P; *path, last = k.split(".")
        for p in path: node = node[p]
        node[last] = bool(val) if isinstance(node[last], bool) else (int(val) if isinstance(node[last], int) and float(val).is_integer() else float(val))
    row(" ".join(f"{k.split('.')[-1]}={val:g}" for (k, _), val in zip(axes, combo)), run(P, ex))
