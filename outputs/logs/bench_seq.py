"""Tree sequence model on the 10-match hold-out. For held-out match m, the second stage trains on
the other 9 matches' saved first-stage outputs (experts_BASE/, each from models that never saw
that match; those models did see m, a second-order leak noted in DECISIONS). Writes:
  experts_seq_BASE/   team and switch from the second stage
  experts_seqT_BASE/  team from the second stage, first-stage switch
  experts_seqM_BASE/  team = mean of first and second stage, second-stage switch
Usage: python bench_seq.py [BASE=mot]"""
import pickle, sys
from pathlib import Path
import numpy as np
from possession import config, model, sequence

B = sys.argv[1] if len(sys.argv) > 1 else "mot"
base = config.load_params(); ids = model.external_matches(base)
preps = {m: model.prepare_idsse(m, base, variant=base["idsse_bench"]["train_degrade"]) for m in ids}
E = {m: pickle.load(open(f"outputs/logs/experts_{B}/{m}.pkl", "rb")) for m in ids}
obs = {m: preps[m]["bt"].observed for m in ids}
outs = {k: Path(f"outputs/logs/experts_{k}_{B}") for k in ("seq", "seqT", "seqM")}
for p in outs.values(): p.mkdir(exist_ok=True)
for m in ids:
    models = sequence.fit([(E[o], obs[o], preps[o]["gt"]) for o in ids if o != m])
    t2, s2 = sequence.predict(models, E[m], obs[m])
    e = E[m]
    for k, d in (("seq", dict(dead=e["dead"], team=t2, switch=s2)), ("seqT", dict(dead=e["dead"], team=t2, switch=e["switch"])),
                 ("seqM", dict(dead=e["dead"], team=0.5 * (t2 + np.nan_to_num(e["team"])), switch=s2))):
        pickle.dump(d, open(outs[k] / f"{m}.pkl", "wb"))
    live = np.isin(preps[m]["gt"], ["H", "A"])
    acc1 = np.mean((np.nan_to_num(e["team"])[live] > 0) == (preps[m]["gt"][live] == "H"))
    acc2 = np.mean((t2[live] > 0) == (preps[m]["gt"][live] == "H"))
    print(f"{m}: frame team accuracy on live frames, first stage {acc1:.3f} -> second stage {acc2:.3f}", flush=True)
