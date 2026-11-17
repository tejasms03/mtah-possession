"""In-memory expert saving (disk is tight). Usage: python save_experts_mem.py NAME feature_flag ...
e.g. save_experts_mem.py toca turnover_extra carrier_extra -> experts_toca/ (resumes)."""
import os, pickle, sys
from pathlib import Path
os.environ["MTAH_NO_CACHE_WRITE"] = "1"
from possession import config, model, selection

name, flags = sys.argv[1], sys.argv[2:]
base = config.load_params()
for f in flags:
    base["features"][f] = True
ids = model.external_matches(base)
preps = {m: model.prepare_idsse(m, base, variant=base["idsse_bench"]["train_degrade"]) for m in ids}
print("features ready", flags, flush=True)
params = selection.version_params(base, selection.VERSIONS["v6"])
params["model"]["sequence"]["use"] = False
OUT = Path(f"outputs/logs/experts_{name}"); OUT.mkdir(exist_ok=True)
for mid in ids:
    if (OUT / f"{mid}.pkl").exists():
        continue
    train = [preps[o] for o in ids if o != mid]
    dead, team = model.predict_ensemble(model.fit_ensemble(train, params, label="  pass 1"), preps[mid], params)
    switch = model.predict_switch(model.fit_switch(train, params), preps[mid])
    with open(OUT / f"{mid}.pkl", "wb") as fh:
        pickle.dump(dict(dead=dead, team=team, switch=switch), fh)
    print("saved", name, mid, flush=True)
