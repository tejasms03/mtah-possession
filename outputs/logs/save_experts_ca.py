"""Carrier retrain: v18 inputs + ball-carrier notes (features.carrier_extra) for model B and the
change detector. Features built in memory (disk is tight); experts -> experts_ca/."""
import os, pickle
from pathlib import Path
os.environ["MTAH_NO_CACHE_WRITE"] = "1"
from possession import config, model, selection

base = config.load_params()
base["features"]["carrier_extra"] = True
ids = model.external_matches(base)
preps = {m: model.prepare_idsse(m, base, variant=base["idsse_bench"]["train_degrade"]) for m in ids}
print("features ready", flush=True)
params = selection.version_params(base, selection.VERSIONS["v6"])
OUT = Path("outputs/logs/experts_ca"); OUT.mkdir(exist_ok=True)
for mid in ids:
    if (OUT / f"{mid}.pkl").exists():
        continue                                  # resume: keep matches already saved
    train = [preps[o] for o in ids if o != mid]
    dead, team = model.predict_ensemble(model.fit_ensemble(train, params, label="  pass 1"), preps[mid], params)
    switch = model.predict_switch(model.fit_switch(train, params), preps[mid])
    with open(OUT / f"{mid}.pkl", "wb") as fh:
        pickle.dump(dict(dead=dead, team=team, switch=switch), fh)
    print("saved ca", mid, flush=True)
