"""Turnover (to_) and restart-layout (rl_) retrain: build the 10 matches' features once (cached
ones loaded, missing ones kept in memory only), then save three expert sets: torl (both), to
(turnover only), rl (restart layout only)."""
import copy, os, pickle
from pathlib import Path
os.environ["MTAH_NO_CACHE_WRITE"] = "1"
from possession import config, model, selection

base = config.load_params()
base["features"]["turnover_extra"] = True
base["features"]["restart_extra"] = True
ids = model.external_matches(base)
preps = {m: model.prepare_idsse(m, base, variant=base["idsse_bench"]["train_degrade"]) for m in ids}
print("features ready", flush=True)
D = list(base["model"]["dead_drop_prefixes"])
VARIANTS = {"torl": {}, "to": {"dead_drop_prefixes": D + ["rl_"], "switch_drop_prefixes": ["rl_"]},
            "rl": {"switch_drop_prefixes": ["to_", "rl_"]}}
import sys
ONLY = sys.argv[1:] or list(VARIANTS)
for name, over in VARIANTS.items():
    if name not in ONLY:
        continue
    params = selection.version_params(base, selection.VERSIONS["v6"])
    params["model"].update(copy.deepcopy(over))
    OUT = Path(f"outputs/logs/experts_{name}"); OUT.mkdir(exist_ok=True)
    for mid in ids:
        if (OUT / f"{mid}.pkl").exists():
            continue                              # resume
        train = [preps[o] for o in ids if o != mid]
        dead, team = model.predict_ensemble(model.fit_ensemble(train, params, label="  pass 1"), preps[mid], params)
        switch = model.predict_switch(model.fit_switch(train, params), preps[mid]) if params["decode"].get("w_switch", 0) else None
        with open(OUT / f"{mid}.pkl", "wb") as fh:
            pickle.dump(dict(dead=dead, team=team, switch=switch), fh)
        print("saved", name, mid, flush=True)
