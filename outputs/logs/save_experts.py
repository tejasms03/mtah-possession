"""Save each held-out match's expert outputs (models trained on the other 9) so decoder settings
can be swept in seconds. Outside matches only. Usage:
  python save_experts.py                      -> experts/        (v17)
  python save_experts.py gap                  -> experts_gap/    (features.gap_extra on)
  python save_experts.py NAME key=val ...     -> experts_NAME/   (params overrides)
  add --tmpcache to delete feature caches this run created once they are loaded (disk is tight)"""
import pickle, sys
from pathlib import Path
from possession import config, model, selection

args = [a for a in sys.argv[1:] if a != "--tmpcache"]
TMP = "--tmpcache" in sys.argv[1:]
name = args[0] if args else ""
over = dict(a.split("=", 1) for a in args[1:])
if name == "gap" and not over:
    over = {"features.gap_extra": "1"}
OUT = Path("outputs/logs/experts" + (f"_{name}" if name else "")); OUT.mkdir(exist_ok=True)
base = config.load_params()


def setp(P, k, val):
    node = P; *path, last = k.split(".")
    for p in path: node = node[p]
    cur = node.get(last)
    if isinstance(cur, list):
        node[last] = [x for x in val.split(",") if x]
        return
    node[last] = (val in ("1", "true", "True")) if isinstance(cur, bool) else (int(float(val)) if isinstance(cur, int) and float(val).is_integer() else float(val))


for k, val in over.items():
    setp(base, k, val)
print("overrides:", over, flush=True)
ids = model.external_matches(base)
v = base["idsse_bench"]["train_degrade"]
cache = model.CACHE
before = set(cache.glob("*.pkl"))
preps = {m: model.prepare_idsse(m, base, variant=v) for m in ids}
if TMP:
    for f in set(cache.glob("*.pkl")) - before:
        f.unlink(); print("removed temp cache", f.name, flush=True)
params = selection.version_params(base, selection.VERSIONS["v6"])
for k, val in over.items():
    setp(params, k, val)
for mid in ids:
    train = [preps[o] for o in ids if o != mid]
    dead, team = model.predict_ensemble(model.fit_ensemble(train, params, label="  pass 1"), preps[mid], params)
    switch = None
    if params["decode"].get("w_switch", 0):
        switch = model.predict_switch(model.fit_switch(train, params), preps[mid])
    with open(OUT / f"{mid}.pkl", "wb") as fh:
        pickle.dump(dict(dead=dead, team=team, switch=switch), fh)
    print("saved", mid, flush=True)
