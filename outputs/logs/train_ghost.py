"""Train the ghost-ball models (idea D) on the outside matches' degraded, cleaned tracking."""
from possession import config, ghost, model
from possession.clean import clean_atd

base = config.load_params()
v = base["idsse_bench"]["train_degrade"]
dp = model.degrade_params(base, v)
trs, pitches = {}, {}
for mid in model.external_matches(base):
    m = model.load_external(mid, dp)
    trs[mid], pitches[mid] = clean_atd(m.tracking, base), m.pitch
    print("loaded", mid, flush=True)
ghost.train(trs, pitches, log=lambda s: print(s, flush=True))
