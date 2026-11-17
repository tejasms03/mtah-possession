"""How high can we go? The 10-match hold-out on CLEAN tracking (no damage at all: ball and players
as the providers give them), v19 models and decoder settings. Compare with the faithful-damage
bench (0.7098): the difference is what the damage (missing ball, dropouts) costs us.
Features in memory only. Experts -> experts_clean/."""
import os, pickle
from pathlib import Path
os.environ["MTAH_NO_CACHE_WRITE"] = "1"
import numpy as np, pandas as pd
from possession import config, decode, model, selection
from possession.clean import clean_atd
from possession.external import idsse, metrica
from possession.external.bench import _vendor, bench_gt
pm = _vendor("possession_metrics")
base = config.load_params(); ids = model.external_matches(base)
preps = {}
for m in ids:
    mt = (metrica if m.startswith("MET") else idsse).load_match(m)
    pr = model.prepare(clean_atd(mt.tracking, base), mt.halves, mt.pitch, base, gt=bench_gt(mt)); pr["id"] = m
    preps[m] = pr; print("features", m, "ball seen", round(float(np.mean(pr["bt"].observed)), 3), flush=True)
P = selection.version_params(base, selection.VERSIONS["v6"])
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
OUT = Path("outputs/logs/experts_clean"); OUT.mkdir(exist_ok=True)
rows = []
for m in ids:
    train = [preps[o] for o in ids if o != m]
    dead, team = model.predict_ensemble(model.fit_ensemble(train, P, label="  clean"), preps[m], P)
    switch = model.predict_switch(model.fit_switch(train, P), preps[m])
    pickle.dump(dict(dead=dead, team=team, switch=switch), open(OUT / f"{m}.pkl", "wb"))
    lab = decode.decode(preps[m]["bt"], P["model"]["w_dead"] * dead + P["model"]["dead_bias"], preps[m]["halves"], P, empty,
                        model_team_llr=team, pitch=preps[m]["pitch"], switch_logodds=switch)
    r = pm.evaluate(pm.label_codes(preps[m]["gt"]), pm.label_codes(lab), bootstrap_n=0)
    rows.append((m, r["spell_f1_mean"], r["macro_f1"])); print("clean", m, round(r["spell_f1_mean"], 4), round(r["macro_f1"], 4), flush=True)
t = pd.DataFrame(rows, columns=["match", "spellF1", "macroF1"]).set_index("match")
print(t.round(4).to_string()); print("MEAN", t.mean().round(4).to_dict())
