"""Hold out EVERY outside match in turn (train on the other 9, faithful damage), v17 settings.
No stand-in tags (IDSSE has none), so Metrica rows here are 'no tags' and comparable to IDSSE rows.
Prints per-match: spellF1, macroF1, frame exact, dead/live right, team right when both live,
plus label profile of the match (dead share, live spell median s, spells/90)."""
import numpy as np
import pandas as pd
from possession import config, model, selection
from possession.external.bench import _vendor

pm = _vendor("possession_metrics")
base = config.load_params()
ids = model.external_matches(base)
v = base["idsse_bench"]["train_degrade"]
preps = {m: model.prepare_idsse(m, base, variant=v) for m in ids}
params = selection.version_params(base, selection.VERSIONS["v6"])
empty = pd.DataFrame(columns=["code", "type", "team", "start_frame", "end_frame"])
rows = []
for mid in ids:
    prep = preps[mid]
    lab = np.asarray(selection.run_version(prep, [preps[o] for o in ids if o != mid], empty, params))
    gt = np.asarray(prep["gt"]); g = gt != "X"
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(lab))
    a, b = gt[g], lab[g]
    live = (a != "D") & (b != "D")
    e = np.flatnonzero(np.diff(np.r_[0, (a != "D").astype(np.int8), 0]))
    sp = (e[1::2] - e[::2]) / 25
    row = dict(match=mid, spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"], exact=np.mean(a == b),
               deadlive=np.mean((a == "D") == (b == "D")), team_live=np.mean(a[live] == b[live]),
               dead_share=np.mean(a == "D"), live_spell_med_s=np.median(sp),
               spells_per90=len(sp) / (g.sum() / 25 / 5400))
    rows.append(row)
    print(" ".join(f"{k}={x:.3f}" if isinstance(x, float) else f"{k}={x}" for k, x in row.items()), flush=True)
df = pd.DataFrame(rows).set_index("match")
pd.set_option("display.width", 200)
print(df.round(3)); print("\nmean", df.mean().round(3).to_dict()); print("min ", df.min().round(3).to_dict())
