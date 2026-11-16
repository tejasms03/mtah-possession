"""Save v18's held-out predictions (experts_mot + current decoder params) as holdout_preds_v18/{mid}.npz."""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
from possession import config, decode, model, selection

base = config.load_params()
assert base["features"]["motion_extra"] and base["decode"]["dead_bias_miss"] == -0.75
ids = model.external_matches(base)
v = base["idsse_bench"]["train_degrade"]
q = {**base, "features": {**base["features"], "motion_extra": False}}   # v17 caches: bt/gt/halves are identical
P = selection.version_params(base, selection.VERSIONS["v6"])
empty = pd.DataFrame(columns=["start_frame", "end_frame", "type", "team"])
out = Path("outputs/logs/holdout_preds_v18"); out.mkdir(exist_ok=True)
for mid in ids:
    pr = model.prepare_idsse(mid, q, variant=v)
    e = pickle.load(open(f"outputs/logs/experts_mot/{mid}.pkl", "rb"))
    lab = decode.decode(pr["bt"], P["model"]["w_dead"] * e["dead"] + P["model"]["dead_bias"], pr["halves"], P, empty,
                        model_team_llr=e["team"], pitch=pr["pitch"], switch_logodds=e["switch"])
    np.savez(out / f"{mid}.npz", gt=pr["gt"], lab=lab)
    print("saved", mid, flush=True)
