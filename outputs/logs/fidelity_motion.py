"""Label-free check for v18: the motion features (features.motion_block) on our match's tracking vs
each faithfully degraded outside match, through the same cleaning the pipeline uses. Tracking only:
inside-the-halves frames, no labels, no tags. Prints p10 / median / p90 per feature and match."""
import numpy as np
import pandas as pd
from possession import config, features, model
from possession.clean import clean_atd
from possession.load import load_atd

base = config.load_params()
dp = model.degrade_params(base, base["idsse_bench"]["train_degrade"])
rows = {}


def summ(tr, mask):
    f = features.motion_block(tr)[mask]
    return {f"{c} {q}": np.nanpercentile(f[c], p) for c in f.columns for q, p in (("p10", 10), ("p50", 50), ("p90", 90))}


rows["OURS"] = summ(clean_atd(load_atd(), base), config.graded_mask())
print("ours done", flush=True)
for mid in model.external_matches(base):
    m = model.load_external(mid, dp)
    rows[mid] = summ(clean_atd(m.tracking, base), m.gt != "X")
    print(mid, "done", flush=True)
df = pd.DataFrame(rows).T
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
for c in ["mx_top3", "mx_sd", "mx_run", "mx_acc3", "mx_align", "mx_block", "mx_duels"]:
    t = df[[f"{c} p10", f"{c} p50", f"{c} p90"]].copy(); t.columns = ["p10", "p50", "p90"]
    out = t.loc[[i for i in t.index if i != "OURS"]]
    print(f"\n{c}\n" + t.round(3).to_string())
    print(f"outside range of medians: {out.p50.min():.3f} - {out.p50.max():.3f}   ours {t.loc['OURS','p50']:.3f}")
