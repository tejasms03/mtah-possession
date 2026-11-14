"""Choose between pipeline versions without the public window (RULES.md / EVALUATION.md).

Each Metrica sample match plays "our match" in turn: models train on the other 9 outside matches,
the match's own SET PIECE events become stand-in Smart Tagging tags (shifted `tag_lead_s` earlier,
since our tags start before the restart; no team for throw-ins and kick-offs, like ours), the
version's full pipeline runs (including the second pass that learns from tags), and spellF1 is
scored against that match's labels. `uv run possession select` prints the table.
"""

import copy

import numpy as np
import pandas as pd

from possession import config, deadball, decode, model
from possession.external import metrica
from possession.progress import Progress

VERSIONS = {
    "v6":   {"own.use": False, "ensemble.size": 1},
    "v7":   {"own.use": True, "ensemble.size": 1},
    "v8":   {"own.use": True, "ensemble.size": 1, "own.combine": "mean3"},
    "v9":   {"own.use": True, "ensemble.size": 1, "own.combine": "max"},
    "v10":  {"own.use": True, "ensemble.size": 1, "own_team.use": True},
    "v11b": {"own.use": True, "ensemble.size": 5},
}
HOME = {"MET1": "Home", "MET2": "Home", "MET3": "Team A"}


def version_params(base: dict, changes: dict) -> dict:
    q = copy.deepcopy(base)
    for k, v in changes.items():
        node = q["model"]
        *path, last = k.split(".")
        for part in path:
            node = node[part]
        node[last] = v
    return q


def stand_in_tags(match_id: str, n: int, lead_s: float) -> pd.DataFrame:
    """Smart-Tagging-like SET PIECES rows from a Metrica match's own events (row = frame - 1)."""
    ev = metrica._events(metrica.MATCHES[match_id], HOME[match_id])
    sp = ev[ev["type"] == "SET PIECE"]
    start = np.clip(sp["start"].to_numpy() - 1 - int(lead_s * config.FPS), 0, n - 1)
    no_team = sp["sub"].str.contains("THROW|KICK OFF")
    return pd.DataFrame({"code": "SET PIECES", "type": sp["sub"].to_numpy(),
                         "team": np.where(no_team, "", sp["team"].to_numpy()),
                         "start_frame": start, "end_frame": start + 5 * config.FPS})


def run_version(prep, train, tags, params) -> np.ndarray:
    """The pipeline's label_with_models, on a prepared outside match."""
    members = model.fit_ensemble(train, params, label="  pass 1")
    dead, team = model.predict_ensemble(members, prep, params)
    if params["model"]["own"]["use"]:
        idx, y = model.own_match_rows(prep, tags, dead, params, exclude=())
        own_team = None
        if params["model"]["own_team"]["use"]:
            ti, tl = model.own_team_rows(prep, tags, params, exclude=())
            own_team = (prep["own_H"][ti], prep["own_A"][ti], tl)
        members = model.fit_ensemble(train, params, own=(prep["dead"][idx], y), own_team=own_team,
                                     label="  pass 2")
        dead, team = model.predict_ensemble(members, prep, params)
    switch = None
    if params["decode"].get("w_switch", 0):
        switch = model.predict_switch(model.fit_switch(train, params), prep)
    m = params["model"]
    return decode.decode(prep["bt"], m["w_dead"] * dead + m["dead_bias"], prep["halves"], params,
                         deadball.set_piece_hints(tags), model_team_llr=team, pitch=prep["pitch"],
                         switch_logodds=switch)


def main(versions=None, extra_team: bool = False) -> None:
    from possession.external.bench import _vendor

    pm = _vendor("possession_metrics")
    base = config.load_params()
    base["features"]["extra_team"] = extra_team or base["features"].get("extra_team", False)
    lead = base["selection"]["tag_lead_s"]
    versions = versions or list(VERSIONS)
    ids = model.external_matches(base)
    preps = {m: model.prepare_idsse(m, base) for m in ids}
    mets = list(metrica.MATCHES)
    bar = Progress("select", len(versions) * len(mets))
    scores = {v: [] for v in versions}
    for v in versions:
        params = version_params(base, VERSIONS[v])
        for mid in mets:
            prep = preps[mid]
            tags = stand_in_tags(mid, len(prep["gt"]), lead)
            lab = run_version(prep, [preps[o] for o in ids if o != mid], tags, params)
            r = pm.evaluate(pm.label_codes(prep["gt"]), pm.label_codes(lab))
            scores[v].append((r["spell_f1_mean"], r["macro_f1"]))
            bar.step(f"{v} {mid}: spellF1 {r['spell_f1_mean']:.3f} macroF1 {r['macro_f1']:.3f}")
    print("\nversion  spellF1 (MET1 MET2 MET3)          mean   macroF1 mean")
    for v in versions:
        s = scores[v]
        print(f"{v:7}  {' '.join(f'{a:.3f}' for a, _ in s):24}  {np.mean([a for a, _ in s]):.4f}"
              f"  {np.mean([b for _, b in s]):.4f}", flush=True)
