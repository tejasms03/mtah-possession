"""Stage 4b/c: model A (is the ball dead?) and model B (whose ball, on live frames).

Gradient-boosted trees (scikit-learn HistGradientBoostingClassifier, the LightGBM algorithm;
LightGBM itself needs a system libomp on macOS, this needs nothing extra). Fixed seed.

Training data: degraded IDSSE (labels: BallStatus for A; merged BallPossession for B).
Rows: every `row_step`-th graded frame (neighbours are near copies).
Model B is team-blind: each live frame is shown once from each team's side (own/opp columns),
labelled "does own have it". At prediction, H's log-odds = (B(own=H) - B(own=A)) / 2.

Honest scores: `out_of_fold` predicts each IDSSE match with models trained on the other six.
Owner: person 4.
"""

import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import ballfilter, config, features
from possession.clean import clean_atd

CACHE = config.REPO_ROOT / "outputs" / "cache" / "features"


def make_model(params: dict) -> HistGradientBoostingClassifier:
    p = params["model"]
    return HistGradientBoostingClassifier(
        learning_rate=p["learning_rate"], max_iter=p["max_iter"], max_leaf_nodes=p["max_leaf_nodes"],
        min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2"], early_stopping=False,
        random_state=p["seed"])


def prepare(tr, halves, pitch, params: dict, gt=None) -> dict:
    """Stages 2-3 on cleaned tracking: everything the models and the decoder need."""
    bt = ballfilter.run(tr, params)
    blocks = features.build(tr, bt, params, halves, pitch)
    return {
        "bt": bt, "halves": halves, "pitch": pitch, "gt": gt,
        "dead": features.dead_matrix(blocks).to_numpy(np.float32),
        "own_H": features.team_matrix(blocks, "H").to_numpy(np.float32),
        "own_A": features.team_matrix(blocks, "A").to_numpy(np.float32),
    }


def prepare_idsse(match_id: str, params: dict, refresh: bool = False) -> dict:
    """prepare() for one degraded IDSSE match, cached on disk (local only, gitignored)."""
    from possession.external.bench import bench_gt
    from possession.external.idsse import load_degraded

    path = CACHE / f"{match_id}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    m = load_degraded(match_id)
    prep = prepare(clean_atd(m.tracking, params), m.halves, m.pitch, params, gt=bench_gt(m))
    prep["id"] = match_id
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(prep, fh)
    return prep


def rows(prep: dict, step: int) -> np.ndarray:
    idx = np.flatnonzero(prep["gt"] != "X")
    return idx[::step]


def training_sets(preps: list[dict], params: dict):
    step = params["model"]["row_step"]
    Xa, ya, Xb, yb = [], [], [], []
    for p in preps:
        r = rows(p, step)
        Xa.append(p["dead"][r])
        ya.append(p["gt"][r] == "D")
        live = r[np.isin(p["gt"][r], ["H", "A"])]
        Xb += [p["own_H"][live], p["own_A"][live]]
        yb += [p["gt"][live] == "H", p["gt"][live] == "A"]
    return np.concatenate(Xa), np.concatenate(ya), np.concatenate(Xb), np.concatenate(yb)


def fit(preps: list[dict], params: dict):
    Xa, ya, Xb, yb = training_sets(preps, params)
    return make_model(params).fit(Xa, ya), make_model(params).fit(Xb, yb)


def predict(models, prep: dict) -> tuple[np.ndarray, np.ndarray]:
    """(dead log-odds, H-vs-A log-odds) per frame."""
    a, b = models
    dead = a.decision_function(prep["dead"])
    team = (b.decision_function(prep["own_H"]) - b.decision_function(prep["own_A"])) / 2
    return dead, team


def out_of_fold(preps: list[dict], params: dict, log=print) -> None:
    """Adds prep['oof_dead'] and prep['oof_team'] from models that never saw that match."""
    for i, p in enumerate(preps):
        models = fit([q for j, q in enumerate(preps) if j != i], params)
        p["oof_dead"], p["oof_team"] = predict(models, p)
        g = p["gt"]
        live = np.isin(g, ["H", "A"])
        acc_d = ((p["oof_dead"] > 0) == (g == "D"))[g != "X"].mean()
        acc_t = ((p["oof_team"] > 0) == (g == "H"))[live].mean()
        log(f"  {p['id']}: dead/alive acc {acc_d:.3f}  team acc (live) {acc_t:.3f}", flush=True)
