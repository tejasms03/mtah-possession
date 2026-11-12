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


def external_matches() -> list[str]:
    """Every labelled outside match: 7 IDSSE (J03...) and 3 Metrica samples (MET...)."""
    from possession.external import idsse, metrica

    return list(idsse.MATCHES) + list(metrica.MATCHES)


def load_external(match_id: str):
    from possession.external import idsse, metrica

    return (metrica if match_id.startswith("MET") else idsse).load_degraded(match_id)


def prepare_idsse(match_id: str, params: dict, refresh: bool = False) -> dict:
    """prepare() for one degraded outside match (IDSSE or Metrica), cached on disk (local only,
    gitignored)."""
    from possession.external.bench import bench_gt

    load_degraded = load_external
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


_DEAD_COLS = None


def dead_columns(params: dict) -> list[str]:
    """Model A's column names (built once on a tiny synthetic match)."""
    global _DEAD_COLS
    if _DEAD_COLS is None:
        from possession.load import Tracking

        n = 1200                      # longer than the 40 s window
        rng = np.random.default_rng(0)
        tr = Tracking(ball=rng.uniform(10, 60, (n, 2)), px=rng.uniform(5, 100, (n, 4)),
                      py=rng.uniform(5, 60, (n, 4)), ps=np.ones((n, 4)), team=np.array(["H", "H", "A", "A"]),
                      vx=np.zeros((n, 4)), vy=np.zeros((n, 4)))
        bt = ballfilter.run(tr, params)
        _DEAD_COLS = list(features.dead_matrix(features.build(tr, bt, params, [(1, 0, n - 1)], (105.0, 68.0))).columns)
    return _DEAD_COLS


def dead_keep(params: dict) -> np.ndarray:
    """Columns model A may use. DFL freezes the ball whenever play is dead (speed exactly 0 on
    100% of dead frames, even in the clean files), so any feature of ball motion or ball handling
    gives the answer away on IDSSE and fails on our ATD, where the ball keeps moving at stoppages."""
    drop = tuple(params["model"]["dead_drop_prefixes"])
    return np.array([not c.startswith(drop) for c in dead_columns(params)])


def rows(prep: dict, step: int) -> np.ndarray:
    idx = np.flatnonzero(prep["gt"] != "X")
    return idx[::step]


def training_sets(preps: list[dict], params: dict):
    step = params["model"]["row_step"]
    Xa, ya, Xb, yb = [], [], [], []
    for p in preps:
        r = rows(p, step)
        Xa.append(p["dead"][r][:, dead_keep(params)])
        ya.append(p["gt"][r] == "D")
        live = r[np.isin(p["gt"][r], ["H", "A"])]
        Xb += [p["own_H"][live], p["own_A"][live]]
        yb += [p["gt"][live] == "H", p["gt"][live] == "A"]
    return np.concatenate(Xa), np.concatenate(ya), np.concatenate(Xb), np.concatenate(yb)


def fit(preps: list[dict], params: dict):
    """(model A, model B, model A-full or None). Model A-full is a second dead-ball model on the
    Metrica matches only, with every feature including ball motion: Metrica's ball moves at
    stoppages like ours, so there a hard kick out of stillness is a genuine restart cue."""
    Xa, ya, Xb, yb = training_sets(preps, params)
    a, b = make_model(params).fit(Xa, ya), make_model(params).fit(Xb, yb)
    full = None
    met = [p for p in preps if str(p.get("id", "")).startswith("MET")]
    if params["model"]["dead_metrica_full"] and met:
        step = params["model"]["row_step"]
        X = np.concatenate([p["dead"][rows(p, step)] for p in met])
        y = np.concatenate([p["gt"][rows(p, step)] == "D" for p in met])
        full = make_model(params).fit(X, y)
    return a, b, full


def predict(models, prep: dict, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(dead log-odds, H-vs-A log-odds) per frame."""
    a, b, full = models
    dead = a.decision_function(prep["dead"][:, dead_keep(params)])
    if full is not None:
        dead = (dead + full.decision_function(prep["dead"])) / 2
    team = (b.decision_function(prep["own_H"]) - b.decision_function(prep["own_A"])) / 2
    return dead, team


def out_of_fold(preps: list[dict], params: dict, log=print) -> None:
    """Adds prep['oof_dead'] and prep['oof_team'] from models that never saw that match."""
    for i, p in enumerate(preps):
        models = fit([q for j, q in enumerate(preps) if j != i], params)
        p["oof_dead"], p["oof_team"] = predict(models, p, params)
        g = p["gt"]
        live = np.isin(g, ["H", "A"])
        acc_d = ((p["oof_dead"] > 0) == (g == "D"))[g != "X"].mean()
        acc_t = ((p["oof_team"] > 0) == (g == "H"))[live].mean()
        log(f"  {p['id']}: dead/alive acc {acc_d:.3f}  team acc (live) {acc_t:.3f}", flush=True)
