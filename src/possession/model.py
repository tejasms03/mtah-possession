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

import copy
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
        max_features=p.get("max_features", 1.0), random_state=p["seed"])


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


def external_matches(params: dict | None = None) -> list[str]:
    """Every labelled outside match: 7 IDSSE (J03...), 3 Metrica samples (MET...), and the downloaded
    SkillCorner matches (SC...) when `external.skillcorner` is on (team-side experts only)."""
    from possession.external import idsse, metrica, skillcorner

    params = params or config.load_params()
    ids = list(idsse.MATCHES) + list(metrica.MATCHES)
    if params.get("external", {}).get("skillcorner"):
        ids += skillcorner.match_ids()
    return ids


def team_only(prep: dict) -> bool:
    """Outside matches whose labels only cover possession by a team (no dead ball): never rows for the
    dead-ball models."""
    return str(prep.get("id", "")).startswith("SC")


def degrade_params(params: dict, variant: str = "") -> dict:
    """params with idsse_degrade overridden by degrade_variants[variant] ("" = as is)."""
    if not variant:
        return params
    q = copy.deepcopy(params)
    q["idsse_degrade"].update(params["degrade_variants"][variant])
    return q


def load_external(match_id: str, params: dict | None = None):
    from possession.external import idsse, metrica, skillcorner

    if match_id.startswith("SC"):
        return skillcorner.load_degraded(match_id)
    return (metrica if match_id.startswith("MET") else idsse).load_degraded(match_id, params)


def prepare_idsse(match_id: str, params: dict, refresh: bool = False, variant: str = "") -> dict:
    """prepare() for one degraded outside match (IDSSE or Metrica), cached on disk (local only,
    gitignored). variant: a degrade_variants entry (own cache file)."""
    from possession.external.bench import bench_gt

    dp = degrade_params(params, variant) if variant else None

    def load_degraded(mid):
        return load_external(mid, dp)

    tag = f"_{variant}" if variant else ""
    path = CACHE / (f"{match_id}{tag}_xt.pkl" if params["features"].get("extra_team") else f"{match_id}{tag}.pkl")
    if path.exists() and not refresh:
        with open(path, "rb") as fh:
            prep = pickle.load(fh)
        # labels depend on idsse_bench.merge_blip_s, which the cache name does not record
        prep["gt"] = bench_gt(load_degraded(match_id))
        return prep
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


def rows(prep: dict, step: int, offset: int = 0) -> np.ndarray:
    idx = np.flatnonzero(prep["gt"] != "X")
    return idx[offset::step]


def training_sets(preps: list[dict], params: dict):
    step = params["model"]["row_step"]
    Xa, ya, Xb, yb = [], [], [], []
    for p in preps:
        r = rows(p, step, params["model"].get("row_offset", 0))
        if not team_only(p):
            Xa.append(p["dead"][r][:, dead_keep(params)])
            ya.append(p["gt"][r] == "D")
        live = r[np.isin(p["gt"][r], ["H", "A"])]
        Xb += [p["own_H"][live], p["own_A"][live]]
        yb += [p["gt"][live] == "H", p["gt"][live] == "A"]
    return np.concatenate(Xa), np.concatenate(ya), np.concatenate(Xb), np.concatenate(yb)


def own_match_rows(prep: dict, tags, dead_logodds: np.ndarray, params: dict, exclude=None):
    """Weak dead/alive labels from our own match (all algorithmic, no hand labels).

    dead: within +-tag_window_s of each Smart Tagging SET PIECE start (the tag sits inside the
          stoppage: both tags in the public window do).
    live: ball seen and moving faster than live_ball_speed_mps, a detected touch within 1 s, at
          least live_clear_s from any set-piece tag, and the outside-data model already says live.
    The public ground-truth window is left out so the public check stays honest.
    Returns (frame indices, labels as bool dead)."""
    from possession.config import FPS, PUBLIC_GT_FRAMES

    p = params["model"]["own"]
    n = len(dead_logodds)
    graded = np.zeros(n, dtype=bool)
    for _, a, b in prep["halves"]:
        graded[a : b + 1] = True
    ex = PUBLIC_GT_FRAMES if exclude is None else exclude
    if ex:
        graded[ex[0] : ex[1] + 1] = False

    starts = tags.loc[tags["code"] == "SET PIECES", "start_frame"].to_numpy()
    near_tag = np.zeros(n, dtype=bool)
    dead = np.zeros(n, dtype=bool)
    w, clear = int(p["tag_window_s"] * FPS), int(p["live_clear_s"] * FPS)
    for f in starts:
        dead[max(0, f - w) : min(n, f + w + 1)] = True
        near_tag[max(0, f - clear) : min(n, f + clear + 1)] = True
    dead &= graded

    bt = prep["bt"]
    touch_near = np.convolve(bt.touch.astype(float), np.ones(2 * FPS + 1), "same") > 0
    live = (graded & ~near_tag & bt.observed & (np.nan_to_num(bt.speed) > p["live_ball_speed_mps"])
            & touch_near & (dead_logodds < np.log(p["live_max_pdead"] / (1 - p["live_max_pdead"]))))
    step = params["model"]["row_step"]
    idx_dead, idx_live = np.flatnonzero(dead)[::step], np.flatnonzero(live)[::step]
    idx = np.concatenate([idx_dead, idx_live])
    y = np.concatenate([np.ones(len(idx_dead), bool), np.zeros(len(idx_live), bool)])
    return idx, y


def own_team_rows(prep: dict, tags, params: dict, include_public: bool = False, exclude=None):
    """Weak whose-ball labels from our own match: frames inside a run of at least
    `min_touches` consecutive touches by one team (from its first to its last touch of the run),
    ball seen on the frame, at least `clear_s` from any set-piece tag, public window left out.
    Returns (frame indices, team labels "H"/"A")."""
    from possession.config import FPS, PUBLIC_GT_FRAMES

    p = params["model"]["own_team"]
    bt = prep["bt"]
    n = len(bt.x)
    ok = np.zeros(n, dtype=bool)
    for _, a, b in prep["halves"]:
        ok[a : b + 1] = True
    ex = PUBLIC_GT_FRAMES if exclude is None else exclude
    if not include_public and ex:
        ok[ex[0] : ex[1] + 1] = False
    clear = int(p["clear_s"] * FPS)
    for f in tags.loc[tags["code"] == "SET PIECES", "start_frame"]:
        ok[max(0, f - clear) : min(n, f + clear + 1)] = False
    ok &= bt.observed

    ti = np.flatnonzero(bt.touch)
    tt = bt.touch_team[ti]
    label = np.full(n, "", dtype="<U1")
    k = 0
    while k < len(ti):
        j = k
        while j + 1 < len(ti) and tt[j + 1] == tt[k] and ti[j + 1] - ti[j] <= p["max_gap_s"] * FPS:
            j += 1
        if j - k + 1 >= p["min_touches"] and tt[k] in ("H", "A"):
            label[ti[k] : ti[j] + 1] = tt[k]
        k = j + 1
    idx = np.flatnonzero(ok & (label != ""))[:: params["model"]["row_step"]]
    return idx, label[idx]


def fit(preps: list[dict], params: dict, own=None, own_team=None):
    """(model A, model B, model A-full or None, model A-own or None).

    Model A-full is a second dead-ball model on the Metrica matches only, with every feature
    including ball motion: Metrica's ball moves at stoppages like ours, so there a hard kick out
    of stillness is a genuine restart cue.
    own: optional (dead matrix rows, bool labels) from our own match (own_match_rows), added to
    both dead models with weight model.own.weight, and (combine mean3/max) trained alone as
    model A-own."""
    Xa, ya, Xb, yb = training_sets(preps, params)
    wa = np.ones(len(ya))
    keep = dead_keep(params)
    if own is not None:
        Xo, yo = own
        Xa = np.concatenate([Xa, Xo[:, keep]])
        ya = np.concatenate([ya, yo])
        wa = np.concatenate([wa, np.full(len(yo), params["model"]["own"]["weight"])])
    wb = np.ones(len(yb))
    if own_team is not None:
        # own_team: (own_H rows, own_A rows, labels) -> shown from both sides, like the outside data
        XH, XA, lab = own_team
        Xb = np.concatenate([Xb, XH, XA])
        yb = np.concatenate([yb, lab == "H", lab == "A"])
        wb = np.concatenate([wb, np.full(2 * len(lab), params["model"]["own_team"]["weight"])])
    # weights for B only when own-team rows are used: passing all-ones weights changes how sklearn
    # samples rows to place its bin thresholds (>200k rows), so the model differs from plain fit
    a = make_model(params).fit(Xa, ya, sample_weight=wa)
    b = make_model(params).fit(Xb, yb, sample_weight=wb if own_team is not None else None)
    full = None
    met = [p for p in preps if str(p.get("id", "")).startswith("MET")]
    if params["model"]["dead_metrica_full"] and met:
        step = params["model"]["row_step"]
        off = params["model"].get("row_offset", 0)
        X = np.concatenate([p["dead"][rows(p, step, off)] for p in met])
        y = np.concatenate([p["gt"][rows(p, step, off)] == "D" for p in met])
        w = np.ones(len(y))
        if own is not None:
            X, y = np.concatenate([X, Xo]), np.concatenate([y, yo])
            w = np.concatenate([w, np.full(len(yo), params["model"]["own"]["weight"])])
        full = make_model(params).fit(X, y, sample_weight=w)
    own_model = None
    if own is not None and params["model"]["own"]["combine"] in ("mean3", "max"):
        q = copy.deepcopy(params)
        q["model"]["min_samples_leaf"] = params["model"]["own"]["min_samples_leaf"]
        own_model = make_model(q).fit(Xo, yo)
    return a, b, full, own_model


def predict(models, prep: dict, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(dead log-odds, H-vs-A log-odds) per frame."""
    a, b, full, own_model = models if len(models) == 4 else (*models, None)
    dead = a.decision_function(prep["dead"][:, dead_keep(params)])
    if full is not None:
        dead = (dead + full.decision_function(prep["dead"])) / 2
    if own_model is not None:
        mine = own_model.decision_function(prep["dead"])
        if params["model"]["own"]["combine"] == "max":      # either-or: dead if either model says so
            dead = np.maximum(dead, mine)
        else:                                               # mean3: our-match model as a third vote
            dead = (2 * dead + mine) / 3
    team = (b.decision_function(prep["own_H"]) - b.decision_function(prep["own_A"])) / 2
    return dead, team


def out_of_fold(preps: list[dict], params: dict, log=print) -> None:
    """Adds prep['oof_dead'] and prep['oof_team'] from models that never saw that match."""
    from possession.progress import Progress

    bar = Progress("leave-one-out", len(preps))
    for i, p in enumerate(preps):
        models = fit([q for j, q in enumerate(preps) if j != i], params)
        p["oof_dead"], p["oof_team"] = predict(models, p, params)
        g = p["gt"]
        live = np.isin(g, ["H", "A"])
        acc_d = ((p["oof_dead"] > 0) == (g == "D"))[g != "X"].mean()
        acc_t = ((p["oof_team"] > 0) == (g == "H"))[live].mean()
        bar.step(f"{p['id']}: dead/alive acc {acc_d:.3f}  team acc (live) {acc_t:.3f}")


def member_params(params: dict, k: int) -> dict:
    """Ensemble member k: its own seed, its own slice of frames (offset k of every row_step-th),
    and a random share of features per split. With ensemble size 1 this is exactly the plain model."""
    q = copy.deepcopy(params)
    n = params["model"]["ensemble"]["size"]
    if n > 1:
        q["model"]["seed"] = params["model"]["seed"] + k
        q["model"]["row_offset"] = k % params["model"]["row_step"]
        q["model"]["max_features"] = params["model"]["ensemble"]["max_features"]
    return q


def fit_ensemble(preps: list[dict], params: dict, own=None, own_team=None, label: str = "train") -> list:
    from possession.progress import Progress

    n = params["model"]["ensemble"]["size"]
    bar = Progress(label, n)
    members = []
    for k in range(n):
        members.append(fit(preps, member_params(params, k), own, own_team))
        bar.step(f"model {k + 1}/{n}")
    return members


def predict_ensemble(members: list, prep: dict, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the members' dead and team log-odds."""
    out = [predict(m, prep, member_params(params, k)) for k, m in enumerate(members)]
    return np.mean([d for d, _ in out], axis=0), np.mean([t for _, t in out], axis=0)


# -- change detector (council expert #4): "does control switch between teams around here?" ------

SWITCH_LAGS = (12, 25, 50)            # before/after offsets in frames: 0.5, 1, 2 s


def switch_matrix(prep: dict, idx: np.ndarray) -> np.ndarray:
    """Inputs for the change detector at frames idx: the game block plus, for each lag k, the size
    of the change in H-minus-A team features from k frames before to k frames after (absolute
    values, so the expert is team-blind). Only degraded inputs our tracking also has."""
    d = prep["own_H"] - prep["own_A"]
    n = len(d)
    parts = [prep["dead"][idx]]
    for k in SWITCH_LAGS:
        parts.append(np.abs(d[np.minimum(idx + k, n - 1)] - d[np.maximum(idx - k, 0)]))
    return np.nan_to_num(np.hstack(parts), nan=0.0).astype(np.float32)


def switch_target(gt: np.ndarray, half_window: int) -> np.ndarray:
    """True within half_window frames of a direct H<->A change in the labels."""
    sw = np.flatnonzero(np.isin(gt[1:], ["H", "A"]) & np.isin(gt[:-1], ["H", "A"]) & (gt[1:] != gt[:-1])) + 1
    near = np.zeros(len(gt), bool)
    for s in sw:
        near[max(0, s - half_window) : s + half_window] = True
    return near


def fit_switch(preps: list[dict], params: dict):
    """Change detector trained on outside matches only (live frames, every row_step-th)."""
    p = params["decode"]
    X, y = [], []
    for q in preps:
        r = rows(q, params["model"]["row_step"], 2)
        r = r[np.isin(q["gt"][r], ["H", "A"])]
        X.append(switch_matrix(q, r))
        y.append(switch_target(q["gt"], p["switch_half_window"])[r])
    m = make_model(params)
    m.fit(np.concatenate(X), np.concatenate(y))
    return m


def predict_switch(m, prep: dict, chunk: int = 20000) -> np.ndarray:
    n = len(prep["dead"])
    return np.concatenate([m.decision_function(switch_matrix(prep, np.arange(a, min(n, a + chunk))))
                           for a in range(0, n, chunk)])
