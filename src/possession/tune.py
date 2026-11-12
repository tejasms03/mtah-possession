"""Stage 7: tune the decoder and dead-ball dials on spellF1 over the degraded IDSSE matches.

Stages 1-3 run once per match and are cached in memory; only the cheap steps (dead-ball
log-odds + decoder) rerun for each setting. Coordinate search: one dial at a time over its grid,
keeping the best mean spellF1, two sweeps. Never touches the BAR-ATH public window.
Owner: person 5.
"""

import copy
import json
import os

import numpy as np

from possession import ballfilter, config, deadball, decode
from possession.clean import clean_atd

GRID = {
    "decode.lambda_ha": [4, 6, 8, 12, 16, 24],
    "decode.lambda_ha_touch": [0.1, 0.25, 0.5, 1, 2],
    "decode.lambda_d": [0.5, 1, 2, 3, 4, 8],
    "decode.temperature": [0.03, 0.05, 0.1, 0.2, 0.4],
    "decode.min_possession_frames": [1, 3, 5, 10, 15],
    "decode.min_dead_frames": [5, 10, 25, 50],
    "decode.w_touch": [5, 10, 20, 40],
    "decode.w_control": [0, 0.5, 1],
    "dead_ball.speed_ref_mps": [1.6, 1.8, 2.0, 2.2, 2.4],
    "dead_ball.w_speed": [1, 2, 3, 5],
    "dead_ball.bias": [-3, -2, -1, 0, 1],
}


def get(params, key):
    a, b = key.split(".")
    return params[a][b]


def put(params, key, value):
    a, b = key.split(".")
    params[a][b] = value


def prepare(match_ids, params):
    from possession.external.bench import bench_gt
    from possession.external.idsse import load_degraded

    out = []
    for mid in match_ids:
        m = load_degraded(mid)
        tr = clean_atd(m.tracking, params)
        bt = ballfilter.run(tr, params)
        f = deadball.frame_features(tr, bt, params, m.pitch)
        out.append({"id": mid, "bt": bt, "f": f, "halves": m.halves, "gt": bench_gt(m)})
    return out


def evaluate(preps, params) -> tuple[float, list[float]]:
    """Mean spellF1 over preps. Uses the models' out-of-fold log-odds when a prep has them
    (model.out_of_fold), else the hand-set dead-ball rule."""
    from possession.external.bench import _vendor

    pm = _vendor("possession_metrics")
    scores = []
    for p in preps:
        if "oof_dead" in p:
            m = params["model"]
            dead = m["w_dead"] * p["oof_dead"] + m["dead_bias"]
            lab = decode.decode(p["bt"], dead, p["halves"], params, model_team_llr=p["oof_team"])
        else:
            dead = deadball.rule_dead_logodds(p["f"], params)
            lab = decode.decode(p["bt"], dead, p["halves"], params)
        r = pm.evaluate(pm.label_codes(p["gt"]), pm.label_codes(lab))
        scores.append(r["spell_f1_mean"])
    return float(np.mean(scores)), scores


MODEL_GRID = {
    "model.w_dead": [0.05, 0.1, 0.25, 0.5, 1, 2],
    "model.dead_bias": [-2, -1, 0, 1, 2],
    "model.w_team": [0.5, 1, 2, 4, 8],
    "decode.w_touch": [0, 2, 5, 10],
    "decode.lambda_ha": [2, 4, 8, 16, 32],
    "decode.lambda_ha_touch": [0.1, 0.5, 2, 8],
    "decode.lambda_d": [0.5, 1, 2, 4, 8],
    "decode.temperature": [0.05, 0.1, 0.2, 0.4],
    "decode.min_possession_frames": [1, 5, 10, 20],
    "decode.lambda_d_exit": [2, 4, 8, 16, 32],
    "decode.lambda_d_exit_touch": [0.25, 1, 2, 4],
}


def search(preps, params, grid=GRID, sweeps=2, log=print):
    best = copy.deepcopy(params)
    best_score, _ = evaluate(preps, best)
    log(f"start  spellF1 {best_score:.4f}", flush=True) if log is print else log(f"start  spellF1 {best_score:.4f}")
    for sweep in range(sweeps):
        for key, values in grid.items():
            for v in values:
                if v == get(best, key):
                    continue
                trial = copy.deepcopy(best)
                put(trial, key, v)
                s, _ = evaluate(preps, trial)
                if s > best_score + 1e-6:
                    best, best_score = trial, s
            log(f"sweep {sweep + 1}  {key} = {get(best, key)}  spellF1 {best_score:.4f}", flush=True) if log is print else log(f"sweep {sweep + 1}  {key} = {get(best, key)}  spellF1 {best_score:.4f}")
    return best, best_score


def main(match_ids=None) -> None:
    from possession.external.idsse import IDSSE_DIR, MATCHES

    params = config.load_params()
    have = os.listdir(IDSSE_DIR / "clean") if (IDSSE_DIR / "clean").exists() else []
    match_ids = match_ids or [m for m in MATCHES if f"{m}.npz" in have]
    print(f"tuning on {len(match_ids)} IDSSE matches: {' '.join(match_ids)}")
    preps = prepare(match_ids, params)
    best, score = search(preps, params)
    _, per = evaluate(preps, best)
    print("per match: " + "  ".join(f"{p['id']} {s:.3f}" for p, s in zip(preps, per)))
    chosen = {k: get(best, k) for k in GRID}
    print(json.dumps(chosen, indent=2))
    out = config.REPO_ROOT / "outputs" / "tune.json"
    out.write_text(json.dumps({"matches": match_ids, "spellF1": score, "params": chosen}, indent=2))
    print(f"wrote {out}")
