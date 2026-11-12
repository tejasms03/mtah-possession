"""Score pipelines on the 7 degraded IDSSE matches with the organisers' possession_metrics.py.

    uv run possession bench                       # every registered pipeline
    uv run possession bench baseline              # chosen ones

A pipeline is any callable (IdsseMatch) -> per-frame label array ("H"/"A"/"D"/"X"). Register it
in PIPELINES. Results go to outputs/bench/<pipeline>.csv: one row per match, then the mean.

Read the numbers with two caveats:
- IDSSE labels are DFL's BallStatus/BallPossession, which is close to touch-by-touch (median
  possession ~3 s on J03WMX, BAR-ATH public GT mean 14.2 s), so the bench rewards some flicker
  that the BAR-ATH ground truth punishes.
- There is no Smart Tagging for these matches, so set-piece evidence cannot be exercised here.
"""

import sys

import numpy as np

from possession.config import REPO_ROOT, VENDOR_DIR
from possession.external.idsse import MATCHES, IdsseMatch, load_degraded

BENCH_DIR = REPO_ROOT / "outputs" / "bench"
COLS = ("spellF1", "macroF1", "poss_pred", "poss_gt", "dur_pred_s", "dur_gt_s", "F1_D")


def _vendor(module: str):
    for sub in ("evaluation", "starter"):
        path = str(VENDOR_DIR / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    return __import__(module)


# -- pipelines --------------------------------------------------------------------------------

def upstream_baseline(m: IdsseMatch, window: int = 50) -> np.ndarray:
    """vendor/mtah/starter/baseline.py on arrays: nearest tracked player's team, vote +/-window.

    The nearest-player step is a vectorised copy of nearest_player_labels() (that function reads
    the raw BAR-ATH file); the smoothing is the organisers' own smooth(), imported unchanged.
    """
    ub = _vendor("baseline")
    t = m.tracking
    d = np.hypot(t.px - t.ball[:, :1], t.py - t.ball[:, 1:])
    d = np.where(np.isnan(d), np.inf, d)
    j = np.argmin(d, 1)
    seen = np.isfinite(d[np.arange(len(d)), j])
    code = np.where(t.team[j] == "H", ub.HOME, ub.AWAY)
    return ub.smooth(np.where(seen, code, ub.NOBODY).astype(np.int8), window)


def merge_blips(gt: np.ndarray, min_frames: int) -> np.ndarray:
    """IDSSE marks possession almost touch by touch. A spell of one team shorter than min_frames
    with the other team on both sides (a deflection, a touch without control) is given to that
    other team, which is what the challenge counts as no change of possession."""
    out = gt.copy()
    edges = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(gt)]])
    for k in range(1, len(starts) - 1):
        s, e = starts[k], ends[k]
        before, here, after = out[starts[k - 1]], out[s], gt[starts[k + 1]]
        if here in ("H", "A") and e - s < min_frames and before == after and before in ("H", "A") and before != here:
            out[s:e] = before
    return out


def fallback(m: IdsseMatch) -> np.ndarray:
    """Our rule-only pipeline (stages 1-6, no model, no Smart Tagging on IDSSE)."""
    from possession.clean import clean_atd
    from possession.config import load_params
    from possession.pipeline import label_tracking

    params = load_params()
    return label_tracking(clean_atd(m.tracking, params), m.halves, params, pitch=m.pitch)


PIPELINES = {
    "baseline": upstream_baseline,
    "fallback": fallback,
}


# -- scoring ----------------------------------------------------------------------------------

def bench_gt(m: IdsseMatch) -> np.ndarray:
    from possession.config import FPS, load_params

    return merge_blips(m.gt, int(round(load_params()["idsse_bench"]["merge_blip_s"] * FPS)))


def score(m: IdsseMatch, labels: np.ndarray, gt: np.ndarray | None = None) -> dict:
    pm = _vendor("possession_metrics")
    r = pm.evaluate(pm.label_codes(bench_gt(m) if gt is None else gt), pm.label_codes(labels))
    return {
        "match": m.match_id,
        "spellF1": r["spell_f1_mean"],
        "macroF1": r["macro_f1"],
        "poss_pred": r["n_pred_spells"],
        "poss_gt": r["n_gt_spells"],
        "dur_pred_s": r["mean_spell_duration_s"],
        "dur_gt_s": r["gt_mean_spell_duration_s"],
        "F1_D": r["per_class"]["D"]["f1"],
    }


def run(name: str, match_ids=tuple(MATCHES)) -> list[dict]:
    fn = PIPELINES[name]
    rows = []
    for mid in match_ids:
        m = load_degraded(mid)
        rows.append(score(m, fn(m)))
    rows.append({"match": "MEAN", **{c: float(np.mean([r[c] for r in rows])) for c in COLS}})
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCH_DIR / f"{name}.csv", "w") as fh:
        fh.write("match," + ",".join(COLS) + "\n")
        for r in rows:
            fh.write(r["match"] + "," + ",".join(f"{r[c]:.4f}" for c in COLS) + "\n")
    return rows


def print_table(name: str, rows: list[dict]) -> None:
    print(f"\n{name}")
    print(f"  {'match':<8} {'spellF1':>7} {'macroF1':>7} {'poss':>11} {'dur s':>11} {'F1_D':>6}")
    for r in rows:
        print(f"  {r['match']:<8} {r['spellF1']:7.4f} {r['macroF1']:7.4f} "
              f"{r['poss_pred']:5.0f}/{r['poss_gt']:<5.0f} {r['dur_pred_s']:5.1f}/{r['dur_gt_s']:<5.1f} "
              f"{r['F1_D']:6.3f}")


def main(pipelines=None, match_ids=None) -> None:
    pipelines = pipelines or list(PIPELINES)
    match_ids = match_ids or list(MATCHES)
    summary = []
    for name in pipelines:
        rows = run(name, match_ids)
        print_table(name, rows)
        summary.append((name, rows[-1]))
    print("\nmean over matches (poss = predicted/true)")
    for name, r in summary:
        print(f"  {name:<12} spellF1 {r['spellF1']:.4f}  macroF1 {r['macroF1']:.4f}  "
              f"poss {r['poss_pred']:.0f}/{r['poss_gt']:.0f}")
