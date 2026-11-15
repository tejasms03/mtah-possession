"""Command line entry point: `uv run possession <command>`."""

import argparse
import sys
from pathlib import Path

import numpy as np

from possession import config, submission


def label_tracking(tr, halves, params, tags=None, pitch=None) -> np.ndarray:
    """Stages 2-6 on cleaned tracking (BAR-ATH or IDSSE). Rule-only fallback: no trained model."""
    from possession import ballfilter, deadball, decode

    bt = ballfilter.run(tr, params)
    feats = deadball.frame_features(tr, bt, params, pitch)
    dead = deadball.rule_dead_logodds(feats, params)
    hints = deadball.set_piece_hints(tags) if tags is not None else None
    return decode.decode(bt, dead, halves, params, hints)


def label_with_models(tr, halves, params, tags=None, pitch=None) -> np.ndarray:
    """Stages 2-6 with models A and B trained on all 7 degraded IDSSE matches."""
    from possession import deadball, decode, model

    pitch = pitch or (config.PITCH_LENGTH_M, config.PITCH_WIDTH_M)
    from possession.progress import Progress

    ids = model.external_matches(params)
    bar = Progress("load outside matches", len(ids))
    preps = []
    for m_id in ids:
        preps.append(model.prepare_idsse(m_id, params, variant=params["idsse_bench"].get("train_degrade", "")))
        bar.step(m_id)
    models = model.fit_ensemble(preps, params, label="train pass 1 (outside matches)")
    print("[features] building our match's features", flush=True)
    prep = model.prepare(tr, halves, pitch, params)
    dead, team = model.predict_ensemble(models, prep, params)
    if params["model"]["own"]["use"] and tags is not None:
        # second pass: retrain the dead models with weak labels from our own match
        idx, y = model.own_match_rows(prep, tags, dead, params)
        print(f"own-match rows: {int(y.sum())} dead, {int((~y).sum())} live", flush=True)
        own_team = None
        if params["model"]["own_team"]["use"]:
            ti, tl = model.own_team_rows(prep, tags, params)
            print(f"own-match team rows: {int((tl == 'H').sum())} H, {int((tl == 'A').sum())} A", flush=True)
            own_team = (prep["own_H"][ti], prep["own_A"][ti], tl)
        models = model.fit_ensemble(preps, params, own=(prep["dead"][idx], y), own_team=own_team,
                                    label="train pass 2 (+ our set-piece tags)")
        dead, team = model.predict_ensemble(models, prep, params)
    switch = None
    if params["decode"].get("w_switch", 0):
        print("[change detector] training on outside matches", flush=True)
        switch = model.predict_switch(model.fit_switch(preps, params), prep)
    m = params["model"]
    dead = m["w_dead"] * dead + m["dead_bias"]
    hints = deadball.set_piece_hints(tags) if tags is not None else None
    return decode.decode(prep["bt"], dead, halves, params, hints, model_team_llr=team, pitch=pitch,
                         switch_logodds=switch)


def run(out: Path, rules_only: bool = False) -> None:
    """Full pipeline on BAR-ATH -> submission CSV."""
    from possession.clean import clean_atd
    from possession.load import load_atd, load_smart_tagging

    params = config.load_params()
    tr = clean_atd(load_atd(), params)
    label = label_tracking if rules_only else label_with_models
    labels = label(tr, config.HALVES, params, tags=load_smart_tagging())
    submission.write(labels, out)
    print(f"wrote {out}")


def stage1(source: str = "tactical"):
    """Load and clean the ATD; print the cleaning report (record it in docs/DECISIONS.md)."""
    from possession.clean import clean_atd
    from possession.load import load_atd

    tr = clean_atd(load_atd(source), config.load_params())
    graded = config.graded_mask()
    print(f"slots: {tr.n_slots}  H: {(tr.team == 'H').sum()}  A: {(tr.team == 'A').sum()}  other: {(tr.team == '').sum()}")
    for k, v in tr.report.items():
        print(f"  {k}: {v}")
    print(f"  ball missing on graded frames: {100 * (~tr.ball_observed[graded]).mean():.1f}%")
    return tr


def check_data() -> int:
    missing = [p for src in ("tactical", "broadcast") for p in config.asset_paths(src).values() if not p.exists()]
    for p in missing:
        print(f"missing: {p}")
    if not config.PUBLIC_GT.exists():
        print(f"missing public GT: {config.PUBLIC_GT} (clone the organisers' repo, see vendor/README.md)")
    print("ok" if not missing else f"{len(missing)} data files missing (data dir: {config.DATA_DIR})")
    return 1 if missing else 0


def idsse(match_ids: list[str]) -> None:
    """Download, convert and degrade the IDSSE matches; print the calibration aggregates."""
    import subprocess

    from possession.external import idsse as ix

    for mid in match_ids or list(ix.MATCHES):
        if not (ix.CLEAN_DIR / f"{mid}.npz").exists():
            # One process per match: kloppy peaks at ~3.9 GB and does not give it all back.
            subprocess.run([sys.executable, "-m", "possession.external.idsse", mid], check=True)
        r = ix.degrade_report(ix.load_degraded(mid))
        print(f"{mid}: " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                     for k, v in r.items()))


def main() -> None:
    parser = argparse.ArgumentParser(prog="possession")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    p_run.add_argument("--rules-only", action="store_true", help="fallback without models A and B")
    p_val = sub.add_parser("validate")
    p_val.add_argument("path", type=Path)
    sub.add_parser("check-data")
    p_cmp = sub.add_parser("compare", help="frame-by-frame agreement between two submission CSVs")
    p_cmp.add_argument("a", type=Path)
    p_cmp.add_argument("b", type=Path)
    p_sel = sub.add_parser("select", help="choose between versions on the Metrica stand-in bench (no public window)")
    p_sel.add_argument("--versions", nargs="*", help="subset of versions (default: all)")
    p_sel.add_argument("--extra-team", action="store_true", help="idea 2 team features on")
    sub.add_parser("tune", help="stage 7: tune decoder and dead-ball dials on spellF1 over degraded IDSSE")
    p_ix = sub.add_parser("idsse", help="download + convert + degrade the IDSSE matches")
    p_ix.add_argument("matches", nargs="*")
    p_bench = sub.add_parser("bench", help="score pipelines on the degraded IDSSE matches")
    p_bench.add_argument("pipelines", nargs="*")
    p_bench.add_argument("--matches", nargs="*")
    p_clean = sub.add_parser("clean", help="stage 1 only: load + clean the ATD and print the report")
    p_clean.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    args = parser.parse_args()

    if args.cmd == "run":
        run(args.out, args.rules_only)
    elif args.cmd == "validate":
        problems = submission.validate(args.path)
        for p in problems:
            print(p)
        print(submission.summary(args.path))
        sys.exit(1 if any(not p.startswith("warning") for p in problems) else 0)
    elif args.cmd == "clean":
        stage1(args.source)
    elif args.cmd == "compare":
        from possession import compare

        compare.main(args.a, args.b)
    elif args.cmd == "select":
        from possession import selection

        selection.main(args.versions, args.extra_team)
    elif args.cmd == "tune":
        from possession import tune

        tune.main()
    elif args.cmd == "check-data":
        sys.exit(check_data())
    elif args.cmd == "idsse":
        idsse(args.matches)
    elif args.cmd == "bench":
        from possession.external import bench

        bench.main(args.pipelines, args.matches)


if __name__ == "__main__":
    main()
