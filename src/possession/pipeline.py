"""Command line entry point: `uv run possession <command>`."""

import argparse
import sys
from pathlib import Path

import numpy as np

from possession import config, submission


def run(out: Path) -> None:
    """Full pipeline. Until the stages exist, writes a placeholder (all H on graded frames)."""
    labels = np.full(config.N_FRAMES, "H", dtype="<U1")
    # TODO: load -> clean -> ballfilter -> features -> deadball/labeller/model -> decode
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
    p_val = sub.add_parser("validate")
    p_val.add_argument("path", type=Path)
    sub.add_parser("check-data")
    p_ix = sub.add_parser("idsse", help="download + convert + degrade the IDSSE matches")
    p_ix.add_argument("matches", nargs="*")
    p_bench = sub.add_parser("bench", help="score pipelines on the degraded IDSSE matches")
    p_bench.add_argument("pipelines", nargs="*")
    p_bench.add_argument("--matches", nargs="*")
    p_clean = sub.add_parser("clean", help="stage 1 only: load + clean the ATD and print the report")
    p_clean.add_argument("--source", default="tactical", choices=["tactical", "broadcast"])
    args = parser.parse_args()

    if args.cmd == "run":
        run(args.out)
    elif args.cmd == "validate":
        problems = submission.validate(args.path)
        for p in problems:
            print(p)
        print(submission.summary(args.path))
        sys.exit(1 if any(not p.startswith("warning") for p in problems) else 0)
    elif args.cmd == "clean":
        stage1(args.source)
    elif args.cmd == "check-data":
        sys.exit(check_data())
    elif args.cmd == "idsse":
        idsse(args.matches)
    elif args.cmd == "bench":
        from possession.external import bench

        bench.main(args.pipelines, args.matches)


if __name__ == "__main__":
    main()
