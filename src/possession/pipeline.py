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


def check_data() -> int:
    missing = [p for src in ("tactical", "broadcast") for p in config.asset_paths(src).values() if not p.exists()]
    for p in missing:
        print(f"missing: {p}")
    if not config.PUBLIC_GT.exists():
        print(f"missing public GT: {config.PUBLIC_GT} (clone the organisers' repo, see vendor/README.md)")
    print("ok" if not missing else f"{len(missing)} data files missing (data dir: {config.DATA_DIR})")
    return 1 if missing else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="possession")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--out", type=Path, default=Path("outputs/submission.csv"))
    p_val = sub.add_parser("validate")
    p_val.add_argument("path", type=Path)
    sub.add_parser("check-data")
    args = parser.parse_args()

    if args.cmd == "run":
        run(args.out)
    elif args.cmd == "validate":
        problems = submission.validate(args.path)
        for p in problems:
            print(p)
        print(submission.summary(args.path))
        sys.exit(1 if any(not p.startswith("warning") for p in problems) else 0)
    elif args.cmd == "check-data":
        sys.exit(check_data())


if __name__ == "__main__":
    main()
