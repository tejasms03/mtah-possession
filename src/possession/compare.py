"""Compare two submission CSVs frame by frame on the graded frames; no ground truth needed.

`uv run possession compare a.csv b.csv` prints: label agreement overall and per label, a confusion
table, dead-ball shares, possession counts, and where they disagree most (per 5-minute block).
Prints summary numbers only, never per-frame labels.
"""

import numpy as np

from possession import config


def read_labels(path) -> np.ndarray:
    return np.array([line.split(",")[1] for line in open(path).read().split()[1:]])


def possessions(lab: np.ndarray) -> int:
    e = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return int(sum(lab[s] in ("H", "A") for s in np.r_[0, e]))


def main(a_path, b_path) -> None:
    g = config.graded_mask()
    a, b = read_labels(a_path)[g], read_labels(b_path)[g]
    print(f"graded frames: {g.sum()}   same label on {np.mean(a == b):.1%}")
    print("\n     " + "".join(f"{'B=' + k:>8}" for k in "HAD") + "   (rows: A's label)")
    for k in "HAD":
        row = [(a == k) & (b == j) for j in "HAD"]
        print(f"A={k}  " + "".join(f"{m.mean():8.1%}" for m in row))
    for name, x in (("A", a), ("B", b)):
        print(f"{name}: dead {np.mean(x == 'D'):.1%}  possessions {possessions(x)}  "
              f"Barca share of live {np.mean(x[x != 'D'] == 'H'):.1%}")
    live = (a != "D") & (b != "D")
    print(f"both live: same team on {np.mean(a[live] == b[live]):.1%}")
    print("\nagreement per 5-minute block:")
    for half, s, e in config.HALVES:
        blocks = [(x, min(x + 7500, e + 1)) for x in range(s, e + 1, 7500)]
        vals = [np.mean(read_labels(a_path)[p:q] == read_labels(b_path)[p:q]) for p, q in blocks]
        print(f"  half {half}: " + " ".join(f"{v:.0%}" for v in vals))
