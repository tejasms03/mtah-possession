"""Write and check the submission file.

The checks mirror SUBMISSION.md. The organisers' `evaluation/validate_submission.py` remains the
authority; `make submission` runs ours, and the final upload should also go through theirs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from possession.config import LABELS, N_FRAMES, graded_mask


def write(labels: np.ndarray, path: Path) -> None:
    """Write `frame,label` for every video frame. Ungraded frames are forced to X."""
    labels = np.asarray(labels, dtype="<U1")
    if labels.shape != (N_FRAMES,):
        raise ValueError(f"expected {N_FRAMES} labels, got {labels.shape}")
    labels = labels.copy()
    labels[~graded_mask()] = "X"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))


def validate(path: Path) -> list[str]:
    """Return a list of problems; empty means the file is valid."""
    problems = []
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = [c.strip().lower() for c in df.columns]
    if cols != ["frame", "label"]:
        return [f"header must be 'frame,label', got {list(df.columns)}"]
    if len(df) != N_FRAMES:
        problems.append(f"expected {N_FRAMES} rows, got {len(df)}")
    frames = pd.to_numeric(df.iloc[:, 0].str.strip(), errors="coerce")
    if frames.isna().any() or not np.array_equal(frames.to_numpy(), np.arange(len(df))):
        problems.append("frame column must be 0..199318 in order")
    labels = df.iloc[:, 1].str.strip().str.upper()
    if (labels == "").any():
        problems.append(f"{(labels == '').sum()} blank labels (write X to abstain)")
    bad = ~labels.isin(LABELS) & (labels != "")
    if bad.any():
        problems.append(f"invalid labels: {sorted(labels[bad].unique())[:5]}")
    if not problems:
        graded = labels.to_numpy()[graded_mask()]
        n_x = int((graded == "X").sum())
        if n_x:
            problems.append(f"warning: {n_x} X labels on graded frames (costs macroF1, never helps)")
    return problems


def summary(path: Path) -> str:
    labels = pd.read_csv(path, dtype=str)["label"].str.strip().str.upper().to_numpy()
    g = labels[graded_mask()]
    counts = {lab: int((g == lab).sum()) for lab in LABELS}
    runs = np.flatnonzero(g[1:] != g[:-1]) + 1
    starts = np.concatenate([[0], runs])
    n_poss = int(np.isin(g[starts], ["H", "A"]).sum())
    return f"graded frames {counts}, possessions (runs of H or A): {n_poss}"
