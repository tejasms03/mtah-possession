"""Paths and match constants. Every number here comes from the organisers' DATA.md."""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR", REPO_ROOT / "data"))
VENDOR_DIR = REPO_ROOT / "vendor" / "mtah"
PUBLIC_GT = VENDOR_DIR / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
PARAMS_FILE = REPO_ROOT / "params.yaml"

GAME = "BAR-ATH"
FPS = 25
N_FRAMES = 199_319                      # tactical video; one submission row per frame
PITCH_LENGTH_M = 105.3                  # games.py value, measured from corrected tracking
PITCH_WIDTH_M = 68.0

# Graded video frames, inclusive: (half, first, last)
HALVES = [(1, 11_185, 90_651), (2, 110_808, 190_197)]
PUBLIC_GT_FRAMES = (11_185, 18_684)     # first 5 minutes of play, labelled in the public GT

LABELS = ("H", "A", "D", "X")


def asset_paths(source: str = "tactical") -> dict[str, Path]:
    """File names exactly as the bundle ships them."""
    base = DATA_DIR / source
    stem = f"{GAME}_{source}"
    paths = {
        "video": base / f"{stem}.mp4",
        "atd_meta": base / f"{stem}_atd.xml",
        "atd_raw": base / f"{stem}_atd.txt",
        "homography": base / f"{stem}_homography.csv",
    }
    if source == "tactical":
        paths["smart_tagging"] = base / f"{stem}_smart_tagging.xml"
    return paths


def graded_mask():
    """Boolean array over video frames, True where the frame is graded."""
    import numpy as np

    mask = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in HALVES:
        mask[first : last + 1] = True
    return mask


def half_of_frame():
    """Array over video frames: 1 or 2 inside a half, 0 elsewhere."""
    import numpy as np

    half = np.zeros(N_FRAMES, dtype=np.int8)
    for h, first, last in HALVES:
        half[first : last + 1] = h
    return half


def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)
