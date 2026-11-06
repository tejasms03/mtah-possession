"""Load the provided assets onto the tactical video-frame index (row i = video frame i).

Untested against the real files until someone with the bundle runs `possession check-data`.
Alignment rules from DATA.md:
  ATD           frame_id 1..199319   -> video_frame = frame_id - 1
  Smart Tagging seconds from start   -> video_frame = round(seconds * 25)
  homography    frame 0..199318      -> already a video frame
"""

import numpy as np
import pandas as pd

from possession.config import FPS, N_FRAMES, PITCH_LENGTH_M, PITCH_WIDTH_M, asset_paths


def load_atd(source: str = "tactical") -> tuple[pd.DataFrame, dict[str, str]]:
    """ATD as one row per video frame, positions in metres.

    Returns (df, slot_team). df columns: ball_x, ball_y, then <slot>_x, <slot>_y, <slot>_s.
    slot_team maps slot id -> "H" or "A" (from the metadata team membership).
    """
    from kloppy import metrica

    paths = asset_paths(source)
    ds = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"])
    df = ds.to_df()
    df.index = df["frame_id"].astype(int) - 1
    df = df.reindex(range(N_FRAMES))

    xcols = [c for c in df.columns if c.endswith("_x")]
    ycols = [c for c in df.columns if c.endswith("_y")]
    df[xcols] = df[xcols] * PITCH_LENGTH_M
    df[ycols] = df[ycols] * PITCH_WIDTH_M

    home, away = ds.metadata.teams
    slot_team = {p.player_id: "H" for p in home.players}
    slot_team |= {p.player_id: "A" for p in away.players}
    # TODO(phase 0): confirm kloppy returns normalised 0-1 here and that teams[0] is Barcelona.
    return df, slot_team


def load_smart_tagging() -> pd.DataFrame:
    """Smart Tagging intervals with start_frame / end_frame columns added."""
    from kloppy import sportscode

    ev = sportscode.load(asset_paths()["smart_tagging"]).to_df()
    ev["start_frame"] = (ev["timestamp"].dt.total_seconds() * FPS).round().astype(int)
    ev["end_frame"] = (ev["end_timestamp"].dt.total_seconds() * FPS).round().astype(int)
    ev["code"] = ev["code"].str.upper()
    return ev


def load_homography(source: str = "tactical") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(H[n,3,3], reprojection_error[n], valid[n]). Only needed for visual debugging."""
    df = pd.read_csv(asset_paths(source)["homography"])
    H = df.iloc[:, 1:10].to_numpy(dtype=float).reshape(-1, 3, 3)
    return H, df["reprojection_error"].to_numpy(), df["valid"].to_numpy().astype(bool)


def load_public_gt() -> np.ndarray:
    from possession.config import PUBLIC_GT

    return pd.read_csv(PUBLIC_GT)["label"].str.strip().str.upper().to_numpy()
