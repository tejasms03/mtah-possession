"""Load the provided assets onto the tactical video-frame index (row i = video frame i).

Alignment rules from DATA.md:
  ATD           frame_id 1..199319   -> video_frame = frame_id - 1
  Smart Tagging seconds from start   -> video_frame = round(seconds * 25)
  homography    frame 0..199318      -> already a video frame

ATD raw line format (checked on the broadcast file):
  <frame_id>:<x>,<y>,<speed>;<x>,<y>,<speed>;... (one triple per track slot) :<ball_x>,<ball_y>,<ball_z>
Coordinates are 0-1 of the pitch. Slot speed can be NaN while x,y are set. ball_z is always NaN.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from possession.config import FPS, N_FRAMES, PITCH_LENGTH_M, PITCH_WIDTH_M, asset_paths


@dataclass
class Tracking:
    """Positions in metres, origin at a pitch corner, one row per video frame.

    ball: (N, 2), NaN when missing. px, py, ps: (N, S) per track slot (x, y, speed).
    team: (S,) "H", "A" or "" (referee / unknown). Slots have no identity: never follow a slot.
    Stage 1 adds vx, vy (N, S) and ball_observed (N,).
    """

    ball: np.ndarray
    px: np.ndarray
    py: np.ndarray
    ps: np.ndarray
    team: np.ndarray
    vx: np.ndarray | None = None
    vy: np.ndarray | None = None
    ball_observed: np.ndarray | None = None
    report: dict = field(default_factory=dict)

    @property
    def n_slots(self) -> int:
        return self.px.shape[1]


def read_atd_raw(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the raw ATD txt. Returns (frame_id[F], players[F, S, 3], ball[F, 3]) in 0-1 units."""
    text = Path(path).read_text()
    first = text.split("\n", 1)[0]
    n_slots = first.split(":")[1].count(";") + 1
    table = pd.read_csv(
        io.StringIO(text.replace(":", ",").replace(";", ",")),
        header=None,
        dtype=np.float64,
        na_values=["NaN"],
        engine="c",
    ).to_numpy()
    expected = 1 + 3 * n_slots + 3
    if table.shape[1] != expected:
        raise ValueError(f"{path}: {table.shape[1]} columns, expected {expected} for {n_slots} slots")
    frame_id = table[:, 0].astype(np.int64)
    players = table[:, 1 : 1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = table[:, 1 + 3 * n_slots :]
    return frame_id, players, ball


def slot_teams(meta_path: Path, raw_path: Path, players: np.ndarray, frame_id: np.ndarray) -> np.ndarray:
    """Team ("H"/"A"/"") of each raw slot.

    kloppy reads the team membership from the EPTS metadata. Rather than trust that its player
    order equals the raw column order, match each kloppy player to the raw slot whose positions are
    identical on a sample of frames.
    """
    from kloppy import metrica

    ds = metrica.load_tracking_epts(
        meta_data=meta_path, raw_data=raw_path, sample_rate=1 / 125, coordinates="metrica"
    )
    home, away = ds.metadata.teams
    # TODO(phase 0): confirm teams[0] is Barcelona from the metadata team name.
    row_of = {f: i for i, f in enumerate(frame_id)}
    team = np.full(players.shape[1], "", dtype="<U1")
    for t, label in ((home, "H"), (away, "A")):
        for p in t.players:
            hits = np.zeros(players.shape[1], dtype=np.int64)
            for fr in ds.frames:
                pd_ = fr.players_data.get(p)
                if pd_ is None or pd_.coordinates is None or fr.frame_id not in row_of:
                    continue
                row = players[row_of[fr.frame_id]]
                c = pd_.coordinates
                hits += (np.abs(row[:, 0] - c.x) < 1e-5) & (np.abs(row[:, 1] - c.y) < 1e-5)
            if hits.max() > 0:
                team[int(hits.argmax())] = label
    return team


def load_atd(source: str = "tactical") -> Tracking:
    """ATD on the video-frame index, in metres."""
    paths = asset_paths(source)
    frame_id, players, ball = read_atd_raw(paths["atd_raw"])
    team = slot_teams(paths["atd_meta"], paths["atd_raw"], players, frame_id)
    return to_tracking(frame_id, players, ball, team)


def to_tracking(frame_id: np.ndarray, players: np.ndarray, ball: np.ndarray, team: np.ndarray) -> Tracking:
    """Place parsed rows on video frames (frame_id - 1) and scale to metres."""
    idx = frame_id - 1
    ok = (idx >= 0) & (idx < N_FRAMES)
    n_slots = players.shape[1]

    def blank(*shape):
        return np.full((N_FRAMES, *shape), np.nan)

    b, px, py, ps = blank(2), blank(n_slots), blank(n_slots), blank(n_slots)
    b[idx[ok]] = ball[ok, :2] * (PITCH_LENGTH_M, PITCH_WIDTH_M)
    px[idx[ok]] = players[ok, :, 0] * PITCH_LENGTH_M
    py[idx[ok]] = players[ok, :, 1] * PITCH_WIDTH_M
    ps[idx[ok]] = players[ok, :, 2]
    return Tracking(ball=b, px=px, py=py, ps=ps, team=np.asarray(team))


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
