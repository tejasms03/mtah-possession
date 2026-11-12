"""Metrica Sports sample data: 3 matches, same provider and ball behaviour as our ATD.

Why: in IDSSE the ball is frozen whenever play is dead, so dead-ball features learned there do not
transfer. Metrica's ball keeps moving at stoppages and goes missing ~39% of the time, like ours.

Source: https://github.com/metrica-sports/sample-data (games 1-2 CSV, game 3 EPTS + JSON events),
read with kloppy. Terms: "be responsible ... acknowledge the source". Downloaded on first use into
external_data/metrica/ (gitignored); never committed.

Labels are built from the events (there are no per-frame labels):
- possession = team of the latest PASS / CARRY / RECOVERY / SET PIECE / SHOT / BALL OUT, carried
  forward (BALL LOST and CHALLENGE do not change it; the opponent's RECOVERY does)
- dead ball = from the ball going out (BALL OUT end frame, a SHOT that goes out or scores) or a
  FAULT RECEIVED, until the next SET PIECE starts (capped at dead_max_s)
Only the players are damaged (identities, fragments, noise, as for IDSSE). The ball is left as it is.
"""

import json
import urllib.request
import zlib

import numpy as np
import pandas as pd

from possession.config import REPO_ROOT, load_params
from possession.external.idsse import IdsseMatch, _read, _save, degrade_players, speeds
from possession.load import Tracking

MATCHES = {"MET1": 1, "MET2": 2, "MET3": 3}
PITCH = (105.0, 68.0)
DIR = REPO_ROOT / "external_data" / "metrica"
RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
POSSESSION_EVENTS = ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL OUT")


def _events(game: int, home_name: str) -> pd.DataFrame:
    """team ("H"/"A"), type, sub (upper-case text), start, end (frames), period."""
    DIR.mkdir(parents=True, exist_ok=True)
    if game in (1, 2):
        path = DIR / f"game{game}_events.csv"
        if not path.exists():
            urllib.request.urlretrieve(f"{RAW}/Sample_Game_{game}/Sample_Game_{game}_RawEventsData.csv", path)
        df = pd.read_csv(path)
        return pd.DataFrame({
            "team": df["Team"].map({"Home": "H", "Away": "A"}).fillna(""),
            "type": df["Type"].str.upper(), "sub": df["Subtype"].fillna("").str.upper(),
            "start": df["Start Frame"].astype(int), "end": df["End Frame"].astype(int),
            "period": df["Period"].astype(int),
        })
    path = DIR / "game3_events.json"
    if not path.exists():
        urllib.request.urlretrieve(f"{RAW}/Sample_Game_3/Sample_Game_3_events.json", path)
    rows = []
    for e in json.load(open(path))["data"]:
        sub = e.get("subtypes") or []
        sub = [sub] if isinstance(sub, dict) else sub
        team = (e.get("team") or {}).get("name", "")
        rows.append({
            "team": "H" if team == home_name else ("A" if team else ""),
            "type": e["type"]["name"].upper(), "sub": " ".join(s["name"] for s in sub).upper(),
            "start": int(e["start"]["frame"]), "end": int((e.get("end") or e["start"])["frame"] or e["start"]["frame"]),
            "period": int(e["period"]),
        })
    return pd.DataFrame(rows)


def labels(ev: pd.DataFrame, frame_id: np.ndarray, period: np.ndarray, dead_max_frames: int) -> np.ndarray:
    """Per-row "H"/"A"/"D", "X" before the first event of each period."""
    n = len(frame_id)
    row = {f: i for i, f in enumerate(frame_id)}
    at = lambda f: row.get(int(f), int(np.searchsorted(frame_id, f)))           # noqa: E731
    ev = ev.sort_values(["start", "end"], kind="stable").reset_index(drop=True)

    owner = np.full(n, "", dtype="<U1")
    marks = ev[ev["type"].isin(POSSESSION_EVENTS) & ev["team"].isin(["H", "A"])]
    for f, t in zip(marks["start"], marks["team"]):
        owner[min(at(f), n - 1)] = t
    cur = ""
    for i in range(n):                                                         # carry forward
        cur = owner[i] or cur
        owner[i] = cur

    dead = np.zeros(n, dtype=bool)
    restarts = np.sort(ev.loc[ev["type"] == "SET PIECE", "start"].to_numpy())
    goes_out = (ev["type"] == "BALL OUT") | ((ev["type"] == "SHOT") & ev["sub"].str.contains("OUT|GOAL"))
    for f in pd.concat([ev.loc[goes_out, "end"], ev.loc[ev["type"] == "FAULT RECEIVED", "start"]]):
        k = np.searchsorted(restarts, f)
        stop = restarts[k] if k < len(restarts) else f + dead_max_frames
        a, b = at(f), at(min(stop, f + dead_max_frames))
        same_period = period[a : b] == period[a]
        dead[a : a + int(same_period.sum())] = True

    out = np.where(dead, "D", owner)
    out[out == ""] = "X"
    return out


def _convert(match_id: str) -> IdsseMatch:
    from kloppy import metrica

    ds = metrica.load_open_data(match_id=MATCHES[match_id], coordinates="metrica")
    home, away = ds.metadata.teams
    frame_id = np.array([fr.frame_id for fr in ds.frames])
    period = np.array([fr.period.id for fr in ds.frames])
    players = [(p, "H") for p in home.players] + [(p, "A") for p in away.players]
    px = np.full((len(frame_id), len(players)), np.nan)
    py = np.full_like(px, np.nan)
    ball = np.full((len(frame_id), 2), np.nan)
    for i, fr in enumerate(ds.frames):
        if fr.ball_coordinates is not None and fr.ball_coordinates.x is not None:
            ball[i] = fr.ball_coordinates.x, fr.ball_coordinates.y
        for j, (p, _) in enumerate(players):
            d = fr.players_data.get(p)
            if d is not None and d.coordinates is not None:
                px[i, j], py[i, j] = d.coordinates.x, d.coordinates.y
    L, W = PITCH
    ball *= (L, W)
    px *= L
    py *= W
    team = np.array([t for _, t in players])
    ev = _events(MATCHES[match_id], home.name)
    gt = labels(ev, frame_id, period, int(load_params()["metrica"]["dead_max_s"] * 25))
    halves = []
    for h in np.unique(period):
        idx = np.flatnonzero((period == h) & (gt != "X"))
        if len(idx):
            halves.append((int(h), int(idx[0]), int(idx[-1])))
            gt[: idx[0]][period[: idx[0]] == h] = "X"
    tr = Tracking(ball=ball, px=px, py=py, ps=speeds(px, py, 5), team=team)
    return IdsseMatch(match_id, tr, np.full(len(ball), np.nan), gt, halves, PITCH)


def load_match(match_id: str, refresh: bool = False) -> IdsseMatch:
    path = DIR / "clean" / f"{match_id}.npz"
    if path.exists() and not refresh:
        return _read(match_id, path)
    m = _convert(match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(m, path)
    return m


def load_degraded(match_id: str) -> IdsseMatch:
    """Players damaged like IDSSE (same idsse_degrade settings); the ball left as Metrica has it."""
    params = load_params()
    p, seed = params["idsse_degrade"], params["model"]["seed"]
    m = load_match(match_id)
    rng = np.random.default_rng([seed, zlib.crc32(match_id.encode())])
    px, py, team = degrade_players(rng, m, p)
    tr = Tracking(ball=m.tracking.ball, px=px, py=py, ps=speeds(px, py, p["speed_smooth_frames"]), team=team)
    return IdsseMatch(match_id, tr, m.ball_z, m.gt, m.halves, m.pitch)
