"""SkillCorner open data (MIT licence, https://github.com/SkillCorner/opendata): broadcast tracking of
A-League 2024/25 matches at 10 fps with per-frame ball possession.

Used only for the team-side experts (model B, change detector), never for the dead-ball models:
SkillCorner's "no possession" frames mix dead ball with loose balls, and broadcast replays fall in
stoppages, which would teach "missing data = dead".

    load_match(id)     raw files -> IdsseMatch at 25 fps, seen positions only, cached as npz
    load_degraded(id)  player identities broken up like the IDSSE/Metrica degrade

Labels: per-frame possession team. A "none" run shorter than fill_same_s with the same team on both
sides (a pass) gets that team; one shorter than split_s between different teams is split at its
middle (a clean switch); everything else is "U" (unknown: never a training row for any expert).
Frames outside the periods are "X". Download: external_data/skillcorner/raw/<id>/ (see README).
"""

import json
import zlib
from pathlib import Path

import numpy as np

from possession.config import FPS, REPO_ROOT, load_params
from possession.external.idsse import IdsseMatch, _read, _save, degrade_players, speeds
from possession.load import Tracking

DIR = REPO_ROOT / "external_data" / "skillcorner"
RAW = DIR / "raw"
SRC_FPS = 10
PITCH = (105.0, 68.0)


def match_ids() -> list[str]:
    return sorted(f"SC{p.name}" for p in RAW.iterdir() if p.is_dir()) if RAW.exists() else []


def _upsample(t_src: np.ndarray, v: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation between neighbouring seen samples only (no filling across gaps)."""
    out = np.full(len(t_dst), np.nan)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return out
    j = np.searchsorted(t_src, t_dst, side="right") - 1
    j = np.clip(j, 0, len(t_src) - 2)
    a, b = v[j], v[j + 1]
    w = (t_dst - t_src[j]) / (t_src[j + 1] - t_src[j])
    inside = (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    val = a + w * (b - a)
    # exact hits on a seen sample keep it even if the next one is missing
    exact = np.isclose(w, 0) & np.isfinite(a)
    out[inside] = np.where(exact[inside], a[inside], val[inside])
    return out


def labels_10fps(group: np.ndarray, period: np.ndarray, fill_same_s: float = 3.0, split_s: float = 2.0) -> np.ndarray:
    """group: per frame "H" / "A" / "" (no possession). Returns H/A/U/X at 10 fps."""
    lab = np.where(period > 0, np.where(group == "", "U", group), "X").astype("<U1")
    for h in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == h)
        g = lab[idx]
        e = np.flatnonzero(g[1:] != g[:-1]) + 1
        starts, ends = np.r_[0, e], np.r_[e, len(g)]
        for k in range(1, len(starts) - 1):
            s, t = starts[k], ends[k]
            if g[s] != "U":
                continue
            before, after = g[starts[k - 1]], g[starts[k + 1]]
            n = t - s
            if before in "HA" and after in "HA":
                if before == after and n < fill_same_s * SRC_FPS:
                    g[s:t] = before
                elif before != after and n < split_s * SRC_FPS:
                    g[s : s + n // 2] = before
                    g[s + n // 2 : t] = after
        lab[idx] = g
    return lab


def _convert(match_id: str) -> IdsseMatch:
    sc = match_id[2:]
    d = RAW / sc
    meta = json.loads((d / f"{sc}_match.json").read_text())
    home_id = meta["home_team"]["id"]
    team_of = {p["id"]: ("H" if p["team_id"] == home_id else "A") for p in meta["players"]}
    frames = [json.loads(line) for line in open(d / f"{sc}_tracking_extrapolated.jsonl")]
    n = len(frames)
    pids = sorted(team_of)
    col = {p: j for j, p in enumerate(pids)}
    period = np.array([f["period"] or 0 for f in frames])
    bx = np.full(n, np.nan); by = np.full(n, np.nan)
    px = np.full((n, len(pids)), np.nan); py = np.full_like(px, np.nan)
    group = np.full(n, "", dtype="<U1")
    for i, f in enumerate(frames):
        b = f.get("ball_data") or {}
        if b.get("is_detected") and b.get("x") is not None:
            bx[i], by[i] = b["x"], b["y"]
        for p in f.get("player_data") or []:
            if p.get("is_detected") and p["player_id"] in col:
                px[i, col[p["player_id"]]], py[i, col[p["player_id"]]] = p["x"], p["y"]
        g = (f.get("possession") or {}).get("group")
        group[i] = "H" if g == "home team" else "A" if g == "away team" else ""
    lab10 = labels_10fps(group, period)
    # 10 -> 25 fps; centred coordinates -> 0..L, 0..W with y down (Metrica / our ATD convention)
    L, W = float(meta.get("pitch_length") or PITCH[0]), float(meta.get("pitch_width") or PITCH[1])
    t_src = np.arange(n) / SRC_FPS
    m = int(np.floor(t_src[-1] * FPS)) + 1
    t_dst = np.arange(m) / FPS
    near = np.clip(np.rint(t_dst * SRC_FPS).astype(int), 0, n - 1)
    ball = np.stack([_upsample(t_src, bx, t_dst) + L / 2, W / 2 - _upsample(t_src, by, t_dst)], 1)
    PX = np.stack([_upsample(t_src, px[:, j], t_dst) for j in range(len(pids))], 1) + L / 2
    PY = W / 2 - np.stack([_upsample(t_src, py[:, j], t_dst) for j in range(len(pids))], 1)
    gt = lab10[near]
    per = period[near]
    halves = []
    for h in np.unique(per[per > 0]):
        idx = np.flatnonzero(per == h)
        halves.append((int(h), int(idx[0]), int(idx[-1])))
    team = np.array([team_of[p] for p in pids])
    tr = Tracking(ball=ball, px=PX, py=PY, ps=speeds(PX, PY, 5), team=team)
    return IdsseMatch(match_id, tr, np.full(m, np.nan), gt, halves, (L, W))


def load_match(match_id: str, refresh: bool = False) -> IdsseMatch:
    path = DIR / "clean" / f"{match_id}.npz"
    if path.exists() and not refresh:
        return _read(match_id, path)
    m = _convert(match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(m, path)
    return m


def load_degraded(match_id: str) -> IdsseMatch:
    """Player identities broken up like the IDSSE/Metrica degrade; ball and seen-only gaps as delivered."""
    params = load_params()
    p, seed = params["idsse_degrade"], params["model"]["seed"]
    m = load_match(match_id)
    rng = np.random.default_rng([seed, zlib.crc32(match_id.encode())])
    px, py, team = degrade_players(rng, m, p)
    tr = Tracking(ball=m.tracking.ball, px=px, py=py, ps=speeds(px, py, p["speed_smooth_frames"]), team=team)
    return IdsseMatch(match_id, tr, m.ball_z, m.gt, m.halves, m.pitch)
