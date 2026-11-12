"""IDSSE open data (Bassek et al., Scientific Data 2025; DFL; CC-BY 4.0).

7 DFL league matches (2 Bundesliga, 5 2. Bundesliga), 25 fps, with per-frame ball possession and ball status. Downloaded by
script into external_data/idsse/ (gitignored), never committed. Attribution: name the DFL and cite
the paper in the technical document.

    download()        fetch the 7 tracking + metadata XMLs (same URLs kloppy builds)
    load_match(id)    kloppy -> clean arrays + per-frame H/A/D/X labels, cached as npz
    degrade(m, p, s)  make clean tracking look like our ATD (params.yaml `idsse_degrade`)
    load_degraded(id) degraded match, cached per parameter set

Labels: D where BallStatus is dead, otherwise the team BallPossession names (home = H). The two
periods are laid out on one timeline with an X gap between them, so the organisers' scorer sees
two halves exactly as it does for BAR-ATH.

Memory: kloppy peaks at ~3.9 GB per match, so load one match per process on small machines.
Owner: person 4.
"""

import hashlib
import json
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession.config import FPS, REPO_ROOT, load_params
from possession.load import Tracking

IDSSE_DIR = REPO_ROOT / "external_data" / "idsse"
CLEAN_DIR = IDSSE_DIR / "clean"
DEGRADED_DIR = IDSSE_DIR / "degraded"
HALF_GAP_FRAMES = 1000

# kloppy's table. The Hugging Face dataset card lists J03WPF and J03WQF instead of J03WQQ and
# J03WR9, but those two do not exist in the repository; the files match kloppy.
MATCHES = {
    "J03WMX": "1. FC Köln vs. FC Bayern München",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen",
    "J03WOH": "Fortuna Düsseldorf vs. SSV Jahn Regensburg",
    "J03WOY": "Fortuna Düsseldorf vs. F.C. Hansa Rostock",
    "J03WPY": "Fortuna Düsseldorf vs. 1. FC Nürnberg",
    "J03WQQ": "Fortuna Düsseldorf vs. FC St. Pauli",
    "J03WR9": "Fortuna Düsseldorf vs. 1. FC Kaiserslautern",
}


@dataclass
class IdsseMatch:
    match_id: str
    tracking: Tracking           # same layout as load.Tracking for BAR-ATH
    ball_z: np.ndarray           # (N,) metres; NaN after degrading (our ATD has no usable z)
    gt: np.ndarray               # (N,) "H" / "A" / "D" / "X"
    halves: list                 # [(half, first, last)] inclusive, like config.HALVES
    pitch: tuple                 # (length, width) metres from the match metadata


# -- download / load ---------------------------------------------------------------------------

def _files(match_id: str, dest: Path = IDSSE_DIR) -> dict[str, tuple[str, Path]]:
    from kloppy._providers.sportec import get_IDSSE_url

    out = {}
    for kind in ("meta", "tracking"):
        url = get_IDSSE_url(match_id, kind)
        out[kind] = (url, dest / url.rsplit("/", 1)[1])
    return out


def download(dest: Path = IDSSE_DIR, match_ids=tuple(MATCHES)) -> list[Path]:
    """Fetch any missing file. ~2.6 GB in total."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for mid in match_ids:
        for url, path in _files(mid, dest).values():
            if not path.exists() or path.stat().st_size == 0:
                tmp = path.with_suffix(".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.rename(path)
            paths.append(path)
    return paths


def _convert(match_id: str) -> "IdsseMatch":
    from kloppy import sportec
    from kloppy.domain import BallState

    files = _files(match_id)
    download(match_ids=(match_id,))
    ds = sportec.load_tracking(meta_data=str(files["meta"][1]), raw_data=str(files["tracking"][1]),
                               coordinates="secondspectrum")  # metres, origin at the centre
    home, away = ds.metadata.teams
    L = float(ds.metadata.pitch_dimensions.pitch_length)
    W = float(ds.metadata.pitch_dimensions.pitch_width)
    players = [p for t in (home, away) for p in t.players]
    col = {p.player_id: j for j, p in enumerate(players)}
    team = np.array(["H"] * len(home.players) + ["A"] * len(away.players))

    by_period: dict = {}
    for fr in ds.records:
        by_period.setdefault(fr.period.id, []).append(fr)
    periods = sorted(by_period)[:2]

    n = sum(len(by_period[p]) for p in periods) + HALF_GAP_FRAMES * (len(periods) - 1)
    px = np.full((n, len(players)), np.nan)
    py = np.full_like(px, np.nan)
    ball = np.full((n, 2), np.nan)
    bz = np.full(n, np.nan)
    gt = np.full(n, "X")
    halves = []
    i = 0
    for half, p in enumerate(periods, start=1):
        start = i
        for fr in sorted(by_period[p], key=lambda f: f.frame_id):
            b = fr.ball_coordinates
            if b is not None:
                ball[i] = (b.x + L / 2, b.y + W / 2)
                bz[i] = getattr(b, "z", None) or 0.0
            for pl, pdata in fr.players_data.items():
                c = pdata.coordinates
                if c is not None and pl.player_id in col:
                    px[i, col[pl.player_id]] = c.x + L / 2
                    py[i, col[pl.player_id]] = c.y + W / 2
            if fr.ball_state == BallState.ALIVE and fr.ball_owning_team is not None:
                gt[i] = "H" if fr.ball_owning_team == home else "A"
            else:
                gt[i] = "D"
            i += 1
        halves.append((half, start, i - 1))
        i += HALF_GAP_FRAMES

    tr = Tracking(ball=ball, px=px, py=py, ps=speeds(px, py, 5), team=team)
    return IdsseMatch(match_id, tr, bz, gt, halves, (L, W))


def _save(m: "IdsseMatch", path: Path) -> None:
    """Write to a temporary file and rename, so an interrupted run never leaves a broken cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = m.tracking
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(tmp, ball=t.ball.astype(np.float32), px=t.px.astype(np.float32),
                        py=t.py.astype(np.float32), ps=t.ps.astype(np.float32), team=t.team,
                        ball_z=m.ball_z.astype(np.float32), gt=m.gt,
                        halves=np.array(m.halves), pitch=np.array(m.pitch))
    tmp.replace(path)


def _read(match_id: str, path: Path) -> "IdsseMatch":
    z = np.load(path)
    f64 = lambda k: z[k].astype(np.float64)  # noqa: E731
    tr = Tracking(ball=f64("ball"), px=f64("px"), py=f64("py"), ps=f64("ps"), team=z["team"])
    return IdsseMatch(match_id, tr, f64("ball_z"), z["gt"],
                      [tuple(map(int, h)) for h in z["halves"]], tuple(map(float, z["pitch"])))


def load_match(match_id: str, refresh: bool = False) -> "IdsseMatch":
    """Clean match with labels, from external_data/idsse/clean/ (built on first use)."""
    path = CLEAN_DIR / f"{match_id}.npz"
    if refresh or not path.exists():
        _save(_convert(match_id), path)
    return _read(match_id, path)


# -- degrade -----------------------------------------------------------------------------------

def _lognormal(rng, median_s, sigma):
    return max(1, int(round(rng.lognormal(np.log(median_s * FPS), sigma))))


def _runs(mask):
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def graded(m: "IdsseMatch") -> np.ndarray:
    return m.gt != "X"


def _ball_gaps(rng, hazard, h0, p):
    n = len(hazard)
    miss = np.zeros(n, bool)
    u = rng.random(n)
    t = 0
    while t < n:
        if u[t] < h0 * hazard[t]:
            g = _lognormal(rng, p["ball_gap_median_s"], p["ball_gap_sigma"])
            miss[t:t + g] = True
            t += g
        else:
            t += 1
    return miss


def degrade_ball(rng, m: "IdsseMatch", p: dict) -> np.ndarray:
    """Drop the ball in long runs, more often near crowds and in the air.

    The base hazard is bisected so the missing share on graded frames lands on target. Every
    trial replays the same random stream, so the result is deterministic.
    """
    t = m.tracking
    d = np.hypot(t.px - t.ball[:, :1], t.py - t.ball[:, 1:])
    crowd = np.maximum((d < p["crowd_radius_m"]).sum(1) - 1, 0)
    air = np.nan_to_num(m.ball_z) > p["air_height_m"]
    hazard = (1 + p["crowd_gain"] * crowd) * np.where(air, p["air_gain"], 1.0)
    g = graded(m)
    seed = int(rng.integers(2**32))
    lo, hi = 1e-5, 1e-1
    for _ in range(30):
        h0 = np.sqrt(lo * hi)
        miss = _ball_gaps(np.random.default_rng(seed), hazard, h0, p)
        lo, hi = (h0, hi) if miss[g].mean() < p["ball_missing_share"] else (lo, h0)
    ball = t.ball.copy()
    ball[miss | np.isnan(ball[:, 0])] = np.nan
    return ball


def _fragments(rng, start, stop, p):
    t = start
    while t < stop:
        f = _lognormal(rng, p["frag_median_s"], p["frag_sigma"])
        yield t, min(stop, t + f)
        t += f
        if rng.random() < p["gap_long_share"]:
            t += _lognormal(rng, p["gap_long_median_s"], p["gap_long_sigma"])
        else:
            t += _lognormal(rng, p["gap_short_median_s"], p["gap_short_sigma"])


class _SlotPool:
    """Slots of one team. A fragment goes to a random slot that is free over its span."""

    def __init__(self, rng):
        self.rng = rng
        self.busy_until: list[int] = []

    def take(self, start, stop):
        free = [k for k, b in enumerate(self.busy_until) if b <= start]
        if free:
            k = free[self.rng.integers(len(free))]
        else:
            k = len(self.busy_until)
            self.busy_until.append(0)
        self.busy_until[k] = stop
        return k


def degrade_players(rng, m: "IdsseMatch", p: dict):
    """Identities removed (team only), random fragments on reused slots, gaps, duplicates, jitter."""
    t = m.tracking
    pieces = []  # (team, start, stop, source column, dx, dy)
    for j in range(t.n_slots):
        for a, b in _runs(~np.isnan(t.px[:, j])):
            for s, e in _fragments(rng, a, b, p):
                pieces.append((t.team[j], s, e, j, 0.0, 0.0))
            for _ in range(rng.poisson(p["dup_per_player_min"] * (b - a) / FPS / 60)):
                s = int(rng.integers(a, b))
                e = min(b, s + _lognormal(rng, p["dup_median_s"], p["dup_sigma"]))
                off = rng.normal(0, p["dup_offset_m"], 2)
                pieces.append((t.team[j], s, e, j, off[0], off[1]))
    pieces.sort(key=lambda q: q[1])

    pools = {"H": _SlotPool(rng), "A": _SlotPool(rng)}
    assigned = [(q, pools[q[0]].take(q[1], q[2])) for q in pieces]
    n_home = len(pools["H"].busy_until)
    n_slots = n_home + len(pools["A"].busy_until)
    px = np.full((len(t.px), n_slots), np.nan)
    py = np.full_like(px, np.nan)
    noise = p["pos_noise_m"]
    for (tm, s, e, j, dx, dy), k in assigned:
        c = k if tm == "H" else n_home + k
        px[s:e, c] = t.px[s:e, j] + dx + rng.normal(0, noise, e - s)
        py[s:e, c] = t.py[s:e, j] + dy + rng.normal(0, noise, e - s)
    team = np.array(["H"] * n_home + ["A"] * (n_slots - n_home))
    return px, py, team


def speeds(px: np.ndarray, py: np.ndarray, smooth: int) -> np.ndarray:
    """m/s per slot: displacement over `smooth` frames centred on each frame, divided by the time.
    Averaging per-frame speed magnitudes instead would add the position noise up (it never
    cancels), which gave ~6.7 m/s for players standing still at a restart."""
    h = max(1, smooth // 2)
    v = np.full_like(px, np.nan)
    v[h:-h] = np.hypot(px[2 * h:] - px[:-2 * h], py[2 * h:] - py[:-2 * h]) * FPS / (2 * h)
    return v


def degrade(match: "IdsseMatch", noise_params: dict, seed: int) -> "IdsseMatch":
    """Damaged copy of a clean match. Labels, halves and pitch are unchanged."""
    p = noise_params
    rng = np.random.default_rng([seed, zlib.crc32(match.match_id.encode())])
    ball = degrade_ball(rng, match, p)
    px, py, team = degrade_players(rng, match, p)
    tr = Tracking(ball=ball, px=px, py=py, ps=speeds(px, py, p["speed_smooth_frames"]), team=team)
    return IdsseMatch(match.match_id, tr, np.full(len(ball), np.nan), match.gt, match.halves,
                      match.pitch)


def _params_key(p: dict, seed: int) -> str:
    return hashlib.sha1(json.dumps([p, seed], sort_keys=True).encode()).hexdigest()[:10]


def load_degraded(match_id: str, params: dict | None = None) -> "IdsseMatch":
    """Degraded match, cached under a hash of the degrade parameters and seed."""
    params = params or load_params()
    p = params["idsse_degrade"]
    path = DEGRADED_DIR / f"{match_id}_{_params_key(p, p['seed'])}.npz"
    if not path.exists():
        _save(degrade(load_match(match_id), p, p["seed"]), path)
    return _read(match_id, path)


def degrade_report(m: "IdsseMatch") -> dict:
    """The aggregates the degrade parameters were calibrated against (see DECISIONS.md)."""
    g = graded(m)
    vis = ~np.isnan(m.tracking.px[g])
    nv = vis.sum(1)
    frag = [b - a for j in range(vis.shape[1]) for a, b in _runs(vis[:, j])]
    miss = np.isnan(m.tracking.ball[g, 0])
    gaps = [b - a for a, b in _runs(miss)]
    return {
        "slots": int(m.tracking.n_slots),
        "visible_per_frame": float(nv.mean()),
        "frac_frames_over_22": float(np.mean(nv > 22)),
        "slot_run_median_s": float(np.median(frag)) / FPS,
        "ball_missing": float(miss.mean()),
        "ball_gap_median_s": float(np.median(gaps)) / FPS if gaps else 0.0,
        "ball_gaps": len(gaps),
    }


if __name__ == "__main__":
    import sys

    for mid in sys.argv[1:] or list(MATCHES):
        load_match(mid)
        print(f"{mid}: converted")
