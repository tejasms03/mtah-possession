"""Teammate's environment: write the CLEAN Metrica matches (before any player damage) plus labels."""
import sys
from pathlib import Path
import numpy as np
from possession.external import metrica
from possession.external.bench import bench_gt
out = Path(sys.argv[1])
for mid in metrica.MATCHES:
    m = metrica.load_match(mid); t = m.tracking
    np.savez_compressed(out / f"{mid}_clean.npz", ball=t.ball, px=t.px, py=t.py, team=t.team, ball_z=m.ball_z,
                        gt_raw=m.gt, gt_merged=bench_gt(m), halves=np.array(m.halves), pitch=np.array(m.pitch))
    step = np.hypot(np.diff(t.px, axis=0), np.diff(t.py, axis=0)) * 25
    jit = np.nanmedian(np.abs(np.diff(t.px, 2, axis=0)) + np.abs(np.diff(t.py, 2, axis=0)))
    print(f"{mid}: players {t.px.shape[1]}, clean player jitter {jit:.4f} m, median step speed {np.nanmedian(step):.2f} m/s")
