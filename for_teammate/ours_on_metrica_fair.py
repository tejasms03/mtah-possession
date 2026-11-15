"""OUR environment: clean Metrica players damaged with OUR damage (fragments, no added noise),
Metrica's own ball left as it is, then our frozen rules pipeline and the baseline."""
import importlib.util, sys, zlib
from pathlib import Path
import numpy as np
from possession import config, pipeline
from possession.load import Tracking
from possession.external import damage
from possession.external.idsse import IdsseMatch, speeds
src = Path(sys.argv[1]); cfg = config.load_config(); p = cfg["damage"]
spec = importlib.util.spec_from_file_location("bench", config.ROOT / "scripts" / "bench.py")
bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
names = ("rules_v3 (ours)", "baseline_smooth50"); rows = {n: [] for n in names}
for path in sorted(src.glob("MET*_clean.npz")):
    z = np.load(path); mid = path.stem.split("_")[0]
    halves = [tuple(int(v) for v in h[-2:]) for h in z["halves"]]; pitch = tuple(float(v) for v in z["pitch"])
    clean = IdsseMatch(mid, Tracking(ball=z["ball"], px=z["px"], py=z["py"], ps=np.zeros_like(z["px"]), team=z["team"]),
                       z["ball_z"], z["gt_merged"], halves, pitch)
    px, py, team = damage.damage_players(np.random.default_rng([0, zlib.crc32(mid.encode())]), clean, p)
    tr = Tracking(ball=z["ball"], px=px, py=py, ps=speeds(px, py, p["speed_smooth_frames"]), team=team)
    m = IdsseMatch(mid, tr, z["ball_z"], z["gt_merged"], halves, pitch)
    g = z["gt_merged"] != "X"
    for name, lab in ((names[0], pipeline.rule_labels(tr, halves, pitch, 25, cfg)), (names[1], bench.upstream_baseline(m, cfg))):
        a, b = bench.score(cfg, z["gt_merged"], lab), bench.score(cfg, z["gt_raw"], lab)
        rows[name].append((mid, a, b, float(np.mean(lab[g] == "D")), float(np.mean(z["gt_merged"][g] == "D"))))
for name, rs in rows.items():
    print(f"\n{name}")
    for mid, a, b, dp, dt in rs:
        print(f"  {mid}: spellF1 {a['spellF1']:.4f} (raw labels {b['spellF1']:.4f})  macroF1 {a['macroF1']:.4f}  "
              f"F1_D {a['F1_D']:.3f}  poss {a['poss_pred']:.0f}/{a['poss_gt']:.0f}  dead {dp:.2f}/{dt:.2f}  lat {a['lat_med']:.0f}")
    print(f"  MEAN: spellF1 {np.mean([a['spellF1'] for _, a, *_ in rs]):.4f} (raw labels "
          f"{np.mean([b['spellF1'] for _, _, b, *_ in rs]):.4f})  macroF1 {np.mean([a['macroF1'] for _, a, *_ in rs]):.4f}  "
          f"F1_D {np.mean([a['F1_D'] for _, a, *_ in rs]):.3f}")
