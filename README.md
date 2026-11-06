# mtah-possession

Frame-level possession labels (H / A / D) for the More Than A Hack 2027 possession challenge,
BAR-ATH. Positional data in, one label per frame out. No video, no hand labels.

## Setup

```bash
git clone https://github.com/tejasms03/mtah-possession.git
cd mtah-possession
make setup                 # installs the pinned environment (needs uv: https://docs.astral.sh/uv/)
```

Then:

1. **Match data:** download the bundle with your own link and unpack it into `data/`
   (see [data/README.md](data/README.md)). It is gitignored. Never commit it.
2. **Organisers' repo** (scorer + public ground truth): clone it into `vendor/mtah`
   (see [vendor/README.md](vendor/README.md)).
3. Check everything is in place: `uv run possession check-data`

## Commands

| Command | What it does |
|---|---|
| `make test` | unit tests |
| `make submission TEAM=<name>` | full pipeline → `outputs/<name>.csv`, then validates it |
| `make eval TEAM=<name>` | scores it on the public 5 minutes with the organisers' scorer |
| `make verify` | regenerates and checks the file is byte-identical to the recorded hash |

## Layout

```
src/possession/
  config.py       paths, frame ranges, pitch size
  load.py         ATD / Smart Tagging / homography / public GT onto the video-frame index
  clean.py        reject impossible jumps, velocities, slot-free player chains
  ballfilter.py   IMM Kalman (rolling / airborne / carried)
  features.py     per-frame features (docs/FEATURES.md)
  deadball.py     D segments and restart possessor
  labeller.py     deterministic confident labels (+ fallback system)
  model.py        LightGBM / temporal NN emissions
  decode.py       HSMM / Viterbi over H, A, D
  sanity.py       full-match sanity panel and public-GT boundary audit
  submission.py   write and validate the CSV
  external/idsse.py  IDSSE download, load, degrade
params.yaml       every threshold
docs/             PLAN, FEATURES, DECISIONS, challenge SUMMARY
```

## Working together

- Branch per piece of work (`git switch -c deadball`), open a pull request into `main`.
- Every threshold goes in `params.yaml`, and the reason for it in `docs/DECISIONS.md`.
- The public 5 minutes is for checking correctness, never for choosing between versions.

## External data

IDSSE: Bassek, Rein, Weber, Memmert (2025). *An integrated dataset of spatiotemporal and event
data in elite soccer.* Scientific Data 12, 195. Data © Deutsche Fußball Liga (DFL), CC-BY 4.0.
