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
| `make submission TEAM=<name>` | full pipeline → `outputs/<name>.csv`, then validates it (about 1 h 40 min on one laptop processor thread; it rebuilds all features and models from scratch) |
| `make eval TEAM=<name>` | scores it on the public 5 minutes with the organisers' scorer |

**Our submission (team La Remontada):** `make submission TEAM=la_remontada` writes `outputs/la_remontada.csv`,
byte-identical to the submitted `la remontada.csv` (SHA-256 in `SUBMISSION_SHA256`; `make verify` checks it).
Make cannot take a space in `TEAM`, hence the underscore; only the file name differs.

**v22 candidate (branch `tcn-candidate`):** adds a network sequence model (`src/possession/tcn.py`,
`model.tcn.use`). Its weights `models/tcn.npz` are committed; the pipeline runs them with numpy, so
`make submission` needs no PyTorch. To retrain them (optional, needs PyTorch 2.x, not in the uv env):
build the per-match inputs from the out-of-fold first-stage outputs, then
`python tools/train_tcn.py <ds_dir> models/tcn.npz <the 10 outside match ids, comma separated> 1500 0`.
Bench against v21: `tools/bench_tcn.py` (leave-one-match-out folds from the same script).
| `make verify` | regenerates and checks the file is byte-identical to the recorded hash |
| `uv run possession idsse` | downloads, converts and degrades the 7 IDSSE matches into `external_data/idsse/` (~2.6 GB, ~10 min) |
| `uv run possession bench [pipeline ...]` | scores pipelines on the degraded IDSSE matches → `outputs/bench/<pipeline>.csv` |

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
  external/bench.py  score any pipeline on the 7 degraded IDSSE matches
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

Metrica Sports sample data (3 matches): https://github.com/metrica-sports/sample-data, used with acknowledgement as its README asks.

## Use of AI tools

An AI coding assistant (Claude, by Anthropic, through Claude Code) helped write and debug the
code, design and run the experiments, and draft the documentation. Every result was checked by
the team, and every adoption decision is recorded with its evidence in `docs/DECISIONS.md`.

**No competition data was shared with any AI service.** Following the challenge terms (OVERVIEW.md,
"Use of Large Language Models"), no match video, tracking, Smart Tagging, homography, public
ground truth or output label file was uploaded or shown to a hosted AI. The match data stayed on
our own machines; the assistant worked with source code, documentation and aggregate statistics
computed locally (for example a bench score or a count of possessions). No label in the
submission was written by an AI or by hand: every label comes from the pipeline in this
repository.
