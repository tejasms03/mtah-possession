# Plan

Deadline Mon 17 Nov 2026. **Submit Sat 15 Nov.**

## Data flow

```
OUR MATCH                                   IDSSE (7 matches, labelled)
raw ATD                                     clean DFL tracking + possession/ball-status labels
  │                                           │
  │                                           ▼
  │                                         degrade to look like our ATD
  ▼                                           │
clean (src/possession/clean.py) ◄──── same code ┘
  ▼
ball IMM filter (ballfilter.py)
  ▼
features per frame (features.py, see FEATURES.md)
  ├──► labeller.py ──► confident labels ─┐
  │                    IDSSE labels ─────┤
  └──► model.py (±2 s window) ◄──────────┘ trained on both
          ▼
       H/A/D scores per frame + deadball.py segments
          ▼
       decode.py (HSMM/Viterbi, back-dating)
          ▼
       outputs/<team>.csv → validate → submit
```

## Days

| Days | Work | Exit |
|---|---|---|
| 6–7 Nov | loaders, cleaning, phase 0 audit (gap histogram, noise, public-GT boundary inventory), download IDSSE | `possession check-data` ok, audit notes in DECISIONS.md |
| 7–9 Nov | IMM, features, dead ball, labeller, decoder | **fallback submission saved by 9 Nov** |
| 9–12 Nov | degrade IDSSE, LightGBM on IDSSE + confident labels; temporal NN if time | keep only what beats fallback beyond bootstrap noise |
| 13 Nov | freeze params, fresh clone, `make verify` byte-identical | SUBMISSION_SHA256 committed |
| 14–15 Nov | technical document, validate exact file, upload | submitted |

## Owners

| Person | Modules |
|---|---|
| 1 | load.py, clean.py, ballfilter.py, features.py |
| 2 | deadball.py, public-GT boundary inventory |
| 3 | labeller.py, decode.py, sanity.py |
| 4 | external/idsse.py, model.py |
| 5 | repo, reproducibility, technical document |

## Cut order if behind

1. Temporal NN (LightGBM is enough)
2. IDSSE pretraining (train on our confident labels only)
3. Never the fallback
