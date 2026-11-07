# Plan

Deadline Mon 17 Nov 2026. **Submit Sat 15 Nov.**

Why this design: [reports/Possession labelling method comparison.md](../reports/Possession%20labelling%20method%20comparison.md).
Short version: the decoder and the match between training data and our data matter more than the
model. A neural net is an optional second opinion, never the core.

## The system in one picture

```
raw ATD ─► 1 clean ─► 2 bridge ball gaps + find touches ─► 3 notes per frame (±2 s)
                                                              │
                 IDSSE (7 Bundesliga matches, labelled)       │
                 damaged to look like our ATD ─► same 1-3 ────┤
                                                              ▼
                         4 two LightGBM models: "in play?" and "whose ball?"
                           trained on damaged IDSSE + our confident frames
                                                              ▼
                         5 decoder: calmest story that respects football rules
                                                              ▼
                         6 move each switch back to the new team's first touch
                                                              ▼
                         7 CSV → validate → submit
```

## The stages

| # | Stage | What it does | Module |
|---|---|---|---|
| 1 | Clean | Parse the ATD, convert to metres, delete physically impossible ball points (> 35 m/s) and off-pitch junk, give every player a velocity without trusting slot ids | `load.py`, `clean.py` |
| 2 | Bridge gaps, find touches | Fill **short** ball gaps from the ball's last motion (not long ones); mark touches = sudden change in ball speed or direction | `ballfilter.py` |
| 3 | Notes per frame | Ball state, each team's 5 nearest players to the ball (sorted), last toucher, team shape, stoppage clues, all over ±2 s | `features.py` (see FEATURES.md) |
| 4 | Score | Model A: in play vs dead. Model B: Barça vs Athletic. Each outputs a probability per frame | `model.py`, `labeller.py` (confident frames), `deadball.py` (stoppage clues), `external/idsse.py` (training data) |
| 5 | Decode | Viterbi over H / A / D: cost per switch, minimum lengths, H↔A only near a touch, restart rules, confident frames locked | `decode.py` |
| 6 | Back-date | Move each switch to the new team's first controlled touch within about −2 s … +0.5 s | `decode.py` |
| 7 | Tune and ship | Tune decoder dials on spellF1, leave one IDSSE match out at a time; one check on the public 5 min; write, validate, verify | `decode.py`, `sanity.py`, `submission.py` |

### Rules we hold to

- The public 5 minutes is never trained or tuned on. It is the exam paper: if it is badly off,
  our understanding of the labels is wrong, so fix logic, not numbers.
- IDSSE is damaged to match **our** ball-gap pattern, not random dropout. Check: a classifier that
  tries to tell damaged IDSSE from our data should score AUC ≈ 0.5.
- Day-1 check: does IDSSE mean the same thing by possession and dead ball as the organisers' GT?
- One round of self-training at most. Splits are by match or time block, never random frames.
- LightGBM with `deterministic=True`, `force_row_wise=True`, fixed seeds, fixed threads.

### The neural net (optional)

A small temporal CNN (TCN) on the same inputs as LightGBM. **One person, 10–11 Nov.** Kept only if
it beats LightGBM on held-out IDSSE matches over 3 seeds; if kept, its probabilities are averaged
with LightGBM's before the decoder.

## Days

| Days | Work | Exit |
|---|---|---|
| 7–8 Nov | Stages 1–3; download IDSSE; audit (ball-gap pattern, public-GT boundaries, IDSSE label meaning) | features for our match and IDSSE, audit notes in DECISIONS.md |
| 8–9 Nov | Confident labeller, dead-ball + restart rules, decoder v1 on rule scores | **fallback submission saved by 9 Nov** |
| 9–12 Nov | Damage IDSSE (AUC check), models A and B, tune decoder on spellF1; NN time-box 10–11 Nov | keep only what beats the fallback beyond noise |
| 13 Nov | Freeze params, fresh clone, `make verify` byte-identical | SUBMISSION_SHA256 committed |
| 14–15 Nov | Technical document, validate exact file, upload | submitted |

## Owners

| Person | Work |
|---|---|
| 1 | Stages 1–3: `load.py`, `clean.py`, `ballfilter.py`, `features.py` |
| 2 | `deadball.py`: stoppages, set-piece templates, restart rules; public-GT boundary inventory |
| 3 | Stages 5–7: `decode.py`, back-dating, tuning on spellF1, `sanity.py` |
| 4 | `external/idsse.py` (download, damage, AUC check), `labeller.py`, `model.py` |
| 5 | Repo, reproducibility, technical document; NN time-box |

## Cut order if behind

1. Neural net
2. IDSSE training (models trained on our confident frames only)
3. Never the fallback
