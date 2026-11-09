# Plan

Deadline Mon 17 Nov 2026. **Submit Sat 15 Nov.** 16–17 Nov are buffer only.

Merged from two research passes: [reports/Possession labelling method comparison.md](../reports/Possession%20labelling%20method%20comparison.md)
and the team's "Football possession inference research" PDF. Shared plan doc: https://claude.ai/artifact/WVMjHFFs64J7t9kHtK8v89. Where they disagreed, the choice and
the reason are in DECISIONS.md.

**What the score depends on, in order:**
1. **Dead ball.** The ball is out of play ~40% of LaLiga match time. Every missed stoppage merges two
   possessions into one long wrong one and deletes a D segment. The published baseline almost never
   predicts D.
2. **Boundary timing.** For a possession of length L that starts δ late, IoU ≈ (L−δ)/(L+δ). A 3 s
   possession 1 s late only passes thresholds 0.3–0.5.
3. **Flicker.** One stray switch turns 1 match into 0 matches + 2 false positives.
4. **Team identity on live frames** is the least urgent: already 83–89% right in the baseline.

So the work goes into a dead-ball detector that fuses several signals, and a decoder that switches
only at real events and on time. The model choice (LightGBM vs NN) matters least.

## The system

```
raw ATD ─► 1 clean ─► 2 touches + short gap bridging ─► 3 notes per frame (multi-scale windows)
                                                              │
   IDSSE (7 Bundesliga, labelled) ──► damage to match our ATD ┤   Metrica samples: label convention
                                                              ▼
   4a dead-ball evidence: stasis, ball out, dropout runs,  4b LightGBM A: P(dead)
      Smart Tagging set pieces, [whistle if audio exists]   4c LightGBM B: P(Barça | live)
                                                              ▼
   boundary score b_t (touch peaks, nearest-team change, ball out, set-piece start, [whistle])
                                                              ▼
            5 decoder: Viterbi H/A/D, switch cost λ_t = λ0 − α·b_t, min durations, restart rules
                                                              ▼
            6 timing: back-date to first touch, then subtract median lag measured on IDSSE
                                                              ▼
            7 tune all dials on spellF1 (leave one IDSSE match out) → CSV → validate → verify
```

| # | Stage | What it does | Module |
|---|---|---|---|
| 1 | Clean | Parse ATD, metres, delete impossible ball detours and off-pitch junk, duplicate players, slot-free velocities. **Done.** | `load.py`, `clean.py` |
| 2 | Touches, gaps | Savitzky–Golay ball smoothing (2nd order, 7 frames); touch = acceleration peak (start ≥ 4 m/s², retune for our noise) or direction change > 10° near a player; bridge gaps ≤ 0.5 s only | `ballfilter.py` |
| 3 | Notes per frame | Ball state; each team's 5 nearest players sorted by distance; 1 m possession zone, duel flag (2+ opponents within 1 m); team shape; same-team logic only (never same player). Windows at several scales: ±0.4, ±1, ±2, ±4 s for live play and up to ±40 s for stoppage context | `features.py` |
| 4a | Dead-ball evidence | Team stasis (mean speed drop), ball beyond a line in ATD coordinates, ball-dropout runs, Smart Tagging set pieces (and which team restarts), set-piece formations; whistle only if the tactical video has a usable audio track | `deadball.py` |
| 4b/c | Scores | LightGBM A: dead vs alive. LightGBM B: Barça vs Athletic on live frames. Trained on damaged IDSSE (+ confident frames from our match, down-weighted) | `model.py`, `labeller.py`, `external/idsse.py` |
| 5 | Decode | 3-state Viterbi, each state expanded into a chain of 5–10 sub-states for a minimum duration. H↔A dearer than ↔D. Switch cost drops where the boundary score is high. Restart rules: after ball out, the opponent of the last toucher restarts; after a goal, the conceding team. Confident frames locked | `decode.py` |
| 6 | Timing | Back-date each switch to the new team's first controlled touch (−2 s … +0.5 s), then measure the predicted-minus-true lag histogram on damaged IDSSE and subtract its median | `decode.py` |
| 7 | Tune and ship | Grid search λ0, α, min durations, offsets on spellF1 across held-out IDSSE matches. Public 5 min: one sanity look for gross convention errors only | `decode.py`, `sanity.py`, `submission.py` |

### Rules we hold to

- **Day-1 harness first.** Local spellF1/macroF1 = the organisers' `possession_metrics.py`, run on
  damaged IDSSE and on the public window. Reproduce the 0.2433 baseline before changing anything.
- **Label convention is the biggest unmeasured risk:** when does dead ball start (ball over the line,
  whistle, last touch) and end (restart touch)? Check IDSSE BallStatus and Metrica sample-data events
  (same provider as this challenge) against the public window on day 1. Where they disagree, follow
  what the public window shows.
- IDSSE damage matches **our** pattern: ball missing ~41% in bursts (more in crowds and in flight),
  tracks cut into fragments with slot swaps inside a team, jitter. Check: a classifier separating
  damaged IDSSE from our frames should score AUC ≈ 0.5.
- Never train or tune on the public 5 minutes. Splits by match or time block, never random frames.
- Deterministic: LightGBM `deterministic=True`, `force_row_wise=True`, fixed seeds and threads;
  final file made on one machine and its SHA-256 committed.
- Keep a change only if it beats the fallback by more than the ~0.04 noise band.
- Never write X on a graded frame (spellF1 forward-fills it, macroF1 counts it as a miss).
- Every threshold lives in `params.yaml`, its reason in `DECISIONS.md`.

### Not doing (and why)

- **Video ball detection (WASB etc.):** team decision, not a vision project; hours of GPU per run
  that the organisers would have to repeat to reproduce the file.
- **Homography for ball-out:** the tactical ATD is already in pitch coordinates.
- **PathCRF / Ballradar retraining, label models (Snorkel), DANN:** improve frames that are already
  mostly right, cost days, need player IDs or target labels we don't have.
- **Neural net:** optional only, see below.

### Optional, each owned by one person and dropped first if behind

- **Whistle detector (11 Nov):** only if `ffprobe` shows an audio track on the tactical video.
  Self-calibrated narrow band with short repeated bursts, persistence ≥ 150–250 ms, no hardcoded
  frequency; check sync by aligning whistles with drops in team speed. Unverified that audio exists.
- **Small TCN (12 Nov):** same inputs as LightGBM; kept only if it beats it on held-out IDSSE
  matches over 3 seeds; then averaged with LightGBM before the decoder.
- **HSMM with IDSSE duration priors (13 Nov):** only if the Viterbi plateaus.

## Days

| Day | Work | Exit |
|---|---|---|
| Mon 9 Nov | Tactical data in, `possession clean` on it. Harness: scorer on public window + IDSSE loader + damage v0; reproduce baseline. Convention check (IDSSE, Metrica events, public window). `ffprobe` the tactical video for audio | baseline reproduced; convention written in DECISIONS.md |
| Tue 10 Nov | Stage 2 touches; dead-ball v1 (stasis, ball out, dropout runs, set-piece tags); decoder v1 on rule scores with λ_t | **fallback submission saved** |
| Wed 11 Nov | Stage 3 multi-scale features; LightGBM A and B on damaged IDSSE; whistle (optional) | models beat rule scores on held-out IDSSE |
| Thu 12 Nov | Restart rules, back-dating, lag correction; grid search on spellF1; TCN (optional) | tuned pipeline |
| Fri 13 Nov | Ablations, AUC domain check, HSMM if plateau; freeze params | params frozen |
| Sat 14 Nov | Fresh clone, `make verify` byte-identical; technical document | SUBMISSION_SHA256 committed |
| Sun 15 Nov | Validate exact file, upload | **submitted** |
| 16–17 Nov | Buffer and resubmission only | |

## Owners

| Person | Work |
|---|---|
| 1 | Stages 1–3: `clean.py` (done), `ballfilter.py` touches, `features.py` |
| 2 | Stage 4a: `deadball.py` (stasis, ball out, dropout, set pieces, restart team); whistle if audio |
| 3 | Stages 5–6: `decode.py` (λ_t, sub-state chains, restart rules, back-dating, lag correction) |
| 4 | `external/idsse.py` (+ Metrica samples), damage + AUC check, `labeller.py`, `model.py` |
| 5 | Harness and tuning loop (stage 7), reproducibility, technical document; TCN if time |

## Risks and open questions (all checked Mon 9 Nov)

| Risk | Check | Fallback |
|---|---|---|
| Dead-ball convention: D starts at ball over the line, whistle or last touch? ends at the restart touch? | IDSSE BallStatus and Metrica events vs the public window | Follow the public window |
| Damaged IDSSE may not resemble our ATD | AUC of a classifier separating the two | Train on our confident frames only |
| Baseline diagnostics (83–89% team, D recall 8–34%) are from our own run | Re-run with the harness | Re-rank priorities if D is not the main error |
| Tactical video may have no audio | `ffprobe` | Drop the whistle |
| Team-to-slot matching untested on the real metadata | `possession clean` shows ~11 H and ~11 A slots | Fix `slot_teams` before stage 2 |

## Cut order if behind

1. TCN, HSMM
2. Whistle
3. IDSSE-trained models (decoder runs on rule scores)
4. Never the fallback
