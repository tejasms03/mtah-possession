# Decisions

One entry per threshold, design choice or ablation. This file becomes most of the technical
document, so record what was tried and dropped as well as what was kept.

| Date | Decision | Value | Evidence | Who |
|---|---|---|---|---|
| 2026-11-06 | Model input is ATD-derived features only, no video | | scope: "not a tracking problem" (CHALLENGE) | |
| 2026-11-06 | Never write X on graded frames | | spellF1 forward-fills X; macroF1 counts it as a miss (EVALUATION) | |
| 2026-11-06 | Offline processing, labels may use future frames | | no causality rule in RULES; organisers' own control proxies look ahead | |
| 2026-11-06 | No hand-labelled data | | team choice; all labels algorithmic | |
| 2026-11-06 | External data: IDSSE (CC-BY 4.0), degraded to look like our ATD | | allowed by RULES (public, documented, not this fixture) | |
| 2026-11-07 | LightGBM is the main scorer; NN only as a time-boxed second opinion | | Borghesi 2023: architectures within ±1.7 pts; no NN vs boosting comparison exists; trees strong on tabular (Grinsztajn 2022). See reports/ | |
| 2026-11-07 | Decoder is the centrepiece, tuned directly on spellF1 (leave one IDSSE match out) | | PathCRF: constrained Viterbi +8 pts event F1; action segmentation: refinement triples segment F1 at flat frame accuracy | |
| 2026-11-07 | Two models: in play vs dead, then H vs A | | dead/alive and possession use different evidence (Borghesi ablations) | |
| 2026-11-07 | Ball filter only bridges short gaps and finds touches | | no football evidence for IMM; rebuild-ball-then-rules scored 48% in PathCRF | |
| 2026-11-07 | IDSSE match IDs from kloppy, not the HF dataset card | J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9 | card lists J03WPF/J03WQF, which have no files in pysport/idsse-data; kloppy's table matches the files | |
| 2026-11-07 | IDSSE labels: D where BallStatus dead, else BallPossession team | | same mechanical derivation as the organisers' GT (CHALLENGE: "D where the tracking marks the ball dead, otherwise the owning team") | |
| 2026-11-07 | IDSSE possession is touch-level, unlike BAR-ATH GT | J03WMX: 511 spells, median 3.0 s, 37% < 2 s | BAR-ATH public GT: 17 possessions, mean 14.2 s. Bench rewards flicker the BAR-ATH GT punishes; open question whether to add a merged "controlled" label variant | |
| 2026-11-07 | Degrade calibrated to BAR-ATH aggregates, not random dropout | `params.yaml: idsse_degrade` | ball gaps are long runs (see phase 0); per match after degrading: ball missing 40-42%, 140-170 gaps, 19.5-20.4 visible players, slot runs 22-30 s median, ~26 slots (BAR-ATH 46) | |
| 2026-11-07 | Bench baseline = organisers' baseline_smooth50 on degraded IDSSE | spellF1 0.179 mean over 7 matches (range 0.144-0.196) | close to its 0.2433 on BAR-ATH, so the damage is about as hard as our data | |
| 2026-11-07 | Own ATD parser instead of kloppy for the raw txt | | format: `frame:` 46 slots (tactical) `x,y,speed` `;`-separated `:` ball `x,y,s`; the third ball value is speed `s` (metadata `BallChannelRef s`), present on only 8.7% of tactical frames with a ball; kloppy only needed for slot→team | |
| 2026-11-09 | Plans merged: dead ball and boundary timing are the main error, not team identity | | team PDF: baseline 83–89% team-correct on live frames, D recall 8–34%; ball out ~40% of LaLiga time (CIES) | |
| 2026-11-09 | Soft switch cost λ_t = λ0 − α·b_t instead of hard touch gating | | one DP pass, cheap switches at real events; replaces a hard rule that fails when a touch is missed | |
| 2026-11-09 | Multi-scale windows (±0.4 s … ±40 s) instead of ±2 s only | | Lang et al. 2022: time-shifted player features to ±80 s give 92% dead/alive, 0.80 stoppage F1, no ball | |
| 2026-11-09 | Back-date to first touch, then subtract median lag measured on damaged IDSSE | | both plans; lag is systematic for centred-window rules | |
| 2026-11-09 | Metrica sample events used to check the dead-ball convention | | same provider as the challenge; IDSSE convention undocumented | |
| 2026-11-09 | Whistle detection optional, only if the tactical video has audio | | PDF: 93% recall / 88% precision on broadcast (Raventós 2015); DATA.md does not mention audio | |
| 2026-11-09 | No video ball detection (WASB), no homography for ball-out | | team decision (no vision); GPU-hours to reproduce; ATD already in pitch coordinates | |
| 2026-11-12 | Stage 1 on tactical data | | 46 slots → 22 H, 21 A, 3 other; ball missing 41.3% of graded frames; 1,582 off-pitch ball points and 2,838 duplicate player points removed, 0 detours; our speeds match ATD speed to 0.11 m/s (median); Barcelona = first metadata team = H, attacks left to right in the 1st half | |
| 2026-11-12 | Stage 2 touches: velocity change ≥ 3 m/s or turn ≥ 25° with a player within 2 m; trust velocity only ≥ 3 frames inside a ball run | params `ball` | BAR-ATH: 2,357 touches (19.4/min), H 1,527 / A 830; public window: touch team = GT team on 84.4% of live-play touches, 2 of 162 touches in D; bridging adds only 170 frames (gaps are long). Run-edge velocity made false touches on synthetic jitter | |
| 2026-11-12 | Fallback v1: decoder + dead-ball dials tuned on 5 degraded IDSSE matches (coordinate search, merged labels) | params `decode`, `dead_ball`; outputs/tune.json | IDSSE bench spellF1 0.442 (0.377-0.492 per match) vs organisers' baseline 0.189. Public window check only: spellF1 0.381, macroF1 0.718, 26 vs 17 possessions, D F1 0.62 (baseline_smooth50 0.326 / 0.621). Several values at grid edges: extend grid | |
| 2026-11-12 | Models A (dead) and B (whose ball, team-blind) as gradient-boosted trees (sklearn HistGradientBoosting: LightGBM needs a system libomp on macOS) + retuned decoder | params `model`, `decode`; outputs/tune_model.json | IDSSE leave-one-match-out: dead/alive acc 0.94-0.96, team acc 0.86-0.91; bench spellF1 0.632 (0.580-0.663) vs fallback 0.444. Public window check: spellF1 0.563 (fallback 0.381), team acc 0.91, D recall 0.32, median switch 6 frames early. w_dead (0.25) and w_team (2) at grid edges | |
| 2026-11-12 | DFL freezes the ball whenever play is dead (speed exactly 0 on 100% of dead frames, clean files too) | | model A learned "ball still = dead" (95% on DFL) and missed 3 of 4 stoppages in the public window, where our ball keeps moving (median 2.7 m/s in D) | |
| 2026-11-12 | Model A drops ball-motion and ball-handling features (33 of 116) | params `model.dead_drop_prefixes` | v2: honest DFL bench 0.585; public check 0.400, D recall 0.94 but D on 58% of the full match (overshoot). Not adopted | |
| 2026-11-12 | Metrica sample data (3 matches) added: same provider, ball moves at stoppages, 32-41% missing | `external/metrica.py`; labels from events | v3 (10 matches): bench Metrica 0.69, DFL 0.59; public check 0.491 (v1 0.563), macroF1 0.785 (best), D F1 0.74; D still 58% of the full match. Open question: GT D share is 20% in the public window, bench labels 35-48% | |
| 2026-11-12 | v4: dead spells at least 2 s (99% of DFL/Metrica/GT dead spells are longer; not tuned) and leaving D cheap only at a touch (restarts are kicks) | params `decode.min_dead_frames`, `lambda_d_exit*` | v3 extra D in the public window: 19 s late ends, 10 s early starts, 6 s spurious incl. 0.2-0.7 s blips. v4 bench Metrica 0.681, DFL 0.587; public check 0.521, 16 vs 17 possessions, mean 13.4 vs 14.2 s, frag 0.95; spellF1 0.67 @0.3 vs 0.24 @0.7 -> boundaries are the loss | |
| 2026-11-12 | Stage 6 boundary snapping implemented, kept off | `decode.refine: false` | bench: 1 s window +0.001, 2-3 s windows -0.002..-0.005. Public-window convention check: GT switches come 0.5-2 s before our first detected touch by the new team (restarts 1.3-1.5 s before the restart touch); nearest-team flips are within ~0.7 s but flicker. GT is from corrected tracking we do not have; decoder timing is already centred (median lag 0 bench, -1 frame public) | |
| 2026-11-12 | Metrica labels: dead ball ends at the first action after the ball goes out (set piece, pass, carry, shot, recovery), not the next SET PIECE | `external/metrica.py` | Metrica logs goal kicks as the keeper's PASS, so goal-kick stoppages ran to the next throw-in/corner. Fixed: D share 48/42/35% -> 34/32/27%, dead spells median 12-18 s, 3.4-3.9 possessions/min (public GT 3.4) | |
| 2026-11-12 | Model A averaged with a Metrica-only dead model that keeps ball-motion features | `model.dead_metrica_full` | Metrica dead/alive acc 0.82 -> 0.91-0.95; DFL dead acc 0.66-0.76 (frozen ball unseen by the Metrica model; DFL no longer a dead-ball test). Bench Metrica 0.784 | |
| 2026-11-12 | Dead-ball dials tuned on Metrica only, team dials on all 10; min possession >= 10 frames | outputs/tune_model6.json | Metrica optimum = v5's dead settings; v6 public check 0.514 (v4 0.521, v5 0.514), switches within 1 s 0.38 (best), D precision 0.72 recall 0.45, full-match D 47% | |
| 2026-11-12 | v7: weak dead/alive labels from our own match (+-2 s around 60 Smart Tagging set pieces = dead; fast ball + touch, 20 s from any tag, model says live = live; public window excluded), weight 5, mixed into both dead models. Separate our-match model averaged (v8) or either-or (v9) rejected | `model.own`, combine none | public check: v7 spellF1 0.588 (best; v1 0.563, v6 0.514), macroF1 0.856, D F1 0.85 (P 0.79 R 0.92), switches within 1 s 0.47; v8 0.577; v9 0.092 (D on 81% of the match). v7 full-match D 51%: to check | |

## Phase 0 audit

- ball-gap histogram (tactical, graded frames): ball missing 40.7%; 150 gaps, length p10/50/90/99
  = 0.5 / 10.7 / 33.8 / 104.8 s; visible runs 2.2 / 13.8 / 61.7 / 127.8 s. Players within 5 m of the
  ball just before a gap: 1.60 vs 1.41 on all visible frames (crowding raises the gap rate a little).
- track slots (tactical, graded): 46 slots; fragments p10/50/90 = 2.6 / 35 / 285 s; 20.5 visible
  players per frame, > 22 on 2.4% of frames; same-team pairs within 1 m: 0.08 per team-frame.
- ball / player jitter:
- public-GT boundaries (frame, type, cause, ball visible?, GT timing vs first touch):
- attacking direction per half:

## Ablations

| Variant | public spellF1 (95% CI) | boundary error (median frames) | possessions (full match) | kept? |
|---|---|---|---|---|
