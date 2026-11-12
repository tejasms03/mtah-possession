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
