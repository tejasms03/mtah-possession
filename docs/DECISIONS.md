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
| 2026-11-07 | Own ATD parser instead of kloppy for the raw txt | | format: `frame:` 50 slots `x,y,speed` `;`-separated `:` ball `x,y,z` (z always NaN); kloppy only needed for slot→team | |
| 2026-11-09 | Plans merged: dead ball and boundary timing are the main error, not team identity | | team PDF: baseline 83–89% team-correct on live frames, D recall 8–34%; ball out ~40% of LaLiga time (CIES) | |
| 2026-11-09 | Soft switch cost λ_t = λ0 − α·b_t instead of hard touch gating | | one DP pass, cheap switches at real events; replaces a hard rule that fails when a touch is missed | |
| 2026-11-09 | Multi-scale windows (±0.4 s … ±40 s) instead of ±2 s only | | Lang et al. 2022: time-shifted player features to ±80 s give 92% dead/alive, 0.80 stoppage F1, no ball | |
| 2026-11-09 | Back-date to first touch, then subtract median lag measured on damaged IDSSE | | both plans; lag is systematic for centred-window rules | |
| 2026-11-09 | Metrica sample events used to check the dead-ball convention | | same provider as the challenge; IDSSE convention undocumented | |
| 2026-11-09 | Whistle detection optional, only if the tactical video has audio | | PDF: 93% recall / 88% precision on broadcast (Raventós 2015); DATA.md does not mention audio | |
| 2026-11-09 | No video ball detection (WASB), no homography for ball-out | | team decision (no vision); GPU-hours to reproduce; ATD already in pitch coordinates | |

## Phase 0 audit

- ball-gap histogram:
- ball / player jitter:
- public-GT boundaries (frame, type, cause, ball visible?, GT timing vs first touch):
- attacking direction per half:

## Ablations

| Variant | public spellF1 (95% CI) | boundary error (median frames) | possessions (full match) | kept? |
|---|---|---|---|---|
