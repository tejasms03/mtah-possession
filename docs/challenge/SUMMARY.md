# Challenge summary

Our summary of the organisers' CHALLENGE, DATA, EVALUATION, SUBMISSION and RULES documents.
The originals are in the organisers' repository (`vendor/mtah/`); where they disagree with this
file, they win.

## Task

One label per frame of the tactical video, BAR-ATH (Barcelona = H, Athletic = A), 199,319 frames
at 25 fps. Graded: 1st half 11,185–90,651 and 2nd half 110,808–190,197 (158,857 frames).

| Label | Meaning |
|---|---|
| H | Barcelona in possession, ball alive |
| A | Athletic in possession, ball alive |
| D | dead ball, no owner asked for |
| X | no answer; use only on ungraded frames |

- Loose or contested ball stays with the last team. Change = an opponent gains **controlled**
  possession. Not a change: touch without control, deflection that does not intercept, aerial
  duel without subsequent control.
- A possession is a maximal run of H or A. `H H D H H` is two possessions.
- The GT comes from corrected tracking (dead flag + owning team, nothing smoothed). The public
  5 minutes is the authority on what those mean.

## Scoring

- **spellF1** (ranking): possessions matched by temporal IoU, averaged over 0.3–0.7. X is
  forward-filled with the previous label. Noise band ~0.04. Floor: `baseline_smooth50` = 0.2433.
- **macroF1** (tie-break): per-frame F1 averaged over H, A, D. X counts as a miss.
- No leaderboard. Only the public window and `validate_submission.py` before the deadline.

## Given

| Asset | Notes |
|---|---|
| Tactical ATD | ball x,y; ~20 players x,y,speed,team; 0–1 coords (×105.3, ×68.0 m); ball missing 41% graded; no IDs; row = frame_id − 1 |
| Smart Tagging | 638 automatic intervals, frame = round(s × 25); throw-ins have Team = N/A |
| Homography | tactical valid everywhere; visual debugging only |
| Broadcast feed | not synced; not used |
| Public GT | frames 11,185–18,684, 17 possessions; everything else X |

## Rules

1. Labels must come from an algorithm. No hand annotation or hand-correction of output.
2. No private GT, and no other corrected possession/event data **for this fixture**.
3. Repo + data must regenerate the **identical** file.
4. Everything used must be accessible to the organisers and explained in the document.
5. Never commit the match data or publish derived data that reconstructs it.
6. Teams of 3–5 for the Innovation Challenge and Final Pitch.

## Deliverables (deadline 17 Nov 2026, one Google Form, editable)

1. `<team_name>.csv`: header `frame,label`, 199,319 rows, frames 0..199318 in order, H/A/D/X.
2. Public repository: code, pinned deps, run instructions, one command to regenerate. No data.
3. `<team_name>.pdf`: approach, assets used/dropped, possession rules, limitations, reproduction.
