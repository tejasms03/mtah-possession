# La Remontada — frame-level possession for BAR-ATH

More Than A Hack 2027 · Technical document · Code: https://github.com/tejasms03/mtah-possession

**Submission:** `la remontada.csv`, SHA-256 `b886ae5807d19e3bea7d79f949655e890538b1d891bc2eb5b55bfe96298c7158` (version v22).
`make submission TEAM=la_remontada` regenerates it byte for byte, which we checked from a fresh clone.

## 1. Approach in one paragraph

We label every frame H, A or D in four steps:

1. **Per-frame experts.** Three gradient-boosted tree models (scikit-learn `HistGradientBoostingClassifier`)
   read features built only from the tactical tracking (ATD):
   - **A**, the dead-ball model: is play dead?
   - **B**, the team model: which team has the ball? It is team-blind, so every row is also shown with the
     two teams swapped.
   - **a change detector:** does control pass from one team to the other within ±0.5 s?
2. **A tree sequence model** re-reads B's and the detector's outputs at 15 time offsets, from -4 s to +4 s,
   so it sees the *shape* of a turnover (pass, opponent arrives, play goes on). A single frame cannot show that.
3. **A network sequence model (TCN)** re-reads the first-stage outputs (team, change-detector and dead
   log-odds) together with ball visibility, ball speed, each team's distance to the ball and touches, over
   about ±20 s. It is a small non-causal dilated temporal convolution network (64 channels, kernel 3, dilations
   1 to 256, residual blocks; about 150k weights) with an H/A/D head and a turnover head. It is team-blind: it
   runs on the inputs and with H and A swapped, and the two answers are averaged. Its H/A/D log-probabilities
   are blended 50/50 into the decoder's per-frame evidence. It was trained offline in PyTorch on the 10 outside
   matches only (`tools/train_tcn.py`). The weights are committed (`models/tcn.npz`) and the pipeline runs them
   with plain numpy, so the submission needs no PyTorch and is byte-identical on every run.
4. **A Viterbi decoder** over H/A/D turns these per-frame log-odds into possessions. It uses switch costs that
   fall where the change detector fires, a minimum length for dead spells and for possessions, and the
   possession conventions of the training labels (section 4).

**No model is trained on BAR-ATH.** All training and every design choice use outside matches only. To make
those matches look like ours, we damage their tracking (section 3). The ranking metric is spellF1, which scores
whole possessions, so we optimised and chose versions by spellF1 on held-out outside matches, never by the
public 5 minutes.

**Why this design rather than another:**
- **Trees first, then a small network.** Our inputs are tabular, so trees do the per-frame work. A small
  sequence network on top of them adds longer context (±20 s) cheaply.
- **A decoder.** spellF1 punishes flicker (one wrong frame splits a possession into three), and a constrained
  decoder fixes exactly that.
- **Two separate questions.** Dead ball and team need different evidence, so each gets its own model.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (tracking) | **Used.** The only model input. Cleaned: off-pitch ball points, duplicate players and impossible jumps removed. The ball is smoothed and short gaps bridged; touches are detected from velocity changes with a player nearby. |
| Smart Tagging | **Used at inference only, for one constraint:** frames between each half's start and its "Kick Off (Start)" tag are dead ball. Never used for training or tuning. Set-piece hints were tested and dropped (they hurt on every held-out match). |
| Homography, video, broadcast tracking | **Not used.** We made no vision attempt. The ATD is already in pitch coordinates. |
| Public 5-minute ground truth | **Format and correctness check only.** See the disclosure in section 7. |

## 3. Training data

All labels below are algorithmic; none are hand-labelled by us and **none come from BAR-ATH**.

| Dataset | Source / licence | Labels: who and how | Size |
|---|---|---|---|
| IDSSE / DFL Bundesliga | Bassek, Rein, Weber, Memmert (2025), *Scientific Data* 12, 195. Data © DFL, CC-BY 4.0 | From the provider's own fields: D where `BallStatus` is dead, otherwise the `BallPossession` team. This is the same mechanical rule as the organisers' ground truth. Team blips (A-H-A) shorter than 1.5 s are merged (section 4) | 7 matches (J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9) |
| Metrica Sports sample data | github.com/metrica-sports/sample-data, used with acknowledgement | From Metrica's own event data. Dead ball runs from the ball going out to the first action after it; otherwise the team of the last on-ball event | 3 matches |

**Matching the damage.** We damage both sets to look like our tracking, using only label-free summary
statistics of BAR-ATH's tracking:
- the share of frames with the ball missing (40.7%) and the lengths of the ball gaps;
- player slot fragmentation and duplicates;
- ball loss at stoppages (DFL freezes its ball when play is dead; ours disappears instead);
- short ball flickers.

This one change ("faithful" damage) gave our biggest gain: +0.044 on the held-out matches.

SkillCorner open data was tested and dropped: it hurt on the held-out matches.

## 4. Possession rules we implemented

These follow the training labels' conventions, because spellF1 rewards matching them:

- **Dead ball:**
  - minimum 2 s (98.9% of DFL and 95.3% of Metrica stoppages are longer);
  - all frames before each half's kick-off tag are dead;
  - play leaves D cheaply only at a detected touch (restarts are kicks).
- **Loose balls and challenges:**
  - the team label changes only where the evidence supports it;
  - a switch is cheaper where the change detector fires;
  - a blip back to the previous team shorter than 1.5 s (A-H-A) is merged away, as in the training labels;
  - minimum possession 0.4 s.
- **Restarts and stoppages:** the last team to touch the ball within 2 s before a stoppage owns the ball up to it.
- **Missing ball:** while the ball is missing, an extra bias keeps the label live, because our tracking loses
  the ball in live play far more often than the training data does.

## 5. Results and what we tried

**Every spellF1 number in this document is from our outside-match bench, never from BAR-ATH.** The public
5 minutes were only scored to check the file and the pipeline work; those numbers are not reported or used.

**Our test bench.** We hold out each of the 10 outside matches in turn, train on the other 9, and score the
held-out match with the organisers' `possession_metrics.py` (`spell_f1_mean`). The tracking is damaged as in
section 3.

**Adoption rule (v2), set before the later results were seen.** A change is adopted only if, against the
current version:
- the mean gain is at least +0.005;
- at least 8 of 10 matches improve;
- no match drops by more than 0.010;
- a neighbouring setting agrees;
- the block-bootstrap 90% range is above 0.

| Version | Change | Bench spellF1 (10 held-out outside matches) |
|---|---|---|
| v17 | "faithful" training damage | 0.682 |
| v18 | player-motion inputs for the dead-ball model, plus a live-play bias while the ball is missing | 0.698 |
| v19 | decoder matched to the label conventions (blip merge, last touch before a stoppage, switch costs) | 0.710 |
| v21 | turnover inputs for the change detector, plus the tree sequence model | 0.722 (DFL 0.705, Metrica 0.763) |
| **v22 (submitted)** | network sequence model (TCN) blended into the decoder evidence | **0.739** (DFL 0.722, Metrica 0.780) |

For v21 against v19: +0.012, 9 of 10 matches better, bootstrap 90% range +0.007 to +0.019.

For v22 against v21: +0.017, 10 of 10 matches better (DFL +0.017, Metrica +0.017), bootstrap 90% range +0.013 to
+0.023. A second training seed and the neighbouring blend weights (0.25, 0.75) give the same. The gain is almost
all live vs dead ball: live/dead mix-ups where the ball is seen fall from 2.3% to 0.6% of frames. Wrong-team frames
and quick turnovers are unchanged. Using the turnover head as the change signal, or blending only where the ball
is hidden, added nothing and is off. On our match, v22 outputs 295 possessions (v21: 273).

**Tried and dropped (details and numbers in `docs/DECISIONS.md`):**
- 13 hand-written football rules (best +0.0004);
- a ball position guessed from the players;
- gap anchoring;
- extra cuts at change-detector peaks;
- a boundary refiner (+0.0025);
- carrier and ball-control inputs (help Metrica, hurt DFL);
- restart-layout inputs;
- a blend of v21 with the carrier model (about 0);
- a learned stacking layer;
- averaging 5 model copies (a tie);
- SkillCorner data;
- Smart Tagging set-piece hints.

## 6. Limitations

- **We output too few possessions:** 2.9 per minute against 3.6 in the outside matches' labels. Possessions
  shorter than 4 s are found only 32-55% of the time. If team calls were perfect while the ball is visible, the
  bench score would rise from 0.71 to 0.84. Quick turnovers are the main remaining error.
- **While the ball is missing** (about 41% of our match), team calls are weaker, and some live play is still
  called dead.
- **Our test matches are proxies.** The damage is matched to our match's statistics, but DFL and Metrica
  tracking differ from ours in ways we cannot fully copy.
- **With more time we would try:**
  - several damage seeds per training match;
  - PFF World Cup data, which has real occlusions.

## 7. Disclosures

- **Early versions used our match in ways we later removed.**
  - v7-v12 trained with weak, algorithmic labels taken from our own match (Smart Tagging set pieces and runs
    of same-team touches).
  - Versions up to v12 were partly compared on the public 5-minute window.

  After the organisers' clarification (no training on the provided match) and EVALUATION.md (no choosing by
  the public window), both were switched off and all later choices were made on outside matches. The
  submitted v22 uses neither.
- **The TCN idea.** The design follows a generic published architecture (dilated residual temporal convolutions).
  Another team's public write-up (Opus-Operandi) prompted us to try it; we did not use their code. All TCN code
  is ours.
- **AI tools.** An AI coding assistant (Claude by Anthropic, through Claude Code) helped write and debug code,
  run experiments, and draft documentation. **No competition data was shared with any AI service:** no video,
  tracking, Smart Tagging, homography, public ground truth or output labels. The assistant saw only code,
  documentation and summary numbers computed locally. Every label comes from the pipeline in the repository;
  none was written by hand or by an AI.

## 8. Reproduction

```bash
git clone https://github.com/tejasms03/mtah-possession.git && cd mtah-possession
make setup                                            # pinned environment (uv)
# put the BAR-ATH bundle in data/ (data/README.md); clone the organisers' repo into vendor/mtah
uv run possession check-data
uv run possession idsse                               # 7 DFL matches (public); Metrica downloads on first use
make submission TEAM=la_remontada                     # ~1 h 40 min, single-threaded
make verify                                           # byte-identical check against SUBMISSION_SHA256
```

Retraining the TCN weights is optional and needs PyTorch 2.x (see README); the submission run itself does not.

**Reproducibility.** The run is single-threaded and uses fixed seeds and a pinned lock file. It rebuilds all
features and models from scratch and writes `outputs/la_remontada.csv`, which is identical to the submitted
file. We confirmed this with a fresh clone that shared only the raw downloads.
