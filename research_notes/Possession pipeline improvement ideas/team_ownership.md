# Improving per-frame team ownership detection (short possessions, ball-missing switches)

Context: v12 pipeline (HGB model B team-blind P(own) -> 3-state Viterbi H/A/D). 25 fps, ball missing 41% of frames, no player identities. Training data: 7 degraded DFL/IDSSE + 3 Metrica. Remaining errors: short possessions (0.8–8 s). Ground-truth switches land 0.5–2 s before our first detected touch. Metric: spellF1 (IoU 0.3–0.7). Constraints: CPU, deterministic, about 1.5 days.

## Q1. Published methods for possession from player positions when the ball is unobserved: what drove their gains, and what can become GBDT features in hours?

### Takeaway
Three papers are directly relevant. PathCRF (2026) was trained and tested on the same Sportec/DFL matches we use, with no ball input, and reaches **92.4% frame-level team-possession accuracy**. Ballradar reaches 85.3% team-level accuracy and Borghesi's TCN 85–86%. The gains that transfer to our setup come from three things: (a) a structured decoder whose **transition scores depend on context** rather than being constant, (b) **auxiliary/intermediate supervision** (predict the possessor or the sender/receiver first, then team possession), and (c) team-aware set encodings of all players. The neural backbones themselves cannot be rebuilt in 1.5 days on a CPU, but (a) and (b) can be imitated with a second gradient-boosting model.

### Cited Findings
- **PathCRF (Kim et al., arXiv 2602.12080, Feb 2026)**
  - Detects events from player tracking alone. Players are nodes of a fully connected dynamic graph, and possession at each step is one selected edge out of 676 (22 players + 4 "outside" nodes, squared).
  - A CRF with Viterbi decoding enforces a consistent sequence. Both emission and transition scores come from Set-Attention edge embeddings. — [arXiv abs](https://arxiv.org/abs/2602.12080)
  - Data: Sportec Open DFL dataset, 7 matches (5 train / 1 val / 1 test). Downsampled 25 fps -> 5 fps, 10 s windows, predictions upsampled back to 25 fps. Missed ball touches were added to the labels using RDP + Needleman-Wunsch alignment, and event times were synced with ELASTIC. — [arXiv HTML](https://arxiv.org/html/2602.12080)
  - Results:
    - **Frame-level team possession accuracy 92.40%**.
    - Predicted home possession 62.64% vs ground truth 62.13%.
    - Edge accuracy 69.64%; event F1 75.69% (77.21% at 25 fps, at about 12x the training cost).
    - Sender accuracy 80.44%, receiver accuracy 82.28%. — [arXiv HTML](https://arxiv.org/html/2602.12080)
  - Transition score: a shared MLP over the [previous edge, current edge] embeddings, so the cost of switching depends on context. Impossible transitions are masked at −1e4; allowed ones are identity, kick, receive and out-of-play. — [arXiv HTML](https://arxiv.org/html/2602.12080)
  - Event F1 by decoder:

    | Decoder | Event F1 |
    |---|---|
    | Ball-TP baseline (Ballradar-style ball regression + rules) | 48.39% |
    | Non-CRF + greedy | 65.34% |
    | Non-CRF + Viterbi constrained decoding | 73.14% |
    | Static masked CRF | 73.83% |
    | Dynamic dense CRF (no masking) | 67.82% |
    | **Dynamic masked CRF** | **75.69%** |

    Constrained Viterbi alone adds about +8 points; context-dependent transitions add about +2 more. — [arXiv HTML](https://arxiv.org/html/2602.12080)
  - Loss ablation (dynamic masked CRF): removing both auxiliary losses (coarse sender/receiver logits and emission loss) drops event F1 from 75.69% to 69.65%. — [arXiv HTML](https://arxiv.org/html/2602.12080)
  - Backbone ablation: Bi-LSTM vs positional-encoding SAB temporal module made only a marginal difference (F1 75.24% vs 75.69%). The team-aware social module (PPE-FPE, per-team ISAB plus all-player ISAB) gave the largest gain over plain SAB (F1 68.19% -> 75.69%). — [arXiv HTML](https://arxiv.org/html/2602.12080)
  - Cross-provider transfer is weak:
    - Sportec leave-one-match-out: 73.88 ± 1.90% F1
    - K League -> Sportec: 66.25 ± 7.65%
    - Sportec -> K League: 49.16%

    The authors attribute the drop to differences in collection modality and annotation conventions. — [arXiv HTML](https://arxiv.org/html/2602.12080)
  - Cost: one RTX 4090, 565 s per epoch at 5 fps; inference about 23 s per match. — [arXiv HTML](https://arxiv.org/html/2602.12080). Code: https://github.com/hyunsungkim-ds/pathcrf — [search summary of arXiv abstract](https://arxiv.org/abs/2602.12080)
- **Ballradar (Kim et al., KDD 2023, arXiv 2306.08206)**
  - Architecture: hierarchical. A per-team Set Transformer encoder feeds a Player Possession Classifier, whose output feeds the ball-trajectory Bi-LSTM.
  - Inputs per player: x, y, vx, vy, speed, acceleration.
  - Training samples: 10 s windows at 10 Hz, with random pitch flips as augmentation.
  - Training data: 15 GPS matches (Fitogether) + 3 Metrica matches. — [arXiv PDF](https://arxiv.org/pdf/2306.08206), [GitHub](https://github.com/pientist/ballradar)
  - Results: H-LSTM-RL reaches player-level possession accuracy 64.70% and **team-level possession accuracy (TPA) 85.85%**. H-Transformer reaches only about 78%, and the authors argue recurrence suits the task because the latest positions matter most. — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
  - Making the model hierarchical (predict possession first) cut ball position error from 5.47 m (plain LSTM) to 3.69 m. — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
  - Rule-based postprocessing:
    - Possession score = possession probability / distance to the predicted ball.
    - A score above 0.5 counts as a touch.
    - A score between 0.2 and 0.5 counts as a touch only at a local maximum, because one-touch passes don't give the probability time to rise.
    - Otherwise the ball is in transit. — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- **TranSPORTmer (Capellera et al., ACCV 2024, arXiv 2410.17785)**
  - Data: LaLiga 2022-23, 283 matches; 9.6 s windows at 6.25 Hz.
  - Inputs: positions plus an agent-type token (ball / offense / defense); no velocities.
  - A CLS token classifies 4 states (pass, possession, uncontrolled, out-of-play), with about 80.84% state accuracy when the ball is 100% masked. The uncontrolled class is weakest (subjective labels, class imbalance).
  - Ball inference error: 2.71 m vs Ballradar 3.89 m. Ballradar without possessor classification: 5.43 m. — [arXiv HTML v2](https://arxiv.org/html/2410.17785v2)
  - The model was trained on 283 matches on an A6000 GPU for 100 epochs. — [arXiv HTML v2](https://arxiv.org/html/2410.17785v2)
- **Borghesi et al. 2023 (Expert Systems with Applications 223:119780, TCN)**
  - Data: 35 Serie A matches at 16 fps. The model jointly classifies dead/alive and possession, uses a permutation-invariant aggregation (sum / self-attention / 2D-conv), and reaches 86.39% global accuracy with the best variant.
  - It beats the best rule-based method (Morra et al. 2020) by about 7 points. — [author manuscript](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
  - Object/channel ablations (possession accuracy):

    | Input | Possession accuracy |
    |---|---|
    | Full input | 84.95% |
    | Ball + 5 nearest players | 84.98% (no loss) |
    | Ball only | 50.54% (chance) |
    | x, y + roles/team (no velocity or hand-crafted channels) | 82.09% |
    | x, y only (no team/role flag) | 51.89% |

    So **5 nearest players + team flag carry nearly all the possession signal, and hand-crafted kinematic channels add about 3 points**. — [author manuscript](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
  - A single-network multi-task design (dead/alive + possession) beat two separate networks; the authors attribute this to multi-task regularisation. — [author manuscript](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)

### Inferences
- **Benchmark.** PathCRF's 92.4% frame-level team accuracy on DFL with *no ball at all* is a useful reference ceiling for our ball-missing frames. If v12's accuracy on ball-missing in-play frames is far below about 90%, more model capacity (not just decoding) is the bottleneck there.
- **Learned, context-dependent switch costs (GBDT version of PathCRF's dynamic transitions).**
  - Train a second HGB "switch hazard" model: P(team possession changes within ±k frames of t | features). Use −log p (scaled) as the Viterbi switch cost, replacing or adding to the touch-gated constant.
  - This mirrors PathCRF's static -> dynamic transition gain (+1.9 F1 points) and its emission/transition decomposition.
  - Effort: about 4–6 h (label generation, model, decoder hook, tuning the scale). Risk: medium (needs re-tuning of minimum lengths).
- **Possessor-level auxiliary target (GBDT version of "hierarchical/intermediate supervision").**
  - Per frame, for each of our 5 nearest players per team, train a per-player HGB model "this player is the carrier/next receiver" (team-blind, mirrored).
  - Then add max/sum of these per-team probabilities, and their difference, as features to model B.
  - Ballradar, TranSPORTmer and PathCRF all report large gains from an intermediate possessor target: Ballradar 5.47 -> 3.69 m; PathCRF −6 F1 points without auxiliary losses.
  - Labels need the possessing player id. That is available from DFL events (player IDs), but our track slots swap, so labels must be assigned by nearest-to-ground-truth-ball at the touch frame.
  - Effort: about 5–8 h. Risk: medium-high (labelling through identity swaps).
- **Cheap structural features copied from the set encoders.** Per team: sorted distances of all visible players to the estimated ball (or to the opponent centroid when the ball is missing), team centroid velocity projected on the attacking direction, and spread/compactness rate of change. Borghesi's ablation suggests roles plus coordinates already carry most of the signal, so expect modest gains (≤1–3 points) on top of v12's existing shape features.
- **Transfer risk.** Because of PathCRF's cross-provider drop, our Metrica + DFL training may transfer imperfectly to Barcelona–Athletic broadcast tracking. Leave-one-match-out validation should report per-provider scores.

### Gaps
- None of the papers report accuracy stratified by possession length or near switches, so there is no published number for "short possession" performance.
- PathCRF's exact node input features and training hyperparameters (epochs, learning rate) were not in the portions I could read.
- I did not find a published "possession from team shape only" paper with numbers beyond Borghesi's ablations.

## Q2. Small temporal models (1D CNN/TCN/BiGRU) vs gradient boosting on windowed features; deterministic CPU training feasibility

### Takeaway
The published evidence does not favour transformers, and only weakly favours recurrence. In Ballradar, Bi-LSTM beat Transformer by about 7 TPA points. In PathCRF, Bi-LSTM and attention were within about 0.5 F1. Borghesi's architecture variants spanned only 82.9 ± 1.7%. Most of the gains came from structure (decoding, auxiliary targets), not the sequence backbone. With 1.5 days left and a byte-reproducibility requirement, **stacked sequential learning with HGB is a better bet than a new PyTorch model.**

### Cited Findings
- Ballradar: H-LSTM-RL TPA 85.85% vs H-Transformer-RL 78.01%. The authors argue RNNs' bias toward the latest timestep helps, and that recurrence produces smoother outputs. — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- PathCRF: swapping the temporal module between Bi-LSTM and SAB changed F1 by only about 0.45 points (75.24% vs 75.69%). — [arXiv HTML](https://arxiv.org/html/2602.12080)
- Borghesi TCN: the mean accuracy over all design variants was 82.93 ± 1.71%, with the best at 86.39%. — [author manuscript](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- PyTorch determinism:
  - Reproducibility is "not guaranteed across PyTorch releases, individual commits, or different platforms", and not between CPU and GPU.
  - `torch.use_deterministic_algorithms(True)` makes known-nondeterministic ops throw an error, and deterministic ops can be slower.
  - LSTM nondeterminism is documented only for CUDA. — [PyTorch randomness notes](https://pytorch.org/docs/stable/notes/randomness.html)
- Stacked sequential learning (Cohen & de Carvalho, IJCAI 2005):
  - Train a base learner. Get its K-fold cross-validated predictions on the training set, and feed the predictions of neighbouring examples (window W before and after) into a second learner.
  - On a signature-detection task, error fell from 3.20% (base ME) to 0.71% with W = 20, a 46% reduction relative to a CRF (1.17%). Stacked ME beat CRFs on 7 of 9 tasks.
  - Training costs about 7x the base learner (K + 2 fits). The point is to fix the mismatch between training on true neighbouring labels and testing on predicted ones. — [IJCAI 2005 paper](https://wwcohen.github.io/postscript/ijcai-2005-stack.pdf), [mlanthology](https://mlanthology.org/ijcai/2005/cohen2005ijcai-stacked)
- A multi-scale extension of stacked sequential learning (Puertas, Escalera, Pujol) reportedly outperforms standard stacked sequential learning. — [Springer chapter](https://www.springerprofessional.de/en/multi-scale-stacked-sequential-learning/3117412)

### Inferences
- **Recommended: two-stage HGB with leave-one-match-out (LOMO) out-of-fold stage-1 predictions.**
  - We have 10 training matches, so K = 10 folds by match, giving 11 fits of model B. If one model B fit takes M minutes, stage-1 costs about 11·M.
  - Stage-2 features:
    - P(own) at lags {−50, −25, −12, −6, 0, +6, +12, +25, +50} frames
    - multi-scale window mean/min/max/slope of P(own)
    - frames since and until P(own) last crossed 0.5
    - local entropy
    - the existing touch features
  - Effort: about 3–5 h. Risk: low-medium. Fully deterministic with sklearn HGB given a fixed `random_state` and no early-stopping validation shuffle.
  - This is the closest CPU-friendly analogue of a BiGRU over per-frame features, and it directly targets short possessions: stage 2 can learn that "a 1.5 s dip of P(own) right after a touch" is a real possession.
- **If a neural model is attempted anyway.** Use a 1D dilated CNN over the already-computed per-frame feature vectors at 5 fps (1.5M frames -> about 300k steps), with about 4 layers, kernel 5, dilations 1-2-4-8 (about ±6 s receptive field).
  - Use `torch.manual_seed`, `torch.use_deterministic_algorithms(True)` and `torch.set_num_threads(1)`; avoid DataLoader worker shuffling nondeterminism by using a fixed generator.
  - Byte-identical results are plausible only on the same machine and the same torch version (per the PyTorch docs). The `set_num_threads` effect is my inference, not documented in the sources found.
  - Effort: about 8–12 h including tuning. Risk: high for the deadline.

### Gaps
- I found no published head-to-head of GBDT-on-windowed-features vs TCN/GRU for possession specifically.
- I found no documentation confirming that CPU PyTorch LSTM/Conv1d outputs are bitwise identical across thread counts.

## Q3. Features that capture possession CHANGES (reception/interception geometry, ball endpoint, who the ball travels toward, pitch control at future ball position, team-shape transitions)

### Takeaway
The standard physics model is Spearman's time-to-intercept plus a logistic arrival model; Laurie Shaw's Friends of Tracking code gives ready parameter values. Applied to the ball's extrapolated path, or to any point when the ball is missing, it yields "which team reaches the ball first", which is the quantity that decides interceptions. Because ground-truth switches are 0.5–2 s *before* our detected touch, features computed at the *kick/release* moment and along the predicted flight help more than features at the touch.

### Cited Findings
- Friends of Tracking pitch-control defaults (Spearman model, `Metrica_PitchControl.py`):

  | Parameter | Value |
  |---|---|
  | max player speed | 5.0 m/s |
  | reaction time | 0.7 s |
  | time-to-intercept sigma | 0.45 s |
  | control rate lambda (att = def) | 4.3 /s (kappa_def = 1; the paper used 1.72) |
  | goalkeeper lambda | 3x |
  | average ball speed | 15 m/s |
  | integration step | 0.04 s |

  - Time to intercept = reaction time + distance from the reaction-projected position / v_max.
  - P(arrived by T) = logistic((T − tti) · π / (√3·σ)).
  - Ball travel time = distance / 15 m/s. — [LaurieOnTracking Metrica_PitchControl.py](https://raw.githubusercontent.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/master/Metrica_PitchControl.py)
- Bekkers' "Pressing Intensity" (arXiv 2501.04712) extends time-to-intercept with a direction penalty:
  - T = τ_r + ‖target − (r_i + v_i τ_r)‖ / v_max + ‖v_i‖ · β / π, where β is the angle between the player's velocity and the target direction.
  - p = logistic with σ = 0.45 and T threshold 1.5 s.
  - Team pressure on a target = 1 − ∏(1 − p_i), assuming independent defenders. A player below 2 m/s is treated as not actively pressing. — [arXiv HTML v2](https://arxiv.org/html/2501.04712v2)
- Ballradar's postprocessing separates "in transition" from "controlled" using possession probability / distance-to-ball. Low-confidence local maxima are treated as one-touch receptions. — [arXiv PDF](https://arxiv.org/pdf/2306.08206)
- Hudl's cover-shadow work models interception by combining defender time-to-intercept with ball travel time to the target. — [Hudl research PDF](https://static.hudl.com/craft/performance-insights-research-stage/2025/Quantifying-Off-Ball-Defensive-Impact-through-Cover-Shadows-Lorenzo-Casciolio-Allen-Wang.pdf)

### Inferences (concrete feature recipes; effort and risk are my estimates)
1. **Ball-flight interception race** (about 3–4 h, low-medium risk).
   - At each frame with a visible, moving ball (speed > ~5 m/s), sample points along the extrapolated straight line at 0.25–2.0 s ahead. Use the observed ball speed with simple linear decay, or 15 m/s if unknown.
   - For each point, compute min time-to-intercept per team (Spearman parameters above) minus ball arrival time.
   - Features:
     - `race_margin` = min over points of (opponent TTI − own TTI) at the first point where any player can arrive
     - the "first-arrival team" probability using the σ = 0.45 logistic
     - the distance along the path to that point
   - Carry these forward through ball gaps with "time since last valid computation".
   - This directly encodes interceptions and should fire before the receiving touch.
2. **"Ball heading to" with velocity-aware TTI** (about 1–2 h, low risk). We already have heading-to-team. Replace the angular measure with Bekkers' direction-penalised TTI to the ball's predicted position 0.5 s and 1.0 s ahead, and take the per-team min and the difference.
3. **Ball-missing frames: pressing-focus / carrier-likelihood features** (about 3–4 h, medium risk).
   - Compute a Bekkers pressing matrix between each team's players and each opponent.
   - The likely carrier is the player under the highest summed opponent pressure *and* with teammates moving away from them (offering support / making runs).
   - Features: max pressure received per team, top-2 gap, and the team of the argmax.
   - Rationale: defenders converge on the carrier, and PathCRF/Ballradar both work without the ball. Not validated in the literature for this exact feature.
4. **Team-shape transition features** (about 2–3 h, low risk).
   - Use differences of existing shape features over the last 1, 2 and 4 s: centroid velocity along the attacking axis, length/width rate of change, and the defensive-line (deepest 4 outfield players) mean velocity.
   - Counter-press after losing the ball shows as forward velocity and compaction toward the ball. Retreat shows as negative axis velocity of the whole team.
   - Use as both lagged and lead (centred) features, since the decoder is offline.
5. **Boundary-offset correction** (about 1–2 h, low risk, probably the cheapest IoU gain).
   - Ground-truth switches precede our detected touch by 0.5–2 s. On a 2 s possession, a 1 s offset alone caps IoU at about 0.33–0.5.
   - Measure the signed offset distribution (ground-truth switch − decoded switch) on the training matches, split by context (ball visible vs missing, pass vs interception), then shift decoded boundaries earlier by the median per context.
   - Alternatively, train the stage-2 / hazard model on a target that is "switch at t", so boundaries align with the label convention.
   - Note: PathCRF needed ELASTIC sync and RDP/NW alignment to repair DFL event timing ([arXiv HTML](https://arxiv.org/html/2602.12080)). Part of our 0.5–2 s offset may be label-timing convention, not detection error.

### Gaps
- I found no paper quantifying how much pitch-control or interception-race features improve possession *segmentation*; existing uses are pass-success / xPass and pressing metrics.
- I found no published counter-press vs retreat classifier with accuracy numbers.

## Q4. Stacking/ensembling ("auto-context"), calibration, label smoothing near boundaries

### Takeaway
Stacked sequential learning (auto-context) with out-of-fold predictions is well supported. It beat CRFs on 7 of 9 tasks, with up to a 46% error reduction. It is also the most transferable idea from the PathCRF/Ballradar "hierarchical + structured decoding" results. Pair it with learned transition costs. Calibration and boundary label handling matter mainly because the Viterbi decoder consumes the probabilities as log-likelihoods.

### Cited Findings
- Stacking must use cross-validated stage-1 predictions for training. Using in-sample predictions recreates the train/test mismatch the method is meant to fix. Cohen & de Carvalho used K = 5 (K = 2 on larger tasks) and windows W = 20–25 neighbours on each side. — [IJCAI 2005 paper](https://wwcohen.github.io/postscript/ijcai-2005-stack.pdf)
- In PathCRF, masking impossible transitions and context-dependent transitions together gave the best F1. The unmasked dynamic CRF (67.82%) was *worse* than the static masked CRF (73.83%), so flexible transitions need hard constraints. — [arXiv HTML](https://arxiv.org/html/2602.12080)
- PathCRF shows auxiliary-loss supervision is essential: about −6 F1 points without it. — [arXiv HTML](https://arxiv.org/html/2602.12080)

### Inferences
- **Order of work for 1.5 days** (cumulative about 12–16 h):
  1. Boundary-offset correction (1–2 h).
  2. LOMO out-of-fold stage-2 HGB "auto-context" (3–5 h).
  3. Interception-race + direction-penalised TTI features into stage 1 (3–5 h).
  4. Learned switch-hazard costs in Viterbi, keeping the minimum-length and touch masks as hard constraints in PathCRF's masked style (4–6 h).
  5. Only if time remains: possessor-level auxiliary model.
- **Calibration.**
  - HGB log-loss outputs are usually reasonably calibrated, but the decoder adds log P. Fit isotonic regression or temperature on out-of-fold predictions and evaluate spellF1 before vs after (about 1 h).
  - Isotonic regression is deterministic. Keep a single global calibrator (not per match) to avoid overfitting 10 matches.
- **Boundary label handling.** Instead of label smoothing (HGB has no soft-label support for classification), down-weight training frames within ±0.5 s of ground-truth switches (`sample_weight` 0.3–0.5), or drop them. This reduces noise from the label-timing convention. Alternatively, add a separate "near-switch" target for the hazard model. Effort under 1 h; low risk.
- **Ensembling.** Averaging 3–5 HGB fits with different fixed seeds/subsamples is deterministic but gives small gains for HGB without subsampling. Lower priority than stacking.

### Gaps
- I found no source testing auto-context specifically for sports possession segmentation, or giving numbers for sample down-weighting near boundaries.
- I found no source on calibration effects inside HMM/Viterbi decoding for this task.
