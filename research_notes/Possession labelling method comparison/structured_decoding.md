# Structured / segment-level decoding of noisy per-frame H/A/D scores, tuned to spellF1

Context: 25 fps frames labelled H, A or D. spellF1 matches H/A runs by temporal IoU and averages F1 over IoU 0.3 to 0.7; per-frame macro F1 breaks ties. Typical possession lasts 5 to 15 s (125 to 375 frames) and dead ball 5 to 60 s. Processing is offline and deterministic, with about 200k frames per match.

Scope note: this pass used 15 tool calls. Primary sources were fetched for MS-TCN, ASRF, the Ding et al. TPAMI survey, the Murphy segment-model tutorial, the ruptures docs and the VBx diarization papers. Some leads, such as hmmlearn internals, CROPS and expected-F1 decoding, are flagged as Gaps rather than reported from memory.

## Q1. HMM vs explicit-duration HSMM vs semi-CRF vs penalised change-point vs median/hysteresis filters: which reduces over-segmentation best?

### Takeaway
All the structured options are exact dynamic programs over our 3 states. Their main difference is how they model duration. An HMM's self-loop gives geometric durations, whose mode is 1 frame. An HSMM, or an HMM with a chain of states per label, enforces a minimum and a realistic length distribution. In the action-segmentation literature, explicit segment-level post-processing (boundary-based relabelling, or removing short segments) beats Gaussian or median-type smoothing by about 2 to 4 F1 points. Every structured method beats no post-processing by 5 to 9 points.

### Cited Findings
- **HMM equals geometric duration.** In a segment model where p(l|q) is geometric, "this becomes a regular HMM". A variable-duration HMM can be approximated by adding extra states to a regular HMM, which gives a mixture of geometric distributions. — [Murphy, "Hidden semi-Markov models (segment models)"](https://www.cs.ubc.ca/~murphyk/Papers/segment.pdf)
- **HSMM complexity.** Forward-backward for a segment model is O(T·Q²·L²) naively. It drops to O(T·Q²·L) when P(q,l|q',l') = P(q|q')P(l|q) and segment likelihoods are precomputed. Murphy notes that Rabiner's tutorial overstates this cost. The DBN form of a variable-duration HMM costs O(T·D·Q²). — [Murphy segment.pdf](https://www.cs.ubc.ca/~murphyk/Papers/segment.pdf)
- **HSMM reference.** Yu (2010), *Artificial Intelligence* 174(2):215–243, reviews the explicit-duration, variable-transition and residential-time HSMM variants, along with duration distributions and observation models. It cites about 300 papers across about 30 application areas. — [bibbase entry](https://bibbase.org/network/publication/yu-hiddensemimarkovmodels-2010); [PDF copy](https://www.cs.ubc.ca/~murphyk/Teaching/CS540-Spring10/projects/Yu-hsmm09.pdf)
- **Semi-CRF.** Labels attach to whole segments, so a feature can describe a segment, for example its length or its mean score. A semi-CRF reduces to an ordinary CRF only when segment features split into sums of local features. Semi-CRFs generally beat CRFs on 5 NER tasks. — [Sarawagi & Cohen, NIPS 2004](https://proceedings.neurips.cc/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html)
- **Diarization practice.** Minimum turn duration is enforced by "linear chains of HMM states" for each speaker, not by the loop probability alone. The loop probability (P_loop) controls switching frequency and is tuned per dataset. BUT's DIHARD I system used P_loop 0.9, min duration 1 and downsampling 25. The DIHARD III system used P_loop 0.45 with downsampling 25. VB-HMM resegmentation "reliably improves DER and JER". — [Diez et al. Interspeech 2019](https://fit.vut.cz/research/group/speech/public/publi/2019/diez_is2019_192813.pdf); [BUT DIHARD I system description](https://dihardchallenge.github.io/dihard1/system_descriptions/but_systems.pdf); [DIHARD III paper](https://arxiv.org/pdf/2012.01477)
- **Post-processing comparison on 50Salads (ASRF, Table 3), using one action-segmentation backbone:**
  - No post-processing: F1@10 76.1, Edit 68.5.
  - Gaussian smoothing, K=5 / K=15: F1@10 80.7 / 80.8, Edit 73.5 / 73.6.
  - Relabelling short segments, θt=5 / θt=15 frames: F1@10 81.5 / 82.4, Edit 74.8 / 76.2.
  - Boundary-regression refinement (ASRF), which takes boundary peaks above 0.5 and assigns each segment its majority class: F1@10 84.9, Edit 79.3.

  The authors warn that short-segment relabelling is sensitive to its threshold, because large thresholds delete true short segments. — [ASRF, Ishikawa et al. WACV 2021](https://ar5iv.labs.arxiv.org/html/2007.06866)
- **Penalised change-point detection (ruptures PELT).** PELT is an exact optimal segmentation under a linear per-change penalty. Its average cost is O(C·K·n) under conditions on change-point placement. `min_size` sets the minimum distance between change points and `jump` subsamples the candidate grid. Cost models include l1, l2, rbf, normal, AR and custom. Usage is `rpt.Pelt(model="l2", min_size=..., jump=...).fit(signal).predict(pen=...)`. The docs give no guidance on choosing the penalty. — [ruptures PELT docs](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/); method from Killick, Fearnhead & Eckley, JASA 2012
- **Viterbi vs posterior decoding.** Viterbi can do poorly when several paths have similar probability. Pointwise posterior (MAP) decoding minimises expected per-frame errors but can produce transitions the model forbids. Hybrid risks mix the two. — [QATS paper, arXiv 2305.18578](https://arxiv.org/pdf/2305.18578)

### Inferences
- **Change-point detection plus labelling is equivalent to Viterbi.** With 3 labels and per-frame log-scores as the cost, "PELT with penalty β, then label each segment" amounts to Viterbi with a switch cost β and a uniform duration prior. A hand-written 3-state Viterbi is simpler and handles the labelling jointly, so ruptures adds little except a sanity baseline. PELT on raw scores also does not stop two adjacent segments from receiving the same label.
- **The HSMM's real advantage over an HMM is the minimum duration plus a non-geometric prior.** With only 3 states, the cheapest version is an expanded-state HMM: chain k_min states per label (for example 25 frames = 1 s for H/A), then add a geometric tail. Viterbi then costs O(T · 3·k_min · small) and needs no HSMM library.
- **An explicit HSMM Viterbi is also cheap.** With Q=3 and max duration D, the cost is O(T·Q²·D). At T=200k, Q=3 and D=1500 (60 s), that is about 2.7×10⁹ operations, which is slow in pure numpy. Two ways round this:
  - Decode at 5 fps (D=300), then refine boundaries at 25 fps.
  - Cap D for H/A at about 30 s and give D its own longer cap or a geometric tail. Durations beyond the cap can be modelled by letting a state repeat.
- **Median or majority filters are the weakest option.** Hysteresis (switch only when the margin exceeds δ for more than n frames) behaves like a crude minimum duration. It is fine as a baseline but cannot back-date a switch. An offline Viterbi or HSMM chooses the boundary position optimally.
- **The ASRF lesson transfers.** A separate boundary or touch-event detector that places cut points, followed by majority-vote labelling within each segment, beat smoothing by 2 to 4 F1 points. In our setting the "boundary branch" is the touch/event detector.

### Gaps
- I found no head-to-head study of HMM vs HSMM vs semi-CRF on segment-F1 for sports possession. The ranking above is inferred from action segmentation (ASRF) and diarization.
- I found no published comparison of hysteresis filters against median filters for over-segmentation.

## Q2. Lessons from temporal action segmentation (MS-TCN, ASFormer, F1@k, edit, smoothing, boundary refinement)

### Takeaway
F1@k uses essentially our metric, with one-to-one IoU matching, so the literature is directly relevant. Over-segmentation hurts segment F1 badly while barely moving frame accuracy. Multi-stage refinement, a truncated-MSE smoothing loss and boundary-based refinement give the biggest segment-F1 gains, often 5 to 15 points at near-constant accuracy.

### Cited Findings
- **The F1@τ definition is one-to-one.** Each predicted segment counts as a true positive if its IoU with a ground-truth segment of the same class exceeds τ. If several predictions overlap one ground-truth segment, only one counts as a true positive and the rest are false positives. The standard thresholds are τ ∈ {10, 25, 50}%. Edit score is 100·(1 − normalised Levenshtein distance) over label sequences. Frame accuracy (MoF) can stay high while segments are fragmented. — [Ding, Sener & Yao, TPAMI 2023 survey](https://arxiv.org/html/2210.10352v5)
- **MS-TCN smoothing loss.** It is a truncated MSE over frame-to-frame differences in log-probabilities: Δ = |log y_t − log y_{t−1}|, clipped at τ=4, weighted by λ=0.15, with gradients flowing only through y_t. On 50Salads, cross-entropy alone gave F1@{10,25,50} of 71.3/69.7/60.7, Edit 64.2, Acc 79.9. Adding the smoothing loss gave 76.3/74.0/64.5, Edit 67.9, Acc 80.7. — [MS-TCN, Abu Farha & Gall CVPR 2019](https://ar5iv.labs.arxiv.org/html/1903.01945)
- **Number of stages, 50Salads, F1@50 / Edit / Acc:**

  | Stages | F1@50 | Edit | Acc |
  |---|---|---|---|
  | 1 | 21.5 | 20.5 | 78.2 |
  | 2 | 47.3 | 47.9 | 79.8 |
  | 3 | 61.1 | 64.0 | 78.6 |
  | 4 | 64.5 | 67.9 | 80.7 |
  | 5 | 63.6 | 69.2 | 79.5 |

  Frame accuracy is nearly flat while segment F1 triples. The authors say the single-stage model "produces a lot of over-segmentation errors". — [MS-TCN](https://ar5iv.labs.arxiv.org/html/1903.01945)
- **Smoothing in a timestamp-supervision ablation (50Salads).** Adding smoothing and confidence losses raised F1@10 from 65.7 to 70.5 and Edit from 57.7 to 69.9, but F1@50 fell from 50.7 to 47.4. Smoothing mainly helps the low-IoU thresholds. — [Li et al., Timestamp supervision, arXiv 2103.06669](https://arxiv.org/pdf/2103.06669) (reported via search summary; not fully verified)
- **Boundary regression (ASRF) gains over MS-TCN, F1@10/25/50:**

  | Dataset | MS-TCN | ASRF | Gain |
  |---|---|---|---|
  | 50Salads | 76.3/74.0/64.5 | 84.9/83.5/77.3 | +12.8 F1@50 |
  | GTEA | 85.8/83.4/69.8 | 89.4/87.8/79.8 | +10.0 F1@50 |
  | Breakfast | 58.2/52.9/40.8 | 74.3/68.9/56.1 | +15.3 F1@50 |

  Frame accuracy rose only 1 to 4 points. — [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866)
- **Breakfast, F1@10/25/50, Edit, MoF:**

  | Method | F1@10/25/50 | Edit | MoF |
  |---|---|---|---|
  | MS-TCN | 52.6/48.1/37.9 | 61.7 | 66.3 |
  | MS-TCN++ | 64.1/58.6/45.9 | 65.6 | 67.6 |
  | ASFormer | 76.0/70.6/57.4 | 75.0 | 73.5 |
  | DiffAct+ASFormer | 80.3/75.9/64.6 | 78.4 | 76.4 |

  Other refinement methods named in the survey: BCN (pools confident boundaries), DPRN, UARL, SEDT (soft labels near boundaries), GTRM (GCN refinement) and Gaussian smoothing. — [Ding et al. survey](https://arxiv.org/html/2210.10352v5)
- **HMM length models in weak supervision.** These use Poisson length models and Viterbi in O(T·|N|²) (NN-Viterbi, HTK; Richard et al.). — [Ding et al. survey](https://arxiv.org/html/2210.10352v5)

### Inferences
- **spellF1 resembles F1@30–70 restricted to H/A runs.** The low thresholds (0.3) reward fixing over-segmentation, much as smoothing helps F1@10. The high thresholds (0.7) reward accurate boundaries, which is where boundary refinement helps F1@50. Both mechanisms are needed.
- **A flicker costs more than it looks.** One spurious 0.5 s A-run inside a 10 s H possession has the following effects:
  - It adds 1 false positive.
  - It splits the H run into two pieces, each with IoU of roughly 0.5 or less against the true run. That can turn 1 true positive into 0 true positives plus 2 false positives at thresholds of 0.5 and above.

  So a decoder should be biased against short H/A runs much more strongly than per-frame loss suggests.
- **Lateness tolerance.** A switch 1 s late on a 10 s possession costs about 10% IoU on each neighbour. The matched pair goes from IoU 1.0 to about 0.82, which still matches at all thresholds up to 0.7. Boundary precision of ±1 to 2 s is enough, and segment count accuracy matters far more.
- **D segments are not scored as spells.** They matter only through how they truncate H/A runs, plus the macro-F1 tie-break. Mislabelling a short D gap inside one team's spell as H/A merges two true spells into one, which costs 2 false negatives and 1 false positive. So D boundaries need their own care.

### Gaps
- I did not find the exact ASFormer smoothing settings or ablations, or any study of F1 at thresholds 0.3–0.7 specifically.

## Q3. Choosing duration distributions and switching penalties with little labelled data

### Takeaway
Fit duration priors from external event and tracking data such as IDSSE, using a gamma or negative-binomial family. Then tune only 2 to 4 scalars by grid search under leave-one-match-out cross-validation, directly on spellF1. Diarization practice confirms that the switch/loop parameter must be tuned per dataset and that its best value can be far from the "probabilistic" one.

### Cited Findings
- **IDSSE data.** Bassek et al. 2025, *Scientific Data* 12:195, cover 7 German matches from the 2022/23 Bundesliga and 2. Bundesliga. The data are synchronised TRACAB 25 Hz tracking (about 1.0M frames in total) plus official DFL event data (about 11.1k events), under CC-BY 4.0. They are available on Hugging Face (pysport/idsse-data) and loadable with kloppy. The authors note that 7 matches is a limited sample. — [Hugging Face idsse-data](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md); [DSHS Köln announcement](https://www.dshs-koeln.de/en/institut-fuer-trainingswissenschaft-und-sportinformatik/detail/meldung/official-game-data-in-the-journal-scientific-data)
- **Player possession times follow a gamma distribution.** This comes from J-League tracking data, 95 matches. Team possession is defined as a continuous spell of the same team keeping the ball. — [Theory and data analysis of ball possession time, arXiv 2308.05460](https://ar5iv.labs.arxiv.org/html/2308.05460)
- **Event-data team spells.** CIES/InStat 2017 reported that Celtic's possessions averaged 23.5 s, against 11.1 s for their opponents. — [CIES Football Observatory](https://www.cies.ch/cies/news/news-detail/article/ball-possession-celtic-heads-the-rankings)
- **Tracking-based control phases are much shorter.** One analysed match had 57:56 of net playing time and 1,291 in-possession phases totalling 29:48. The roughly 1.4 s average per phase is my own division of those totals. The definitions differ. — [Springer chapter](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/978-3-319-24560-7_13.pdf)
- **Diarization loop probabilities are tuned on dev sets.** Values range from 0.45 (DIHARD III) to 0.9 (DIHARD I) and are coupled with a posterior scale factor β (10 in DIHARD III). — [DIHARD III](https://arxiv.org/pdf/2012.01477); [BUT DIHARD I](https://dihardchallenge.github.io/dihard1/system_descriptions/but_systems.pdf)
- **HMM length-model parameters can be fit with constrained optimisation** (Poisson rates via COBYLA). — [Ding et al. survey](https://arxiv.org/html/2210.10352v5)

### Inferences
- **The definition of "possession" decides the duration prior.** Event-level spells run about 10–25 s, while per-touch control phases run about 1–2 s. The task's "maximal run of H/A" is the team-spell definition, consistent with the stated 5–15 s. Fit the H/A priors on IDSSE by deriving team spells from the event stream (team changes and out-of-play events), not from tracking "control" phases.
- **Recommended parameterisation, with few knobs:**

  | Knob | Value | Starting point |
  |---|---|---|
  | Minimum duration, H/A | d_min | 1–2 s (25–50 frames) |
  | Minimum duration, D | d_min | about 2–3 s |
  | Duration family | gamma or negative-binomial, shape/scale from IDSSE | no tuning needed |
  | Per-switch penalty | β (log-domain) | tuned |
  | Emission temperature | α, scaling the per-frame log-probabilities | tuned, analogous to VBx's β=10 |
  | Transition matrix | estimated from IDSSE | H→A direct, H→D, D→H/A; D→D impossible |

- **Tuning procedure.** Grid-search (β, α, d_min_HA) with leave-one-match-out CV, scoring spellF1 directly. Decoding is a fast deterministic dynamic program, so a grid of about 5×5×4 = 100 decodes per fold is cheap.
- **Use priors as regularisers.** A simple Bayesian step is to shrink the IDSSE-fitted gamma parameters toward the team's own labelled data, for example with a conjugate gamma–Poisson prior on the rate. With few labelled matches, though, the CV-tuned β dominates. Expect the best β to over-penalise switches relative to the generative value, because the metric punishes flicker more than per-frame likelihood does (see Q2 Inferences).
- **Calibrate first.** Without temperature scaling, emission scores at 25 fps are highly autocorrelated. Treating 25 frames per second as independent evidence makes the observation term about 25× overconfident. That is the reason for α, or for decoding at a lower frame rate. VBx's posterior scale and downsampling by 25 do the same job.

### Gaps
- I found no published IDSSE-derived team-spell duration distribution. It has to be computed.
- I found no peer-reviewed dead-ball (out-of-play) duration distribution with concrete percentiles in this pass.

## Q4. Constrained decoding with known structure (restarts, switches at touches, anchoring on confident events)

### Takeaway
All of these constraints can be added to the Viterbi or HSMM dynamic program as hard masks or soft costs on transitions and frames. The semi-CRF framing allows arbitrary segment-level features, such as "a segment must start at a touch".

### Cited Findings
- Semi-CRF segment features can encode arbitrary properties of a whole segment, for example its length or whether it starts at a known boundary. Inference remains an exact semi-Markov dynamic program. — [Sarawagi & Cohen 2004](https://proceedings.neurips.cc/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html)
- ASRF shows that a separate boundary detector which chooses cut points (peaks above 0.5), followed by majority labelling within segments, outperforms smoothing-only post-processing on every metric. — [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866)
- BCN uses confident boundary predictions to smooth low-confidence regions. — [Ding et al. survey](https://arxiv.org/html/2210.10352v5)

### Inferences (design recipe)
1. **Touch-gated switches.** Allow a direct H↔A transition at frame t only if t is within ±k frames of a detected touch or ball-contact candidate. Elsewhere, give that transition cost +∞, or a large soft penalty if the touch detector has misses. This one constraint kills most flicker.
2. **Dead-ball structure.**
   - Enforce H→D→(H|A).
   - Give D a minimum duration of about 2 s.
   - Make D→X depend on restart evidence. For example, the restart team is known from the side of the ball or the kicker's team, and that evidence enters as a strong emission bonus on X for the first n frames after D.
   - D→D self-transition is handled by duration.
3. **Anchoring.** For frames with very high-confidence labels (p>0.98 over a window, or a detected event such as a pass by a known team), clamp the label: set the other labels' emission log-probabilities to −∞. This splits the dynamic program into independent pieces and also prevents drift.
4. **Back-dating.** The dynamic program already places each boundary optimally given the emissions. Explicit back-dating to the first controlled touch is then a boundary refinement step:
   - For each decoded switch, search within [−2 s, +0.5 s] for the earliest touch by the new team.
   - Move the boundary there.

   This mirrors ASRF-style refinement and costs nothing for segment count.
5. **Per-frame fill for the tie-break.** After segment decoding, keep the decoded labels. Do not revert to argmax, which would reintroduce flicker that macro-F1 might reward but spellF1 penalises.

### Gaps
- I found no published football-possession work that uses touch-gated HSMM decoding. The recipe above is a design inference, not an evaluated method.

## Q5. Libraries: hmmlearn, pomegranate, pyhsmm, ruptures, edhsmm (determinism and speed for about 200k frames)

### Takeaway
None of these libraries supports custom touch-gated, time-varying transitions or hard anchors out of the box. A hand-written numpy/numba log-space Viterbi, or a segment dynamic program, is about 50 lines, deterministic, and likely the best choice. ruptures is fine for a change-point baseline. The HSMM libraries are unmaintained or sampling-based.

### Cited Findings
- **edhsmm** implements EM (following Yu 2010), Viterbi (following Benouareth et al. 2008), Gaussian and multinomial emissions, and multiple sequences. The repository was archived on 26 May 2024 and the author says correctness "is not guaranteed" and that the pip package was deleted. PyPI still lists 0.2.2, which is a conflicting signal. — [edhsmm PyPI](https://pypi.org/project/edhsmm/); [GitHub poypoyan/edhsmm](https://github.com/poypoyan/edhsmm)
- **pyhsmm** (Johnson & Willsky) does Bayesian nonparametric HMM/HSMM inference via Gibbs sampling, which makes it stochastic. It needs Cython, and its maintenance status is unclear. — [pyhsmm fork](https://github.com/takumayagi/pyhsmm); the sticky HDP-HMM diarization background is in [Fox et al. arXiv 0905.2592](https://arxiv.org/pdf/0905.2592)
- **ruptures PELT** provides exact penalised segmentation with `min_size` and `jump`. Speed depends heavily on the cost model; l2 is much faster than linear or AR. — [ruptures PELT docs](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/)
- **HSMM Viterbi** costs O(T·Q²·D). — [Murphy segment.pdf](https://www.cs.ubc.ca/~murphyk/Papers/segment.pdf)

### Inferences
- **A custom Viterbi with 3 states over 200k frames is fast.**
  - An expanded-state chain of 3×50 states costs about 200k×150×(few) operations, a fraction of a second with numba.
  - An explicit-duration dynamic program at 5 fps (T=40k, D=300 for 60 s, Q=3) costs about 1e8 operations, a few seconds with numba.
  - Both are fully deterministic if ties are broken by a fixed rule, such as the lowest index from argmax.
- **hmmlearn** `CategoricalHMM`/`GaussianHMM` Viterbi is deterministic at decode time. The randomness lies in EM initialisation, and it supports only a stationary transition matrix with geometric durations. We would supply our own emission log-likelihoods, which hmmlearn does not support cleanly. I did not verify this in this pass.
- **pomegranate.** v1.x was rewritten on PyTorch. It has dense and sparse HMMs, and I did not verify whether it has HSMM support in this pass.

### Gaps
- I did not fetch the hmmlearn and pomegranate docs. The claims about them above are not source-verified.
- I found no benchmark of these libraries at T=200k.

## Q6. Optimising segment-IoU F1 directly (decision-theoretic / MBR decoding)

### Takeaway
I found no published method that does MBR decoding for segment-IoU F1. The practical route has two parts. First, tune the decoder hyperparameters (β, α, d_min) by CV directly on spellF1, which is a black-box metric optimisation. Second, optionally rerank a small candidate set of decodes by their expected spellF1 under sampled posterior paths.

### Cited Findings
- **Viterbi optimises joint path probability, which is a 0/1 loss on the whole sequence.** Posterior (pointwise MAP) decoding optimises expected per-frame accuracy but can produce inadmissible paths. Hybrid risk functions interpolate between them. — [QATS, arXiv 2305.18578](https://arxiv.org/pdf/2305.18578)
- **Segmental MBR in ASR** (Goel, Kumar & Byrne, IEEE TSAP 2004) cuts lattices into segments to make MBR tractable. — [IBM Research listing](https://researcher.ibm.com/publications/segmental-minimum-bayes-risk-decoding-for-automatic-speech-recognition)
- **Pairwise MBR is O(k²).** It is quadratic in the number of candidates for non-linear metrics such as BLEU or F1. A linear-time approximation exists (DeNero, Chiang & Knight, ACL 2009). — [ACL 2009](https://aclanthology.org/www.mt-archive.info/ACL-2009-DeNero-2.pdf)

### Inferences
- **An MBR recipe for spellF1 that is feasible in a few days:**
  1. Run forward-backward on the (H)SMM.
  2. Draw N≈100 posterior paths with forward-filtering backward-sampling, using a fixed seed so the result stays deterministic.
  3. Build K≈20 candidate decodes: Viterbi under different (β, d_min) settings, plus a few samples.
  4. Pick the candidate that maximises the mean spellF1 against the samples.

  This costs O(K·N) metric evaluations, which is trivial. The approach only helps if the posterior is calibrated, so tune α first.
- **Why Viterbi undershoots on F1.** Because spellF1 strongly penalises extra segments, the MBR choice will tend toward fewer and longer segments than Viterbi. That is consistent with expecting CV to favour a higher β than the generative value.
- **Grid search directly on spellF1 is probably 80% of the gain.** It is a simple and deterministic black-box optimisation over decoder knobs. For the macro-F1 tie-break, among near-equal grid points prefer the one with higher frame macro-F1.

### Gaps
- I found no literature on expected-IoU-F1 or MBR decoding for temporal segmentation. Related work on "maximum expected F-measure" training and decoding (for example Jansche 2005/2007, or Nan et al. ICML 2012) was not fetched in this pass.
- I did not research the CROPS algorithm (Haynes et al. 2017), which finds all PELT segmentations over a range of penalties efficiently. It could be useful for sweeping β, but is unverified here.
