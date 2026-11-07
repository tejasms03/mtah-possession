# Weak supervision, self-training, domain adaptation and model choice for frame-level H/A/D possession labelling

Setting recap (from assignment): 1 target match, 25 fps, ~159k graded frames, ball missing 41%, no player IDs, 5 min / 17 possessions of ground truth for checking only, no hand labelling. Public IDSSE data (7 matches, clean TRACAB tracking with per-frame BallPossession/BallStatus) to be degraded to mimic target noise. Plan: deterministic labeller -> model (LightGBM or temporal NN) on confident labels + degraded IDSSE -> HSMM decoding. 8 days, 5 students, laptops.

## Q1. Data programming / Snorkel-style label models (incl. sequential): does a label model beat one hand-written rule set, and how many LFs are needed?

### Takeaway
A learned label model only pays off over majority vote in a "mid-density" regime of many partially-overlapping, independent-ish labelling functions (Snorkel's real tasks used 11-33 LFs and gained 0.1-7 points of label accuracy). With a handful of correlated hand-written rules (closest-player, ball-speed, out-of-bounds) the gain will be near zero; the bigger, better-documented win is the *discriminative end model* trained on the weak labels (+0.5 to +5.5 points over training on unweighted LF votes in Snorkel), and for sequences, models that encode temporal dependence (Dugong, FlyingSquid).

### Cited Findings
- Snorkel (Ratner et al., VLDB 2018) came within an average 3.60% of models trained on large hand-curated sets, and improved 132% on average over prior heuristic approaches; in a user study, experts built models 2.8x faster than 7 h of hand labelling with +45.5% average predictive performance. — [Ratner et al., arXiv 1711.10160](https://arxiv.org/abs/1711.10160)
- Number of LFs used in Snorkel's evaluation tasks: Chem 16, EHR 24, CDR 33, Spouses 11, Radiology 18, Crowd 102 (crowd workers as LFs). — [Ratner et al., Table 2](https://arxiv.org/abs/1711.10160)
- "Modeling advantage" of the generative label model over unweighted majority vote (label accuracy): Radiology 7.0%, CDR 4.9%, Spouses 4.4%, EHR 2.8%, Chem 0.1% (optimizer chose plain MV for Chem). Label densities were only 1.2-2.3 non-abstain votes per point. — [Ratner et al., Table 1](https://arxiv.org/abs/1711.10160)
- Theory/synthetic result: in the low-density regime (few votes per point) there is little room for weighting to differ from majority vote; in the high-density regime majority vote converges to optimal; the label model mainly helps in the *mid-density* regime. Synthetic setup: 1000 points, LFs of 75% average accuracy and 10% voting probability; advantage peaks at ~0.1-0.2 with tens-to-hundreds of LFs. The authors explicitly say modelling accuracies "will not hurt" but majority vote often works just as well. — [Ratner et al., Sec. 3.1 / Fig. 4](https://arxiv.org/abs/1711.10160)
- Modeling advantage on CDR grows as the number of LFs increases (Appendix A.5), i.e. little advantage in early development with few LFs. — [Ratner et al., App. A.5](https://arxiv.org/abs/1711.10160)
- Discriminative end model vs generative label model alone (F1): Chem 33.8 -> 54.1, EHR 74.9 -> 81.4, CDR 38.5 -> 45.3, Spouses 57.4 -> 54.2 (a drop). On CDR/Spouses the end model matched hand supervision within ~1 F1. — [Ratner et al., Table 3](https://arxiv.org/abs/1711.10160)
- Discriminative model trained on Snorkel's probabilistic labels vs trained on unweighted LF average: +5.5 (Chem), +0.5 (EHR), +3.3 (CDR), +1.4 (Spouses), +3.1 acc (Crowd), +5.0 AUC (Radiology); +5.81% average. — [Ratner et al., Table 5](https://arxiv.org/abs/1711.10160)
- Sequential weak supervision: Dugong (Varma/Sala et al., NeurIPS 2019) models multi-resolution sources over sequences (sources that label single frames or groups of frames, with temporal correlations) and reports +16.0 F1 over traditional supervision and +24.2 F1 over existing weak-supervision approaches on video and sensor classification tasks; +36.8 F1 on a clinician-validated biomedical video task. No public code found. — [NeurIPS 2019 abstract](https://proceedings.neurips.cc/paper/2019/hash/93db85ed909c13838ff95ccfa94cebd9-Abstract.html); [Snorkel page](https://snorkel.ai/research-paper/multi-resolution-weak-supervision-for-sequential-data/)
- FlyingSquid (Fu et al., ICML 2020) learns LF accuracies in closed form via triplet methods, supports temporal dependencies between neighbouring frames (video), is up to 170x faster on simple structures and up to 4,000x faster on video tasks with temporal dependencies; evaluated incl. a tennis-rally video task; released as a PyTorch layer by HazyResearch. — [arXiv 2002.11955](https://arxiv.org/abs/2002.11955); [Hazy Research blog](https://hazyresearch.stanford.edu/blog/2020-02-28-flyingsquid); [PMLR](https://proceedings.mlr.press/v119/fu20a.html)
- WRENCH benchmark: 22 real datasets (classification + sequence tagging), >100 method variants, uses majority vote as the baseline for label models. — [WRENCH, NeurIPS 2021 D&B](https://neurips.cc/virtual/2021/47104)
- "Stronger Than You Think" (BoxWRENCH, 2025): on harder, more realistic tasks, supervised learning needs 1000+ labelled examples to match weak supervision in many settings. — [arXiv 2501.07727](https://arxiv.org/html/2501.07727v1)

### Inferences
- For this project, a Snorkel/FlyingSquid label model over 3-6 correlated rules is unlikely to beat a well-tuned single rule cascade by a measurable amount on 17 possessions of ground truth; it is not where the 8 days should go. If used at all, FlyingSquid's temporal (frame-neighbour) model is the right flavour, and it is cheap to try (closed form, pip-installable).
- The analogue of an "LF" that would actually add independent signal here is a *different model family*: e.g. an IDSSE-trained classifier is a labelling source whose errors are less correlated with the hand rules than another distance rule would be. Combining "rules" + "IDSSE model" as two sources with an abstain option is a cheap two-source label model (essentially a weighted vote).
- The Snorkel evidence that the end model beats the label model (coverage + generalisation) supports the planned "rules -> model" step, but note one counterexample (Spouses dropped 3.2 F1).

### Gaps
- No published numbers found for weak supervision specifically on sports-tracking time series, or for label-model vs single-rule-cascade comparisons with fewer than ~10 LFs on real data.
- WRENCH's label-model-vs-MV numbers were not retrieved.
- Dugong has no code located; FlyingSquid's temporal-model accuracy numbers on video tasks were not extracted.

## Q2. Self-training / pseudo-labelling risks; when does a model trained on rule outputs beat the rules?

### Takeaway
A student beats its weak teacher only when (a) it cannot cheaply fit the teacher's mistakes (pseudolabel correction) and (b) it generalises from confidently-labelled frames to unlabelled/uncertain ones (coverage expansion). Plain repeated self-training on own predictions drifts via confirmation bias; mitigations are noise/augmentation on the student, label noise/soft labels, keeping a floor of trusted labels in every batch, and evaluating on held-out time blocks.

### Cited Findings
- Lang, Sontag & Vijayaraghavan (NeurIPS 2024) formalise two effects of training a strong student on weak labels: *pseudolabel correction* (student fixes teacher errors, happens when "the strong model is unable to fit the mistakes of the weak teacher without incurring additional error") and *coverage expansion* (student performs well on inputs the teacher abstained on, even though those inputs were excluded from training). Bounds rely on expansion properties that are checkable from finite data, with empirical evidence they hold. — [arXiv 2405.16043](https://arxiv.org/abs/2405.16043)
- Wei, Shen, Chen & Ma (ICLR 2021): under an expansion assumption, self-training with input-consistency regularisation provably denoises incorrect pseudolabels using correctly-labelled neighbours. — [arXiv 2010.03622](https://arxiv.org/abs/2010.03622v5)
- Arazo et al. (IJCNN 2020): naive pseudo-labelling overfits its own incorrect guesses ("confirmation bias"); label noise and mixup regularise this, but mixup alone is insufficient when labelled samples are few — they add a *minimum number of labelled samples per mini-batch*. — [arXiv 1908.02983](https://arxiv.org/abs/1908.02983v5)
- Noisy Student (Xie et al., CVPR 2020): self-training works when the student is equal-or-larger than the teacher and is *noised* (RandAugment, dropout, stochastic depth) while the teacher labels clean inputs; iterated student->teacher. 88.4% ImageNet top-1, ImageNet-A 16.6% -> 74.2% (v1 numbers 87.4%). — [CVPR 2020 paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.pdf); [arXiv 1911.04252](https://arxiv.org/abs/1911.04252v1)
- Snorkel end models generalised beyond LFs (higher recall; e.g. Chem recall 21.6 -> 39.2) — consistent with coverage expansion. — [Ratner et al., Table 3](https://arxiv.org/abs/1711.10160)

### Inferences (practical scheme for this project)
- "Confident frames" from the deterministic labeller are a biased sample (easy frames: close control, clear dead balls). A model trained only on them will be confident on easy frames and arbitrary on the hard ones (ball missing, contested, long passes) — exactly the 41% that matters. Mixing in degraded IDSSE frames (which have true labels on *hard* frames too) is the main defence; this is the "minimum trusted labels per batch" idea from Arazo et al. translated to this setting.
- Feature-level independence from the teacher matters: if the model sees the same features the rules threshold on (nearest-player distance etc.), it will just re-learn the rules (no pseudolabel correction). Give the model features the rules don't use (team-shape/centroid, velocity fields, pressing structure, windowed context) and/or train with dropout of the rule-derived features.
- One round of self-training is probably the max worth doing in 8 days; iterate only if each round improves the 17-possession check AND an IDSSE held-out match, never on the target's own pseudo-labels alone.
- Holdout: use *time-block* holdout (whole halves / 5-10 min blocks) and *leave-one-match-out* for IDSSE; random-frame splits leak heavily at 25 fps because adjacent frames are near-identical (Borghesi et al. split by match for this reason, see Q4).
- With only 17 ground-truth possessions, any difference of < ~1 possession-boundary error is noise; treat the 5-min GT as a sanity check, and use IDSSE leave-one-match-out (with degradation) as the main model-selection metric.

### Gaps
- No source found quantifying confidence-threshold choice for pseudo-labels in time-series/tracking tasks; thresholds must be tuned on degraded IDSSE.
- No source found on co-training specifically for sports tracking.

## Q3. Sim-to-real / synthetic degradation and domain adaptation

### Takeaway
Training on clean data with injected missingness/noise is supported for sensor time series (sensor dropout, temporal dropout), and a sports-specific precedent shows a tracking-only model can recover ball-in-play status without the ball (92% frame-level on unseen matches). Domain-adversarial methods (DANN etc.) are competitive but hinge on model selection without target labels, which this project cannot do reliably with 5 minutes of GT — so matching degradation statistics explicitly is the safer bet.

### Cited Findings
- IDSSE: Bassek, Rein, Weber & Memmert, Scientific Data 12:195 (Feb 2025); 7 matches 2022/23 (2 Bundesliga, 5 2. Bundesliga), TRACAB Gen5 optical tracking at 25 fps, integrated with official event data; CC-BY 4.0; ~2.4 GB; loaders in floodlight and kloppy. — [PMC article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11787359/); [HF mirror](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md); [floodlight docs](https://floodlight.readthedocs.io/en/latest/modules/io/datasets.html)
- Lang, Wild, Isenko & Link (Sci. Reports 2022): ball in-play status predicted from *player tracking only* (no ball) with up to 92% frame-level accuracy on previously unseen matches; 102 manually annotated Bundesliga matches; models: logistic regression, decision tree, random forest, AdaBoost (best); 81% precision for stoppages > 2 s. — [DOAJ record](https://doaj.org/article/4899fef7e03b4b3aaf5e0d0f1ccb0e3a); [PDF](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- Input Sensor Dropout (masking a whole sensor's features in training) increased robustness to missing sensors on three multi-sensor EO time-series datasets; Temporal Dropout (randomly dropping time steps) made models robust to missing time steps. — [Mena et al., arXiv 2407.15512](https://arxiv.org/html/2407.15512)
- Input dropout is *not* equivalent to data augmentation targeted at missing-input robustness (DropIn study on RNNs) — i.e. simulate realistic missingness patterns, don't just use random dropout. — [DropIn journal paper](https://arpi.unipi.it/bitstream/11568/1000157/2/dropinjournal.pdf)
- AdaTime benchmark (Ragab, Eldele et al., ACM TKDD 2023): 11 DA methods, 5 datasets, 50+ cross-domain scenarios; with careful hyperparameter selection, generic visual DA methods (incl. DANN) are competitive with time-series-specific DA; model-selection strategy (how you pick hyperparameters without target labels) significantly changes results. — [arXiv 2203.08321](https://arxiv.org/abs/2203.08321); [code](https://www.github.com/emadeldeen24/AdaTime)
- Borghesi et al. (2023) used missing flags + Akima-spline interpolation, with missing values set to -2 and an explicit "is missing" channel per object; they selected the 8 games with *least* missing ball data for training/testing, i.e. their 86% possession accuracy is on relatively clean data. — [Borghesi et al., ESWA 223:119780](https://iris.polito.it/handle/11583/2976470)

### Inferences
- Degradation should be *pattern-matched*, not iid: measure on the target match the distribution of ball-gap lengths, where gaps occur (aerial balls, crowded areas, near goal), position jitter, and absence of IDs; then resample gaps in IDSSE from that empirical gap-length distribution conditioned on context (e.g. ball height/speed, number of players within 2 m). Also strip player IDs/jersey numbers and keep only team affiliation, to match the target.
- Validate the degradation by a "domain classifier" check: train a quick LightGBM to distinguish degraded-IDSSE windows from target windows on the model's features; AUC near 0.5 = good match; high AUC with top features listed tells you what to fix. (This is the cheap, non-adversarial version of DANN's idea.)
- Skip DANN: per AdaTime, results depend on model selection, which needs target labels the team doesn't have in quantity; implementation + tuning is not a good use of 8 days.
- Lang et al. supports building the D (dead-ball) head from player-only features (team centroids, speeds, formation spread), which is useful exactly when the ball is missing.

### Gaps
- No published study found of synthetic ball-dropout degradation of clean football tracking to transfer to a noisier provider; the effectiveness must be measured on the 17 GT possessions + held-out degraded IDSSE.
- Domain randomisation literature from robotics (Tobin et al. 2017) not retrieved here.
- Per-method DANN numbers from AdaTime not retrieved.

## Q4. LightGBM vs temporal NNs (1D CNN/TCN/GRU/transformer) vs MiniRocket on windowed time-series classification with small/medium data

### Takeaway
On hand-engineered window features at this data size, gradient-boosted trees are the safe default (Grinsztajn et al.). A temporal NN has published evidence on *this exact task* (TCN, 86.2% possession / 89.2% dead-alive vs rule-based baselines +7.3 / +33.2 points), but that used ~100k human-labelled frames from clean-ish data. MiniRocket is a cheap, almost-deterministic, strong alternative for raw multichannel windows (all 109 UCR datasets in < 10 min), but it is a fixed-length-window classifier that needs a fixed object ordering, which the lack of player IDs complicates.

### Cited Findings
- Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022 D&B): across 45 tabular datasets with hyperparameter search, tree-based models (XGBoost, RF) remain state-of-the-art on medium-sized data (~10K samples) even before counting their speed advantage; NNs struggle with uninformative features, rotation-invariance, and irregular functions. — [arXiv 2207.08815](https://arxiv.org/abs/2207.08815)
- Borghesi, Costa, Morra & Lamberti (ESWA 2023) — TCN on tracking for exactly dead/home/away: 86.2% possession accuracy (+7.3 vs best rule-based, Morra et al. 2020) and 89.2% dead-alive (+33.2 vs state of the art). Rule baselines reimplemented: Link & Hoernig 2017 (ball accel threshold 4 m/s^2, player-ball distance 1.5 m), Morra et al. 2020 (1.09 m, 1.19 m, 8.6 m/s thresholds), Khaustov & Mozgovoy 2020. — [Borghesi et al.](https://iris.polito.it/handle/11583/2976470)
  - Data: 8 matches at 16 fps, labels from a live human operator; ~40% dead, 29.6% home, 30.3% away; train 100K windows from 4 matches, val 5K from 2, test 5K from 2 (match-disjoint, different stadiums/teams); sliding window stride 1. — [same](https://iris.polito.it/handle/11583/2976470)
  - All architecture variants landed within 82.93% +/- 1.71% global accuracy — architecture choice mattered little. — [same](https://iris.polito.it/handle/11583/2976470)
  - Ablations: ball-only input -> 58% global (possession ~chance), but dead-alive 83.6% from ball alone vs 88.3% full; ball + 5 nearest players -> possession 84.98% vs 84.95% full (no loss); roles/team info crucial for possession; velocity + out-of-pitch features crucial for dead-alive (coordinates-only drops acc_DA to 75.8%). — [same](https://iris.polito.it/handle/11583/2976470)
- MiniRocket (Dempster, Schmidt & Webb, KDD 2021): up to 75x faster than Rocket on larger datasets; train+test on all 109 UCR datasets to near-SOTA in < 10 minutes; marginally more accurate than Rocket, slightly less than the most accurate methods; almost deterministic, optionally fully deterministic. — [arXiv 2012.08791](https://arxiv.org/abs/2012.08791); [code](https://GitHub.com/angus924/minirocket)
- Bake Off Redux (Middlehurst, Schäfer & Bagnall, DMKD 2024): on 112 UCR + 30 new datasets, MultiROCKET+Hydra and HIVE-COTEv2 are significantly better than other approaches. — [arXiv 2304.13029](https://arxiv.org/abs/2304.13029v3)
- Lang et al. 2022 got 92% frame-level ball-status with classic tabular learners (AdaBoost best) on engineered player features. — [DOAJ](https://doaj.org/article/4899fef7e03b4b3aaf5e0d0f1ccb0e3a)

### Inferences
- Borghesi's TCN advantage was measured against hand rules, not against a GBM on good window features; Lang et al. shows tabular ensembles do well on related football frame tasks. So "NN > rules" is evidenced, "NN > LightGBM on engineered window features" is not.
- The NN's documented strength (learning from raw multi-object trajectories) assumes abundant clean labels; here labels come from ~7 IDSSE matches (~7 x 90 x 60 x 25 ≈ 945k frames, highly autocorrelated — effective sample size much smaller) plus rule pseudo-labels. Comparable to Borghesi's 100K windows from 4 matches, so an NN is *feasible* on IDSSE, but domain shift, not capacity, is the bottleneck.
- Recommended: LightGBM as the primary model on multi-scale window features (e.g. 0.2 s, 1 s, 3 s windows; nearest-k players per team sorted by distance to ball — Borghesi shows 5 nearest suffice — which also solves the no-ID ordering problem). Then the HSMM gives the temporal smoothing a TCN would have learned. An NN is "worth it" only if one student can time-box a small 1D-CNN/TCN (Borghesi-style input: ball + 5 nearest per team, missing flags) to ~2 days and it beats LightGBM on leave-one-IDSSE-match-out *and* on the 17 GT possessions; otherwise drop it.
- MiniRocket: viable as a 1-day experiment (sktime/aeon implementation) on fixed windows of sorted-nearest-player channels + ball channels with missing-flag channels, followed by a ridge/logistic classifier; its outputs can be stacked into LightGBM as features. Caveat: 159k frames x stride-1 windows is a lot of transforms on a laptop — subsample training windows (e.g. stride 5-10).

### Gaps
- No head-to-head benchmark found of GBM-on-window-features vs TCN for football possession.
- InceptionTime's rank in Bake Off Redux not retrieved.

## Q5. Reproducibility / determinism: NN on CPU vs LightGBM deterministic mode

### Takeaway
LightGBM has a documented `deterministic=true` mode (CPU) that, with `force_row_wise` or `force_col_wise` fixed, constant `num_threads` and fixed seeds, gives stable results on the same machine/version; PyTorch can be made deterministic with seeds + `torch.use_deterministic_algorithms(True)` but explicitly does not guarantee cross-platform/cross-version reproducibility. Neither is bitwise-reproducible across students' different laptops/OSes — pin versions and generate final labels on one machine.

### Cited Findings
- LightGBM `deterministic` (default false, CPU only): gives stable results with same data and parameters; may slow training; results can still differ with other seeds, LightGBM versions, compilers or systems; must set one of `force_col_wise` / `force_row_wise` to true to avoid numerical instability; don't change `num_threads` during training; per-feature seeds (bagging_seed=3, feature_fraction_seed=2, etc.) have fixed defaults, `seed` overrides them. GPU uses 32-bit float sums by default (`gpu_use_dp=true` for 64-bit). — [LightGBM Parameters docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- PyTorch: `torch.use_deterministic_algorithms(True)` uses deterministic implementations where available and throws for ops with no deterministic alternative; constant seeding yields the same random stream "in the same environment"; deterministic ops are often slower; "results are not guaranteed to be reproducible across PyTorch releases, individual commits, or different platforms." Also seed DataLoader workers. — [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- MiniRocket is "almost deterministic" with an optional fully deterministic mode. — [arXiv 2012.08791](https://arxiv.org/abs/2012.08791)

### Inferences
- Practical recipe: `lightgbm.LGBMClassifier(deterministic=True, force_row_wise=True, seed=42, num_threads=4)`, pinned `lightgbm==X.Y.Z` in requirements; for PyTorch: `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(fixed)`, `num_workers=0`. Commit the final label file plus a hash, and check the run-to-run label agreement (should be 100% on one machine; report the % across machines).
- NN training adds variance across seeds; report mean +/- sd over 3 seeds on IDSSE holdout before claiming it beats LightGBM.

### Gaps
- No source quantified CPU-only PyTorch nondeterminism for conv/GRU ops (most documented nondeterminism is CUDA-specific).

---

## Consolidated recommendations (inferences, for the report writer)

1. **Main model: LightGBM** on multi-scale window features, trained on (a) degraded IDSSE (all frames, true labels, leave-one-match-out CV) + (b) the target match's high-confidence rule-labelled frames, with sample weights so IDSSE hard frames are always present (Arazo-style "minimum trusted labels"). Deterministic mode on.
2. **Degradation fidelity is the biggest lever**: match the target's ball-gap length distribution and context, jitter, and drop IDs; verify with a domain-classifier AUC check.
3. **Separate D from H/A** (Borghesi two-branch result; Lang et al. player-only D): a dead-ball model using ball + team-level kinematics, then H/A conditional on alive. HSMM then enforces minimum durations.
4. **Label model**: optional, low priority. If tried, use FlyingSquid (temporal) with the rules and the IDSSE model as sources; expect ~0-few point gain.
5. **NN**: time-box to ~2 days for one student (small TCN on ball + 5 nearest per team with missing flags, Borghesi-style). Keep only if it beats LightGBM on leave-one-IDSSE-match-out averaged over 3 seeds and does not hurt the 17-possession check. MiniRocket as a 1-day alternative/stacking feature.
6. **Self-training**: at most one round, student noised (feature dropout, gap augmentation), never evaluated on its own pseudo-labels; use time-block holdouts.
