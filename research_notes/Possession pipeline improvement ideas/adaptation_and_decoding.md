# Domain adaptation / self-training and segment-F1 decoding for v12 possession labelling

Scope: (a) adapting a frame-level HistGradientBoosting (HGB) classifier to one target match with weak target labels; (b) decoding that directly raises spellF1 (IoU-matched possession-segment F1, IoU 0.3-0.7). Constraints: CPU, deterministic, ~1.5 days.

Note on evidence quality: there is little published work on exactly "tabular GBDT plus weak target labels plus segment-F1 decoding". Most cited evidence is from deep SSL, temporal action segmentation, HMM theory, speaker diarisation and Kaggle event detection. Recommendations that carry this over to v12 are in the Inferences sections and are labelled as such.

## Q1. Self-training / pseudo-labelling, importance weighting, adversarial validation

### Takeaway
Self-training helps only when the starting model is already mostly right on the pseudo-labelled region, when confidence is reliable, and when the student is regularised. If not, it reinforces its own mistakes ("confirmation bias"). For v12, the safest variant is one round of high-confidence, decoder-filtered pseudo-labels at low weight. Adversarial validation (a classifier that tries to tell source frames from target frames) is a cheap diagnostic for features that drift between matches. Importance weighting is riskier.

### Cited Findings
- Naive pseudo-labelling is limited by confirmation bias: the model learns to fit its own prediction errors. Arazo et al. reduce this with two regularisers: mixup, and a minimum number of truly labelled samples in every mini-batch. Mixup alone did not prevent confirmation bias when labelled data was scarce. — [Arazo et al., arXiv 1908.02983](https://arxiv.org/abs/1908.02983v5)
- Thresholding on confidence (keep pseudo-labels above a fixed confidence or a fixed top proportion) is the standard baseline. Wrong confidence values produce noisy pseudo-labels and confirmation bias. CAST lowers the confidence of pseudo-labels that fall in low-density regions of the labelled data. It is described as model-agnostic (gradient-boosted trees or neural nets) and was evaluated on up to 21 real tabular datasets. — [CAST, arXiv 2310.06380](https://arxiv.org/abs/2310.06380)
- Kumar, Ma and Liang show that self-training can fail without limit when the source-to-target shift is large, but works when the shift is gradual. They find regularisation and label sharpening are necessary even with unlimited data. Removing regularisation cut accuracy from 84% to 77% on Portraits and from 88% to 46% on rotating MNIST. Gradual self-training improved Portraits accuracy from 77% (direct adaptation) to 84%. — [Kumar et al., ICML 2020 / arXiv 2002.11361](https://arxiv.org/abs/2002.11361v1)
- Noisy Student: a teacher pseudo-labels unlabelled data, and a student of equal or larger size trains on labelled plus pseudo-labelled data with noise added (RandAugment, dropout, stochastic depth). The student then becomes the next teacher. Reported 88.4% ImageNet top-1, and ImageNet-A top-1 rose from 61.0% to 83.7%. — [Xie et al., arXiv 1911.04252](https://arxiv.org/pdf/1911.04252)
- Adversarial validation: train a classifier to separate train rows from test rows. AUC near 0.5 means little shift; AUC above about 0.8 means strong covariate shift (a change in the feature distribution). Its feature importances point to the drifting features. A suggested procedure drops the top drifting features one at a time, retrains, and keeps a feature removed only if the main score does not suffer. — [KDnuggets](https://www.kdnuggets.com/2018/06/how-dissimilar-train-test-data.html); [Hamel Husain, "Debugging AI With Adversarial Validation"](https://hamel.dev/blog/posts/drift/)
- The same classifier's output p gives a density-ratio weight p/(1-p) for each training row. The cinnamon library caps these weights (maximum ratio, default 10). Impurity-based feature importances are less reliable than alternatives such as SHAP for flagging drift. — [cinnamon docs](https://cinnamon.readthedocs.io/en/latest/generated/cinnamon.drift.adversarial_drift_explainer.AdversarialDriftExplainer.html); [apxml course](https://apxml.com/courses/monitoring-managing-ml-models-production/chapter-2-advanced-drift-detection/adversarial-validation-drift)
- A credit-scoring study uses adversarial validation to manage dataset shift. — [arXiv 2112.10078](https://arxiv.org/pdf/2112.10078)

### Inferences (recommendations for v12; not directly evidenced)
- **Idea A1: one round of decoded pseudo-labels (effort about 3-4 h, risk medium).**
  1. Run v12 end to end on the target match.
  2. Keep only frames that (i) lie inside a decoded H or A segment at least about 2 s long, (ii) are at least about 1 s (25 frames) from any decoded boundary, and (iii) have ensemble max-probability at least about 0.9. Also keep high-confidence dead frames.
  3. Retrain at low weight (about 1-2, compared with 5 for the existing weak labels). Then decode once.
  Do not iterate more than twice. The Kumar et al. result suggests errors compound when the shift is large. Using decoded segments, not raw frame argmax, works like label sharpening plus a structural filter. Dropping frames near boundaries keeps out the region where v12 is weakest (about ±1 s timing error).
  - Expected failure mode: the same as the weak TEAM-label attempt. Those labels were 94% accurate but did not help, which suggests labels that agree with what the model already predicts add little. Pseudo-labels from the model itself are even more redundant. The likely gain is mostly on frames where the base model is unsure, and those are exactly the frames the filter removes. Treat this as low expected value. Promote it only if it improves the stand-in bench mean and does not hurt any of the 3 Metrica matches.
- **Idea A2: adversarial validation for feature pruning (effort about 2 h, risk low as a diagnostic).**
  1. Label frames 0 = outside matches, 1 = target match. Fit a shallow HGB and use permutation importance (not impurity importance).
  2. If AUC is above about 0.8, look at the top 3-5 features. Likely candidates are absolute coordinates, pitch-size-dependent distances, frame-rate-dependent speeds, and tracking-noise statistics. Check whether they are scaling artefacts that per-match normalisation (Q2) would fix.
  3. Drop a feature only if spellF1 on the stand-in bench does not fall.
  With only 10 source matches, nearly every match-level feature will be separable (AUC close to 1). So use it to rank features, not as a pass/fail test.
- **Idea A3: covariate-shift importance weighting (effort about 2 h, risk medium-high).** Weight each source frame by clip(p/(1-p), 0.1, 10) from the adversarial classifier. With highly separable domains, the weights collapse onto a few frames, the effective sample size drops, and variance rises. Try it only if A2 shows moderate AUC (about 0.6-0.8). A safer variant is one weight per source match (the mean of p/(1-p) over its frames), which reduces to "upweight the matches most like the target".
- Determinism: fix `random_state` on every HGB. Pseudo-label filtering is deterministic given the model.

### Gaps
- No found benchmark of pseudo-labelling specifically for GBDT on time-series frames under match-to-match shift. CAST's exact gains per dataset were not retrieved; only the abstract was seen.
- Arazo et al.'s exact minimum count of labelled samples per batch was not retrieved.
- No authoritative source found for clipping thresholds for density-ratio weights beyond the library default of 10.

## Q2. Test-time adaptation via per-match feature standardisation

### Takeaway
The AdaBN idea is that domain-specific information sits in feature statistics, so you recompute those statistics on the target domain. For trees, the equivalent is to z-score or rank-normalise drift-prone features within each match, for both training and target. It is cheap, needs no labels, and is deterministic.

### Cited Findings
- AdaBN: the model weights hold label knowledge and the batch-norm statistics hold domain knowledge. Recomputing BN statistics on target data adapts the model with no extra parameters. A linear SVM could almost perfectly identify the source dataset from BN statistics. Gains were reported on Office-31 and Caltech-Bing. — [Li et al., arXiv 1603.04779](https://arxiv.org/pdf/1603.04779)
- In the CMI Sleep Kaggle 1st-place solution, a second-level model and per-day score normalisation were called "crucial", and post-processing alone did not improve the score. — [Kaggle solutions summary](https://kaggle-solutions-quartz.pages.dev/competitions/Child-Mind-Institute---Detect-Sleep-States)

### Inferences (recommendations for v12)
- **Idea B1: per-match normalised copies of drift-prone features (effort about 2-3 h, risk low-medium).**
  - Add per-match z-scored versions (or per-match percentile ranks) of features whose scale depends on the data source: speeds, accelerations, ball-to-player distances, tracking jitter or noise measures, counts of nearby players. Use median and IQR rather than mean and std, because degraded tracking has outliers.
  - Keep the raw versions too, or test raw-replaced versus raw-plus-normalised on the stand-in bench. HGB can choose between them.
  - This tackles the DFL-versus-Metrica gap directly: degraded DFL tracking versus Metrica sample tracking almost certainly differ in noise and scale.
  - Risk: features whose absolute value carries meaning, such as "ball speed > X m/s means a pass", lose that meaning when z-scored. Hence raw plus normalised.
- **Idea B2: per-match normalisation of model outputs (effort about 1 h, risk low).** Before decoding, rescale the ensemble's class priors to the target match's expected class balance (for example, the fraction of dead-ball frames suggested by the Smart Tagging tags). This is a label-shift correction, and it is close to the CMI per-day normalisation trick. Applying a per-class log-prior offset before Viterbi is one line of code.

### Gaps
- No direct published evidence for per-group standardisation with gradient-boosted trees in sports tracking. The AdaBN evidence is from CNNs.

## Q3. Decoding to improve segment-level IoU F1

### Takeaway
Segment metrics reward getting the number and extent of segments right, not per-frame accuracy. The strongest evidence (ASRF) is that explicitly detecting boundaries and then relabelling each segment by majority vote gives large F1@k gains: +9 to +16 points over MS-TCN. Hysteresis and minimum-duration rules (as in diarisation) and tuned decoding against the metric are the cheap, proven tools. Explicit-duration models (HSMMs) beat geometric durations in activity recognition. v12 already has a minimum-length chain, so the remaining gain is in modelling the duration shape and in boundary placement.

### Cited Findings
- ASRF (WACV 2021) adds a boundary-regression branch. A frame is a boundary if its boundary probability is a local maximum and above θp = 0.5 (results barely change for θp from 0.1 to 0.9: F1@10 about 84.4-84.9). The frames between boundaries form a segment, and the segment takes the majority-vote class of the frame classifier. On 50Salads versus MS-TCN: F1@10 76.3→84.9, F1@25 74.0→83.5, F1@50 64.5→77.3, Edit 67.9→79.3. The refinement step alone added +8.8 / +9.0 / +10.6 F1. A plain Gaussian smoothing post-process gained less than the boundary refinement. Segmental F1@k counts a predicted segment as correct if its IoU with a ground-truth segment exceeds k%. — [Ishikawa et al., arXiv 2007.06866](https://ar5iv.labs.arxiv.org/html/2007.06866)
- ASRF's loss ablation: class-weighted cross-entropy plus a Gaussian-similarity-weighted smoothing loss (GS-TMSE) gave F1@50 69.1→77.3 compared with the plain cross-entropy + smoothing (TMSE) setup. — [same](https://ar5iv.labs.arxiv.org/html/2007.06866)
- HMM state durations are restricted to a geometric shape. HSMMs give each state a duration distribution. In an activity-recognition study, the HSMM consistently beat the HMM. A 1990s comparison of HSMMs and expanded-state HMMs (chains of sub-states, like v12's) reported both improve duration modelling. — [IOS Press JAISE](https://content.iospress.com/articles/journal-of-ambient-intelligence-and-smart-environments/ais070); [Wang 1994, UvA](https://www.fon.hum.uva.nl/archive/1994/Proc18Wang.pdf)
- Viterbi maximises whole-path probability, posterior decoding maximises per-position accuracy, and a boundary-gain decoder can target boundary correctness. Lember and Koloydenko unify these as risk minimisation. Hybrid decoders beat both ends for blocks of intermediate size. — [JMLR v15 Lember & Koloydenko](https://www.jmlr.org/beta/papers/v15/lember14a.html); [arXiv 2504.15156](https://arxiv.org/pdf/2504.15156); [CPM 2010 slides](https://cpm.cs.helsinki.fi/cpm10/16.pdf)
- Speaker diarisation post-processing in pyannote uses hysteresis thresholds (onset ≠ offset) plus `min_duration_on` (remove short active regions) and `min_duration_off` (fill short gaps). These are tuned per dataset. For example, AMI VAD uses onset 0.684, offset 0.577, min_on 0.181 s, min_off 0.037 s; VoxConverse uses 0.767 / 0.713 / 0.182 / 0.501. — [pyannote/segmentation model card](https://huggingface.co/pyannote/segmentation)
- CMI Sleep 2023 (event detection with a timing-tolerance metric):
  - 1st place used decaying (sharpening) targets, a second model that predicts the exact second, greedy peak selection that maximises neighbourhood score sums without reusing a score, and a second-level model with per-day normalisation.
  - 2nd place used a LightGBM second stage on top of first-stage outputs, merged peaks within 100 steps, down-weighted unmatched peaks ×0.5, and added candidates at shifted offsets.
  - 3rd place smoothed with a centred rolling mean, used reversed-series augmentation (about +0.01 CV), and hand-set ensemble weights from local CV.
  — [Kaggle solutions summary](https://kaggle-solutions-quartz.pages.dev/competitions/Child-Mind-Institute---Detect-Sleep-States); [DeNA slides, 1st place](https://speakerdeck.com/unonao/shui-mian-konpe-1st-place-solution)
- Parkinson's Freezing-of-Gait Kaggle: the 1st place was "transformer and acceleration data" (Urazalinov), discussion 416026. The 8th place used a simple 1D-CNN. — [arXiv 2307.03475](https://arxiv.org/pdf/2307.03475)

### Inferences (recommendations for v12, ordered by expected value per hour)
- **Idea C1: tune the decoder directly on spellF1 (effort about 2-4 h, risk low).** Grid or coordinate search over temperature, switch cost, touch-gate strength, minimum possession length (for example 6/8/10/14 frames), minimum dead length, and a per-class log-prior offset. The objective is mean stand-in-bench spellF1. Decoding is cheap compared with training, so cache ensemble probabilities per match and sweep. This is the pyannote approach: hyperparameters tuned per dataset. Guard against overfitting 3 matches: prefer flat optima, and choose the centre of a plateau rather than the peak.
- **Idea C2: ASRF-style boundary model plus segment relabelling (effort about 4-6 h, risk medium).**
  - Train a second HGB to predict "possession change within ±k frames" (k about 5-12), from the same features plus lagged and lead emission probabilities.
  - Use its local maxima above a threshold to cut segments, then label each segment by mean ensemble probability.
  - Or, cheaper: feed boundary probability into the Viterbi switch cost, replacing or adding to touch gating: cost = -log(p_boundary).
  - This targets both remaining error types: missed or merged short possessions, and ±1 s timing. ASRF's +10 F1@50 is the strongest evidence in this area. Its robustness to θp suggests the method is not fragile.
- **Idea C3: explicit duration distributions in place of hard minimums (effort about 3-5 h, risk medium).** Fit possession and dead-ball duration histograms (log-normal or empirical) on the 10 source matches. Implement as a semi-Markov Viterbi over segment lengths up to L_max, or approximate with the existing sub-state chain by giving the self-loop on the last sub-state a learned probability and adding duration-dependent exit costs. Under a hard minimum, a 10-frame and a 200-frame possession are equally likely a priori, which is one reason short possessions get merged.
- **Idea C4: second-stage segment re-scoring (effort about 3-4 h, risk medium).** After decoding, compute features for each candidate segment: length, mean/min probability, touches inside, team of first and last touch, boundary probability at each end. Train an HGB on the source matches to predict whether the segment matches ground truth at IoU ≥ 0.5. Use it to (i) drop or merge low-scoring segments and (ii) choose between alternative decodings, for example the k-best paths or decodings at several temperatures. This mirrors the CMI 2nd place LGBM second stage.
- **Idea C5: boundary shift by touch snapping (effort about 1-2 h, risk low).** Possession changes happen at touches. After decoding, move each H↔A boundary to the nearest detected touch of the incoming team within ±25 frames, if one exists. This is a cheap, rule-based fix for timing error. It matters most at IoU 0.6-0.7 for short segments, where a 1 s offset on a 3 s possession costs about a third of the IoU.
- **Idea C6: hysteresis / merge rule (effort under 1 h, risk low).** If a gap between two same-team segments is shorter than X frames and is dead or opponent with low confidence, merge it. Drop or absorb segments shorter than Y frames whose mean probability is under a threshold. This is the pyannote `min_duration_off/on` approach and is easy to tune in C1.
- A posterior-decoding or MBR hybrid is theoretically appealing, but spellF1 is not a per-frame loss, so true MBR would need sampling paths and scoring expected spellF1. That is feasible (sample N=200 paths with forward-filtering backward-sampling, using a fixed seed for determinism, then pick the sample with the highest average spellF1 against the other samples). It is probably too much work for 1.5 days compared with C1, C2 and C5.

### Gaps
- No found head-to-head of expanded-state HMM (sub-state chains) versus HSMM on segmentation F1. The 1994 paper's numbers were not retrieved.
- The exact parameters of the CMI 1st-place post-processing were not retrieved (the repository README has no details; the Kaggle discussion 459715 was not fetched).
- The Freezing-of-Gait 1st-place details were not retrieved.

## Q4. Ensembling: probability vs log-odds averaging, diversity, member count

### Takeaway
Averaging logits (equivalently, a geometric mean of probabilities) tends to give sharper, better-calibrated ensembles than averaging probabilities, especially when members are underconfident. Theory says powers between 0 (geometric) and 1 (arithmetic) give consistent gains. Because v12 feeds an HMM with temperature 0.1, the averaging choice and the temperature interact, so they must be tuned together.

### Cited Findings
- Averaging logits improved calibration of underconfident models (MC-dropout on CIFAR-10: ECE 12.03% → 5.44%). Probability averaging separated true from false predictions better. — [Njieutcheu Tassi et al., DLR](https://elib.dlr.de/188833)
- Logit averaging gave the highest accuracy at every ensemble size tested on CREMA-D. Uncalibrated probability averaging was worst. — [arXiv 2607.05019](https://arxiv.org/pdf/2607.05019)
- Calibrated-logit ensembles beat uncalibrated logit and probability averaging on most datasets under distribution shift. — [arXiv 2207.08977](https://arxiv.org/pdf/2207.08977)
- Linear pooling (probability mean) and geometric pooling (log-prob mean) are endpoints of a generalised-mean family. Consistent log-likelihood gains over individual members hold for powers in [0, 1]. — [Razafindralambo et al., UAI 2026 / arXiv 2603.04204](https://arxiv.org/abs/2603.04204)
- CMI Sleep 4th place blended two pipelines 0.25/0.75 with weighted box fusion. 3rd place set ensemble weights by hand from local CV. — [Kaggle solutions summary](https://kaggle-solutions-quartz.pages.dev/competitions/Child-Mind-Institute---Detect-Sleep-States)

### Inferences (recommendations for v12)
- **Idea D1 (effort under 1 h, risk low):** switch to averaging log-probabilities, renormalise, and retune the temperature jointly in the C1 sweep. With a Viterbi decoder, log-average equals summing member emissions, which is consistent with the HMM. Expect a small change. It is mainly a calibration fix.
- **Idea D2: diversity over count (effort about 2-4 h CPU, risk low).** Five members built from frame slices plus feature subsampling are fairly correlated, because they all see the same source matches. Cheap diversity options:
  - members trained with a different held-out source match (leave-one-match-out bagging), which gives match-level diversity aimed at domain shift;
  - members with different `max_leaf_nodes` / `learning_rate`;
  - one member on per-match-normalised features (B1).
  Diminishing returns after about 5-10 members is a common rule of thumb, not sourced here. Going from 5 to 10 identical-recipe members will likely add little.
- Report the stand-in bench per-match spread, not just the mean. With 3 Metrica targets, a change of about 0.01 spellF1 is likely within noise.

### Gaps
- No authoritative source retrieved on optimal ensemble size for GBDT members. The 5-10 figure is a heuristic.

## Suggested 1.5-day plan (inference, ranked)
1. Cache ensemble probabilities per stand-in match. Do D1 (log-averaging) and C1 (decoder sweep including C5 touch-snapping and C6 merge rules). About 0.5 day. Low risk, highest certainty of gain.
2. C2: boundary HGB feeding Viterbi switch costs. About 0.5 day. Best evidence-backed structural gain (ASRF).
3. B1: per-match normalised features, plus A2 adversarial validation as a diagnostic to pick which features. About 0.25 day.
4. Only if time remains: C4 segment re-scoring or C3 duration distributions. Skip A1 and A3 unless A2 shows moderate rather than extreme shift; the evidence and v12's own TEAM-label result suggest low payoff.
