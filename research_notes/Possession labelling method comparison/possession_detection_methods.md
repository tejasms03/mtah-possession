# Possession labelling from football tracking data: methods and reported accuracy (H / A / D per frame)

Scope note: 14 tool calls covered rule-based possession (Link & Hoernig, Vidal-Codina, databallpy), learned frame-level game-state models (Borghesi et al. TCN), ball inference from players (Amirli & Alemdar, Kim et al. "ballradar", Capellera et al. TranSPORTmer), the IDSSE dataset and IMM ball tracking. The most relevant paper is **Borghesi et al. 2023 (ESWA)**. It solves the same 3-class task (dead / home / away per frame) from tracking data and benchmarks rule-based methods against a TCN on the same data.

## 1. Rule-based possession and ball-carrier algorithms: definitions and reported accuracy

### Takeaway
Rule-based possession takes the nearest player within a radius (0.5 to 1.5 m), confirmed by a spike in ball acceleration, speed or direction. Against manual labels it reaches about F=0.83 for possession *changes* on clean TRACAB data. On a head-to-head frame-level benchmark with real broadcast-quality data, team-possession accuracy is only **64.5% to 79.1%**. Dead/alive rules are much worse, at about 56%. These methods all depend on a mostly present ball, which is not true here (ball missing on 41% of frames).

### Cited Findings
- **Link & Hoernig 2017 (PLOS ONE), "Individual ball possession in soccer".** Uses 25 Hz TRACAB data with x,y only (no z). A player gains possession when they are nearest and within a trained distance threshold T_P. The kick-detection variant also requires a local ball-acceleration maximum of at least 4 m/s². An action ends when the predicted ball position stays beyond T_A for 1 s. A Bayesian network then classifies "ball control". Threshold values are not given in the text. — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953)
- Link & Hoernig validation: a trained observer labelled 5 matches (6,976 action phases) frame by frame. A detection counted as correct if it fell within 0.6 s and named the right player. Reported F = .88 for action detection and **F = .83 for possession detection**. The ball-control classifier scored 96.7% overall with κ = .67; two human observers agreed at κ = .72. The method was then applied to 60 Bundesliga matches (69,667 intervals). Ball tracking losses caused many errors. — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953)
- Earlier conference version (Hoernig, Link, Herrmann, Radig, Lames 2016): precision and recall of 80 to 92% on clean raw data, about 4 percentage points lower on uncorrected data. — [TUM portal](https://portal.fis.tum.de/en/publications/detection-of-individual-ball-possession-in-soccer/) (summary via search snippet; full text not read)
- **Vidal-Codina et al. 2022 (Sports Engineering 25:18; arXiv 2202.00804).**
  - Possession zone: radius R_pz = 0.5 m (Track160) or 1 m (TRACAB, Hawk-Eye). Duel zone: 1 m around the ball, and two or more opponents inside means a duel.
  - A gain needs a change in ball direction or speed. A loss needs the ball to leave the zone with displacement above a threshold.
  - Ball noise is smoothed with a Savitzky-Golay filter (order 2, 7 frames ≈ 0.25 s).
  - **Possession itself was not benchmarked**, only the downstream events: passes reached >90% precision and recall; shots 78% precision and 53% recall. Data: 29+ matches at 25 Hz from 3 providers.
  - Dead ball came from a boolean flag (TRACAB) or from missing ball data (other providers). Missing ball data is listed as a main limitation.
  - [arXiv HTML](https://arxiv.org/html/2202.00804v3)
- **databallpy (open-source Python) implementation.**
  - Implements Vidal-Codina as `add_individual_player_possession()`. Defaults: pz_radius = 1.5 m, bv_threshold = 5 m/s, ba_threshold = 10°, min_frames_pz = 0. The docs' own example call uses bv_threshold = 1.5.
  - Team possession (`add_team_possession`) comes from **event data**: the last successful pass, dribble or shot.
  - The docs example agrees with the provider's team_possession on alive frames for 5,298 of 5,892 frames, about 90% (my calculation from the documented counts). No formal accuracy is reported, and the docs say results depend heavily on ball-tracking quality.
  - [databallpy docs](https://databallpy.readthedocs.io/en/latest/features/team_and_player_possession.html); [GitHub](https://github.com/alek050/databallpy)
- **Head-to-head benchmark (Borghesi et al. 2023, Table 3)** on the same professional-league tracking data:
  - Team-possession accuracy: Link & Hoernig 2017 **64.5%**, Khaustov & Mozgovoy 2020 **75.4%**, Morra et al. 2020 **79.1%**, TCN **86.2%**.
  - Dead/alive accuracy: Wei et al. 2013 decision tree **56.0%**, TCN **89.2%**.
  - Khaustov & Mozgovoy assign the ball to the closest player when ball speed or direction changes.
  - [Borghesi et al. accepted version, PoliTo IRIS](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)

### Inferences
- Rules are fast to reproduce: databallpy and the Vidal-Codina logic run in about a day of work. They are a necessary baseline, but expect roughly 65 to 80% team-possession accuracy on noisy data, lower still with 41% of the ball missing, because every rule keys on ball kinematics.
- Rules give little help on dead-ball detection. The D class needs a separate model or heuristic, for example the ball-missing pattern, players stopping, or positions at restarts.
- Rules can serve as engineered *features* (nearest-player team, distance margin) for a learned model, not only as the final labeller.

### Gaps
- No public paper found that reports floodlight's or kloppy's possession accuracy. kloppy mostly passes through the provider's ball_owning_team; I did not verify floodlight's implementation.
- Numeric thresholds for Vidal-Codina (ϵs, ϵθ, ϵv) are in Online Resource 1, which I did not read.
- I did not find the DFL/Sportec official possession definition document. IDSSE says only that possession is "defined by the player in control of the ball."

## 2. Learned models for possession and ball carrier: data needed and gains over rules

### Takeaway
The closest analogue to this task is Borghesi et al. 2023. It is a permutation-invariant TCN over 64-frame acausal windows, trained on 100K samples from 4 matches. It reaches 86.2% possession accuracy (+7.1 points over the best rule) and 89.2% dead/alive accuracy (+33 points), but global 3-class accuracy is only about 81.6 to 84.4%. Architecture choice moved accuracy only ±1.7 points. Inputs and data mattered more than the network type.

### Cited Findings
- **Borghesi, Costa, Morra, Lamberti 2023, "Using Temporal Convolutional Networks to estimate ball possession in soccer games" (Expert Systems with Applications 223:119780).**
  - Task: per-frame {dead, home, away}, which is exactly H/A/D. — [PoliTo IRIS](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
  - Data: 35 matches from a top European league (2019-20), about 16 fps. Only the **8 games with the least missing ball** were used, split 4/2/2. Training used 100K random samples (sliding window, stride 1); validation and test used 5K each. Matches came from different stadiums and teams. — same source
  - Input: window nf = **64 frames**, target frame at position 48 (acausal, so it uses about 16 future frames). 36 objects (ball, referee, 34 player slots). 11 channels per object: x, y, vx, vy, role one-hots, home/away flags, out-of-pitch flag and missing flag. Ball gaps are interpolated with an Akima spline before training. — same source
  - Architecture: a shared per-object TCN builds embeddings, then permutation-invariant aggregation (sum, self-attention or 2D conv). The best variant is two-branch: dead/alive plus possession, with the possession loss masked on dead frames. — same source
  - Design alternatives: mean accuracy across all variants was 82.93% ± 1.71%; the best global accuracy was 84.44% (self-attention + sum). — same source
  - Ablation (Table 4, global / dead-alive / possession):
    - Baseline: 81.59 / 88.25 / 84.95%
    - Ball + 5 nearest players: 79.41 / 85.8 / **84.98%**, so possession needs only players near the ball.
    - Ball only: 58.35 / 83.62 / 50.54%
    - x, y + roles: 67.25 / 74.89 / 82.09%
    - x, y only: 56.16 / 75.81 / 51.89%
    - Dead/alive benefits from all players (+2.5 points). — same source
  - The authors argue for acausal windows: after a foul, players stop or move toward the referee, which is visible only in later frames. — same source
- **Kim et al. 2023 (KDD), "Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM" ("ballradar").**
  - Data: 15 K League matches (Fitogether GPS, 10/2/3 split, about 200K training frames) plus 3 Metrica matches. 10 Hz, 10 s windows.
  - Player-only inference (100% of the ball masked):
    - H-LSTM: player possession accuracy (PPA) 64.0%, **team possession accuracy (TPA) 85.3%**, ball position error 3.69 m.
    - H-Transformer: PPA 49 to 52%, TPA 78 to 79%, error 4.9 m. **Bi-LSTM beat the Transformer.**
  - Repo: github.com/pientist/ballradar. The paper links it only for corrected Metrica data.
  - [arXiv HTML](https://arxiv.org/html/2306.08206)
- **Capellera et al. 2024, TranSPORTmer (ACCV).**
  - Data: 283 LaLiga 2022-23 matches, about 83K training sequences, 9.6 s windows at 6.25 Hz.
  - Ball inference from players: average error **2.71 m** vs ballradar 3.89 m.
  - Ball-state classification (pass / possession / uncontrolled / out-of-play) from player movement only: **80.84%**.
  - No team-possession accuracy reported; code availability not stated.
  - [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2); [CVF](https://openaccess.thecvf.com/content/ACCV2024/papers/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.pdf)

### Inferences
- **A neural network is not clearly the best choice for this project.** The published NN gain over the best rules is about 7 points on possession. It needed 100K labelled samples from 4 full matches with professional per-frame labels, and the team has 5 minutes of ground truth (about 7,500 frames). The gain is attainable only if IDSSE (7 matches with per-frame BallPossession and BallStatus) is used for training and the model transfers to the target's noisier, broadcast-derived data. Domain shift is the main risk.
- The ablations suggest that hand-built features cover most of the signal: ball (or estimated ball) plus the 5 nearest players with team flags and velocities, over a roughly 2.5 to 4 s acausal window. Gradient boosting on windowed features (nearest-player team, distance margins, team centroid velocities, ball-missing run length, rolling statistics with past and future offsets) is a credible, cheaper competitor. This is my inference; no paper benchmarks GBM on this exact task.
- The permutation-invariant set or attention design matters because there are no persistent IDs. A per-frame set encoding (sorted by distance to the ball) solves this for GBM as well.

### Gaps
- No published GBM or XGBoost baseline for frame-level team possession found.
- No GNN paper with team-possession accuracy found. One Lund student thesis uses graph attention to locate the ball from player detections, but I did not read it: [LUP](https://lup.lub.lu.se/student-papers/search/publication/9172881).
- No public Borghesi code found.
- Everett, SportsTrack and Sportlogiq possession work: no accessible sources found in this pass.

## 3. Inferring the ball, and team possession, from players only

### Takeaway
With the ball fully hidden, team possession can be recovered from player movement alone at about **85% (Kim et al. 2023, TPA)**. Ball position error is about 2.7 to 3.7 m, with state classification about 81% (TranSPORTmer). Even sparse ball observations help a lot: with 20% of the ball visible, TPA reaches 96.7%. This project has about 59% of the ball visible.

### Cited Findings
- Kim et al. 2023, Table 3 (H-LSTM-RL-PP trained with random ball masking), team possession accuracy by share of the ball masked:
  - 100% masked: TPA 81.06%, PPA 57.3%, error 5.14 m
  - 95% masked: TPA 90.73%
  - 90% masked: TPA 94.50%
  - **80% masked: TPA 96.67%, PPA 91.25%, error 1.41 m**
  - [arXiv HTML](https://arxiv.org/html/2306.08206)
- TranSPORTmer imputation (80% or 90% masked): ballradar error 0.88 / 1.18 m; TranSPORTmer about 0.8 m at 80%, but that table was hard to parse, so verify before citing. — [arXiv 2410.17785](https://arxiv.org/html/2410.17785v2)
- Amirli & Alemdar 2022 (APJESS): an NN maps player positions in a single frame to ball x,y. Trained on 300 Turkish Super League matches. Mean absolute error 7.56 m (x) and 5.01 m (y); R² 0.79 (x) and 0.92 (y). This is the single-frame baseline. — [DergiPark](https://dergipark.org.tr/en/pub/apjess/article/1060725)
- Borghesi et al. interpolate missing ball coordinates with an Akima spline as preprocessing and feed a "missing" flag channel. — [PoliTo IRIS](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)

### Inferences
- **Temporal context vs single frame.** Single-frame ball regression from players gives about 5 to 7.5 m error (Amirli). Windowed Bi-LSTM gives 3.7 m and set-transformer windows give 2.7 m. That is roughly a 2x error reduction from temporal context, though the datasets differ, so this is indicative only.
- Partial ball observations move accuracy into the mid-90s for possession in Kim's setting, though their masking is random and well spread. The project's 41% missing is likely clustered (long gaps during aerial balls and broadcast cuts), which is harder than random masking. Evaluate per gap length.
- A practical pipeline would be:
  1. Interpolate or smooth short ball gaps (spline or Kalman/RTS).
  2. For long gaps, infer the ball or possession from player configuration with a model trained on IDSSE using artificially masked ball segments.
  3. Apply temporal smoothing with an HMM or Viterbi pass over H/A/D, using a minimum-duration prior.

### Gaps
- No paper reports possession accuracy as a function of *gap length* (clustered missingness).
- No HMM or Viterbi-smoothing result for possession found. Possession runs have strong persistence, so HMM smoothing is plausible, but unbenchmarked.
- Ballradar's code reusability (environment, data format) is not verified.

## 4. IMM and multi-model Kalman filters for ball tracking

### Takeaway
I found no paper that applies an IMM Kalman filter to football ball tracking with reported accuracy. The closest is phase-based ball modelling (rolling / flying / in possession / out of play) in Ren et al. 2008. Generic robotics work shows multiple-model filters beat single-model ones for manoeuvring targets. Link & Hoernig used an RTS smoother on ball segments.

### Cited Findings
- Ren, Xu, Orwell & Jones 2008 model the ball in four phases (rolling, flying, in possession, out of play) with a separate model per phase. Unreliable or missing observations are recovered with spatio-temporal constraints and temporal filtering. — [Strathprints](https://strathprints.strath.ac.uk/29273/)
- Kwok & Fox: multiple-model Kalman filters (IMM, GPB) beat a single-model filter for manoeuvring targets, and kick type determines the subsequent ball motion. This is robotics, not football tracking data. — [core.ac.uk PDF](https://core.ac.uk/download/9031182.pdf)
- Birbach & Frese: multiple-hypothesis tracking with a UKF per track for flying balls with missing or false detections. — [DLR elib](https://elib.dlr.de/74466/1/Birbach2011.pdf)
- Link & Hoernig 2017 drop ball segments shorter than 1 s and smooth the rest with an RTS smoother. — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953)

### Inferences
- With 2D (x,y) broadcast-derived data, an IMM with constant-velocity and decelerating "rolling" modes plus a high-noise "kick" mode is easy to build. filterpy has an IMMEstimator. It can bridge short gaps and flag kicks, which are useful as possession-change features.
- An IMM cannot place the ball during long gaps without player context. Expect it to help mainly on short gaps under 1 s. This is my inference and is unbenchmarked.

### Gaps
- No sports-specific IMM accuracy numbers found.

## 5. Public training data: IDSSE (Bassek et al. 2025)

### Takeaway
IDSSE provides about 1.0M frames from 7 DFL matches (2022/23), with ball fields BallPossession (1 = home, 2 = away) and BallStatus (0/1). That is about 130x more labelled frames than the team's 5 minutes. The labels have no published accuracy, and the source data is clean TRACAB (ball present), so masking must be simulated to mimic the target data.

### Cited Findings
- Bassek, Rein, Weber, Memmert 2025, Scientific Data, doi 10.1038/s41597-025-04505-y. 7 matches (2 Bundesliga, 5 2. Bundesliga, 2022/23), 1,002,644 position frames. TRACAB cameras at 25 Hz (16 to 20 cameras). Ball records include BallPossession (1 = home, 2 = away; "defined by the player in control of the ball") and BallStatus (0 = inactive, 1 = active). The method for generating possession (manual or algorithmic) is not described, and there is no validation of possession or ball tracking. CC-BY 4.0. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/); [HF mirror](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- kloppy can load IDSSE (Sportec format). The Hugging Face mirror is maintained by the pysport community. — [HF mirror](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)

### Inferences
- Recommended protocol: train on IDSSE with simulated degradations (drop the ball on about 41% of frames with *clustered* gaps, add position noise, drop player IDs, and drop players to about 20 visible). Use the team's 5 minutes **only** for validation and calibration, or a light fine-tune of thresholds or HMM transition probabilities.
- Model order for 8 days:
  1. Rule baseline (databallpy / Vidal-Codina logic plus a dead-ball heuristic).
  2. GBM on windowed set features plus HMM/Viterbi smoothing.
  3. Optionally a small Bi-GRU or TCN over distance-sorted players.

  Pick by 5-minute validation accuracy, not by prior. The literature does not support "NN is clear-cut best" at this data scale. It does support a temporal (windowed, acausal) model over a single-frame one, and dead-ball detection being the main gain over rules.

### Gaps
- No information on how IDSSE's possession labels were generated or how well they agree with the target match's labelling convention. Check, for example, how DFL labels contested balls or aerial passes.
