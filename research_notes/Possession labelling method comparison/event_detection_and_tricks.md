# Event detection from football tracking data and domain tricks for H/A/D possession labelling

Scope: industry and academic practice for detecting possession, passes, ball-out-of-play and restarts from tracking; how well team shape alone predicts possession; ball inference during gaps; hybrid "clever" ideas; competitions. Written for a team labelling each 25 fps frame H/A/D from broadcast-derived tracking (ball missing 41%, no persistent IDs), 5 min of GT, 8 days left.

Coverage note: about 15 tool calls. Three primary papers were read in full or near full (Vidal-Codina et al. 2022; Borghesi et al. 2023; PathCRF 2026). Others are from abstracts or search summaries and are marked as such.

## 1. Rule-based event and possession detection from tracking (Vidal-Codina, Link & Hoernig, Morra, etc.)

### Takeaway
The standard approach in industry and academia is rule-based: a possession zone (ball within about 0.5 to 1 m of a player), a duel zone, ball speed and direction changes to find touches, and set-piece templates while the ball is dead. On clean in-stadium tracking these rules get over 90% precision and recall for passes and most restarts. The catch is that every one of these methods assumes the ball is tracked. The best rule system re-implemented on a real team-possession task scored only 79% possession accuracy.

### Cited Findings
- Vidal-Codina, Evans, El Fakir & Billingham (Sports Engineering 2022, arXiv 2202.00804) build a decision-tree framework with two steps. The first is a possession model; the second detects events from possession gains and losses (passes, shots, crosses, receptions, interceptions, set pieces). — [arXiv 2202.00804](https://arxiv.org/abs/2202.00804v3)
- Their possession rule: a player is in possession if the ball is inside their circular "possession zone" of radius R_pz. A "duel" is declared if at least two opponents are within R_dz ≥ R_pz of the ball, and duel overrides possession. Noisy ball x,y is smoothed with a Savitzky–Golay filter (order 2, 7-frame window, which is about ±0.25 s at 25 Hz). — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- Chosen radii: R_pz = 0.5 m for Tracab and 1 m for the other two providers, with R_dz = 1 m for all. The authors say the method can be applied to lower-quality data such as broadcast or single-camera tracking "by augmenting the possession zone radius and tuning the hyperparameters". They recommend cross-validating the hyperparameters for each provider. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- Data: Tracab, Track160 and Hawk-Eye, all at 25 Hz, from the FIFA Club World Cup 2019/2020 and 12 Bundesliga 2019-20 games. Dead ball is either a boolean (Tracab) or inferred from missing ball data (Track160, Hawk-Eye). Benchmarked against Sportec Solutions (STS) official event data. Runtime is about 3 minutes per match on 1 CPU. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- Results for open play (over 30K events), read from Fig. 7a: passes have recall about 94% and precision about 90%. Shots have recall 53% and precision 78%; 70% of the 254 missed shots were blocked shots labelled as passes. Crosses have recall 39% and precision 86%. Errors cluster where players occlude each other. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- Results for set pieces, read from Fig. 7b in the order kickoff / penalty / corner / throw-in / goal kick / free kick: recall is about 100/100/97/92/96/89% and precision about 100/100/97/95/95/82%. Free kicks are worst because no template fits them; they are the fallback when no other configuration matches. Goals: 6 scorer errors, half of them because the ball went missing after the assist. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- The authors list their own limitations: "ball tracking data needs to be available", and they depend on an in-play/dead flag that is "critical for set piece detection". Suggested future work includes broadcast tracking (where "the pitch is not always visible"), referee-whistle audio for dead-ball intervals, and team-possession extraction. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- Link & Hoernig (PLOS ONE 2017), "Individual ball possession in soccer", used 60 Bundesliga 2013/14 matches. Ball-to-player distance, direction, speed and acceleration feed an ML model of time spent in a player's "sphere of influence". F-scores were 0.88 for individual ball action and 0.83 for individual ball possession, with κ = 0.67 for individual ball control. Team possession averaged about 56 minutes per match. — [PMC5503225](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225/)
- As re-described by Borghesi et al.: Link & Hoernig detect touches as spikes in ball acceleration above 4 m/s², then assign the ball to the nearest player. Using acceleration avoids false possession changes when the ball flies over intermediate players, since z is usually missing. Morra et al. (2020) use player–ball distance, player movement and ball speed, with thresholds of 1.09 m, 1.19 m and 8.6 m/s. Khaustov & Mozgovoy (2020) detect changes from ball speed, ball direction and prolonged proximity, then assign to the nearest player. — [Borghesi et al. 2023, accepted version](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- Head-to-head on team possession (alive frames only, 20K samples from 2 held-out games): Link & Hoernig 64.5%, Khaustov & Mozgovoy 75.4%, Morra et al. 79.1%, Borghesi TCN 86.2%. On dead-vs-alive, a decision tree on ball coordinates only (Wei et al. 2013) got 56.0% against the TCN's 89.2%. — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- DFL/IDSSE definitions (Bassek, Rein, Weber & Memmert, Scientific Data 2025): BallStatus is 0 (inactive) or 1 (active), and the match is "inactive when the match is interrupted by the referee". BallPossession is 1 (home) or 2 (away), "defined by the player in control of the ball". The paper does not say how possession is coded during contests or dead ball, or whether it is entered manually. Tracking is TRACAB Gen5 with 16–20 cameras at 25 Hz, and position RMSE against VICON is 0.06–0.18 m. Seven matches, in XML, loadable with `floodlight` (v0.4.0). Event-to-tracking sync used DataBallPy. — [Bassek et al. 2025, PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/)
- SkillCorner (broadcast tracking vendor) open data: 10 A-League 2024/25 matches at 10 fps. Every frame has a `possession` dict {player_id, group: home/away} and players carry an `is_detected` flag (detected vs extrapolated off-camera). Phases of play exist only while the ball is in play, and each in-possession phase pairs with an out-of-possession phase. The page does not say how possession is determined. — [SkillCorner Open Data](https://skillcorner.github.io/opendata/)

### Inferences
- The project's GT (organiser-corrected dead flag plus owning team) works like a manual three-button timer. Borghesi et al. say explicitly that their labels were made this way ("three-button timer used by a manual operator"), so their task is close to identical to this one. Expect the GT to lag events slightly and to give the ball to a team through short loose-ball spells. Rule systems that output "no possession / duel" have to be mapped to H/A, for example by carrying forward the last owner.
- Vidal-Codina-style rules are reproducible in 1–2 days: possession zone, duel zone, Savitzky–Golay smoothing, set-piece templates. With 41% of ball frames missing, use them only as anchors on frames where the ball is visible, not as the whole labeller.
- The set-piece templates (all players in own half for a kickoff, a player beyond the sideline for a throw-in, a player at the corner mark, a player in the goal area for a goal kick) need only player positions. That makes them usable when the ball is missing, and each template also identifies the restarting team (see section 5).
- SkillCorner open data is broadcast-derived like this project's data, with player extrapolation and a per-frame team possession label. It could be extra training or validation data for a team-possession model on broadcast-quality input, although it has 10 fps and no explicit dead flag (phases exist only in play). This was not verified hands-on.

### Gaps
- Morra et al. 2020 and the ETH Zurich work were found only as described by Borghesi et al.; the primary sources were not read.
- No published reports were found for Sportlogiq, Opta/Stats Perform, Second Spectrum or Metrica Play describing their tracking-to-possession algorithms. Vendor methods appear to be proprietary.
- No primary source was found for the DFL's official possession definition document, beyond the one-line definition in the IDSSE paper.
- DeepMind "Game Plan" (2021) and Google Research Football were not researched. Neither appears to target possession labelling from real tracking.

## 2. Dead-ball / out-of-play detection from tracking and restart detection

### Takeaway
On a comparable task, a learned model distinguishes dead from alive at about 89% frame accuracy, against about 56% for a ball-only decision tree. Ball trajectory carries about 94% of that signal, and velocities plus "outside pitch" flags are the most useful features. Once the dead interval is known, restart type can be classified from player configuration with over 90% precision and recall, except for free kicks.

### Cited Findings
- Borghesi, Costa, Morra & Lamberti (Expert Systems with Applications 223, 2023), "Using Temporal Convolutional Networks to estimate ball possession in soccer games". Classes are dead / home / away, the same scheme as this project. Labels come from a human operator in real time. Class shares are about 40% dead, 29.6% home and 30.3% away. — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- Data: 35 games from a top European league in 2019-20 at about 16 fps, with 8 games chosen for the fewest missing ball coordinates. Training used 100K windows from 4 games, validation 5K windows from 2 games and test 5K windows from 2 games. — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- Inputs: a window of 64 frames (about 4 s) with the target at frame 48, so the model uses future frames. There are 36 objects (ball, referee, 34 player slots). Each object has 11 channels: x, y, vx, vy, three role flags (ball / referee / GK), two team flags, an out-of-pitch flag and a missing flag. The ball is Akima-spline interpolated and missing values are set to −2. The model applies a per-object TCN embedding, then a permutation-invariant combination (sum, self-attention or 2D conv). — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- Results: the best two-branch model (dead/alive, then home/away given alive) reaches 86.39% global accuracy. Dead-alive accuracy is 89.2% and possession accuracy 86.2%. Across architectures the mean is 82.93% ± 1.71%, so the architecture choice matters little. — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- Ablations (Table 4, accuracy as global / dead-alive / possession):
  - Baseline: 81.59 / 88.25 / 84.95
  - Ball plus 5 nearest players: 79.41 / 85.8 / 84.98
  - Ball only: 58.35 / 83.62 / 50.54
  - x, y and roles only: 67.25 / 74.89 / 82.09
  - x, y only: 56.16 / 75.81 / 51.89

  The authors conclude that ball trajectory accounts for about 94% of the dead-alive signal, while possession needs team and role assignment. Removing velocities and the outside-pitch flag hurts dead-alive accuracy a lot. — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- Vidal-Codina et al. treat each dead interval [d0, dc] by checking which set-piece triggers stay active up to the last dead frame dc. Triggers follow IFAB Laws 8 (kickoff), 14 (penalty), 15 (throw-in), 16 (goal kick) and 17 (corner). The event just before a dead interval is the "dead ball event" (out for throw-in, out for corner, goal, and so on). Missing ball data counts as a dead-ball trigger for providers without a flag. They suggest synchronised referee-whistle audio to improve dead-ball interval boundaries. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- PathCRF models the ball going out as an absorbing self-loop on one of 4 "outside" nodes (the boundary lines). Training uses only in-play episodes, and stoppages are not modelled as states. — [PathCRF, arXiv 2602.12080 (HTML)](https://arxiv.org/html/2602.12080)

### Inferences
- For D frames, useful hand features (all supported by the ablations) are:
  - ball outside the lines, or last seen heading out at speed, followed by a gap
  - a collapse in mean player speed; dead periods show many players walking, which is not cited but matches the importance of velocities in the ablations
  - set-piece templates: a player beyond the sideline, a player at a corner arc, all players in own halves, a cluster in the goal area
  - the Smart Tagging set-piece intervals the project already has

  A gradient-boosted or 1D-CNN classifier on these window features is a direct analogue of Borghesi's dead-alive branch.
- When the ball is missing because play is off-camera (broadcast), "missing" is not the same as "dead". Unlike Track160 or Hawk-Eye, where missing means dead, this signal will be noisy here. Treat missingness as a feature, not a rule.
- Borghesi used about 16 fps and in-stadium cameras; the project's broadcast data is noisier, so expect lower absolute numbers.

### Gaps
- No paper was found that reports dead-ball accuracy specifically from broadcast-derived tracking.
- No numeric results were found for detecting dead ball purely from collapses in player speed. This is a plausible heuristic, but unbenchmarked in the sources read.
- Wei et al. 2013 (the ball-only dead-alive decision tree) was not read directly.

## 3. Team possession from team shape alone (no ball)

### Takeaway
Player positions without the ball can carry most of the possession signal, provided team identity and temporal context are used. PathCRF reaches 92.4% frame-level team-possession accuracy on the same IDSSE matches with no ball input at all. Static, single-frame shape features are much weaker (about 60–82%).

### Cited Findings
- PathCRF (Kim, Lee, Seo, Ko, Yoon & Park; KDD 2026, arXiv 2602.12080):
  - Method: ball-free event detection. Each time step selects one directed edge among 22 players plus 4 outside nodes (676 edges). A self-loop means control; u→v means the ball is in transit. A CRF with masked illegal transitions is decoded with Viterbi.
  - Data: the 7 IDSSE / Sportec Open DFL matches (5 train, 1 validation, 1 test), downsampled to 5 fps.
  - Results on the test match: edge accuracy 69.64%, sender accuracy 80.44%, receiver accuracy 82.28%, event F1 75.69% (77.21% at 25 fps, but about 12 times the training time). Frame-level team possession accuracy is 92.40%; home share is 62.13% in GT against 62.64% predicted.
  - Baseline: regressing the ball from players and then applying rules ("Ball TP") gets only 48.39% event F1.
  - Code: https://github.com/hyunsungkim-ds/pathcrf

  — [PathCRF HTML](https://arxiv.org/html/2602.12080); [abstract](https://arxiv.org/abs/2602.12080)
- In PathCRF, Viterbi or constrained decoding alone, without a learned CRF, already improves event F1 from 65.34% (per-step argmax) to 73.14%. Removing illegal transitions accounts for most of the gain. — [PathCRF HTML](https://arxiv.org/html/2602.12080)
- Kim et al. (KDD 2023, arXiv 2306.08206), "Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM". It predicts the ball possessor first, then the ball trajectory, from player trajectories only, with a "reality loss" and post-processing for physical plausibility. A search summary reports about 3.69 m position error for the Bi-LSTM variant against about 4.91 m for the Transformer variant; this was not verified in the full text. — [arXiv 2306.08206](https://arxiv.org/abs/2306.08206); numbers via [search summary / Liner](https://liner.com/review/ball-trajectory-inference-from-multiagent-sports-contexts-using-set-transformer)
- An optimal-transport embedding paper (arXiv 2501.10299) classifies which team has possession from player locations only. Per a search summary, raw tracking gets about 59.7% or 76.8% (depending on classifier), the OT embedding about 81.7–82.3%, and average positions about 59%. Frames without possession are dropped. The table header was not verified. — [arXiv 2501.10299](https://arxiv.org/pdf/2501.10299)
- Borghesi et al.: in their ablation, x, y and roles (with team flags) without velocities reach 82.09% possession accuracy, while x, y without team or role information get 51.89%, which is chance. — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- In basketball, Wei et al. (ICCV-W 2015) predict the ball owner from player trajectories alone by aligning players to a role template. — [Wei et al. 2015](https://openaccess.thecvf.com/content_iccv_2015_workshops/w21/papers/Wei_Predicting_Ball_Ownership_ICCV_2015_paper.pdf)

### Inferences
- PathCRF's 92.4% team-possession accuracy comes from the same IDSSE data the project may use externally, but from in-stadium tracking with persistent IDs. The project has no persistent IDs, so a per-player-edge model does not transfer directly. A team-level, permutation-invariant set model (sum or attention over players, as in Borghesi) is the matching design.
- Team-level shape features that are cheap to compute without IDs and plausibly carry signal:
  - team centroid and its velocity (the possessing team tends to move toward the opponent goal)
  - team length and width (possessing teams spread out; defending teams compact)
  - the defensive line's x position
  - the number of players of each team near the last-seen ball position
  - which team's nearest player is closest to the extrapolated ball

  No paper was found that gives accuracies for exactly this hand-crafted feature set.
- With 5 minutes of GT, pretraining a team-possession model on IDSSE (7 matches × about 90 min at 25 fps, with BallPossession and BallStatus per frame) and fine-tuning or calibrating on the project's GT is the realistic route to a learned model. Degrade the IDSSE input to look like broadcast data: drop the ball on about 41% of frames in camera-like runs, add noise, shuffle player order so there are no IDs.

### Gaps
- I could not find a published benchmark of team possession from team shape on broadcast-derived tracking.
- I did not read Fernández & Bornn pitch control, SoccerMap or Spearman as possession predictors. No source found reports pitch control used as a possession classifier feature with accuracy numbers.
- Bauer & Anzer (2021, counter-pressing / ball recoveries from DFL data) was not reached.

## 4. Physics-based ball inference during gaps

### Takeaway
Simple interpolation (Akima or cubic spline) is the standard preprocessing in the learned models above. Learned ball imputation from players is weak on its own: about 3.7–4.9 m error, and using it for event detection gives only 48% F1. Gap filling should be a feature or a constraint, not the decision.

### Cited Findings
- Borghesi et al. interpolate missing ball coordinates with an Akima spline before the TCN, and also pass an explicit missing flag. — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- Vidal-Codina et al. smooth noisy ball positions with Savitzky–Golay (order 2, 7 frames) before computing incoming and outgoing direction and speed at each control frame. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- In PathCRF's comparison, "Ball TP" reconstructs the ball from players and then applies Ball Radar possession rules. It gets 47.01% edge accuracy and 48.39% event F1, against 69.64% and 75.69% when possession paths are inferred directly. — [PathCRF HTML](https://arxiv.org/html/2602.12080)
- Kim et al. 2023 report about 3.69 m ball position error from player context (search summary, unverified). — [arXiv 2306.08206](https://arxiv.org/abs/2306.08206)

### Inferences
- For short gaps (under about 1 s) with the same team near both ends, linear or spline interpolation plus "same team before and after means same team throughout" is safe and cheap.
- For long gaps, use the last-seen and next-seen ball positions with their nearest players: who is near the ball at the next sighting is a strong bidirectional cue. Ground passes travel in near-straight lines, so a straight segment between sightings, together with which players lie on or near it at constant speed, is a reasonable ballistic prior. Lofted balls cannot be distinguished without z.
- Directly inferring team possession usually beats reconstructing the ball first and then applying rules. That is PathCRF's finding, and it argues against spending days on ball imputation.

### Gaps
- No paper was found that benchmarks ballistic-arc gap filling against splines on football tracking with real broadcast gaps.

## 5. Clever hybrid ideas: anchors + HMM, bidirectional inference, restart-team rules, ensembling

### Takeaway
The strongest evidence-backed "trick" is structured temporal decoding: an HMM, CRF or Viterbi pass over per-frame scores that forbids impossible transitions. It gave PathCRF about +8 to +10 points of event F1 over per-frame argmax and kept its 92% team-possession accuracy. The second is acausal (look-ahead) windows, which Borghesi used. Restart rules from the Laws of the Game turn D intervals into high-confidence anchors for which team has the ball afterwards.

### Cited Findings
- PathCRF results:
  - per-step argmax: 65.34% event F1, 2.80% illegal transitions
  - plus Viterbi constrained decoding: 73.14%, 0% violations
  - plus a learned dynamic masked CRF: 75.69%

  — [PathCRF HTML](https://arxiv.org/html/2602.12080)
- Borghesi's model is acausal on purpose: the target frame is the 48th of 64, so about 1 s of future context at 16 fps. The two-branch decomposition, dead/alive first and then home/away given alive, was the best architecture. — [Borghesi et al. 2023](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf)
- Vidal-Codina's set-piece patterns identify the player who restarts play (the "triggering player" at the corner mark, the sideline, the goal area or the centre mark). That player's team is the team in possession at the restart. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)
- In Vidal-Codina's framework, possession gains and losses are discrete events between "control frames". Receptions and interceptions are told apart by whether the previous loss came from a teammate or an opponent, a simple bidirectional rule. — [arXiv 2202.00804](https://arxiv.org/pdf/2202.00804v2)

### Inferences
- **Recommended pipeline** (reproducible in about 5–6 days):
  1. **Anchors.** Frames where the ball is visible and within about 1–1.5 m of exactly one team's player(s), per the Vidal-Codina or Morra thresholds, widened for broadcast noise. Also Smart Tagging set-piece intervals (D, then the restarting team).
  2. **Per-frame scorer.** A small model (gradient boosting on hand-crafted window features, or a Borghesi-style permutation-invariant TCN) trained on IDSSE frames degraded to broadcast-like quality, then calibrated on the 5 min of GT. It outputs P(D), P(H), P(A).
  3. **Decoding.** A 3-state HMM/Viterbi (H, A, D) with emissions from step 2 and anchors clamped. Transition costs should forbid sub-second flicker and could require D→H/A to agree with the restart-template team. This is the "logically consistent path" idea from PathCRF, applied at team level.
  4. **Restart-team rules** as constraints. If the ball goes out over a touchline or goal line, the restarting team is the opponent of the last toucher. A goal is followed by a kickoff by the conceding team. A goal kick goes to the defending team. These rules are in the IFAB Laws of the Game; the Laws text was not fetched here, and Vidal-Codina cite Laws 8 and 14–17 for their templates. So the label right after a D interval can be inferred from either side of it.
- **Bidirectional fill.** In a gap, if team X has the ball before and after with no restart in between, label X throughout. If it changes, place the switch where the Viterbi cost is lowest, typically when the ball would arrive near the new team. With future frames allowed, this is cheap and strong.
- **Is an NN clearly best?** Not with 5 minutes of in-domain GT. Borghesi's network beat rules by about 7 points, but it had 100K training windows from the same data source as its test set. PathCRF shows the bigger win is structured decoding, not the backbone, since the architecture spread in Borghesi is only ±1.7%. A hybrid (rule anchors + light model + HMM, optionally pretrained on IDSSE) is the most defensible choice for 8 days.

### Gaps
- No published accuracy was found for a team-level 3-state HMM over H/A/D. The benefit is inferred from PathCRF's player-level CRF results.
- No source quantifies how much restart-team rules improve possession labelling.

## 6. Competitions and hackathons on similar tasks

### Takeaway
The public competitions (DFL Bundesliga Data Shootout 2022, SoccerNet Ball Action Spotting) are video-based action spotting, not tracking-based possession labelling. Their winners use frame-level video CNNs with temporal heads, and the methods do not transfer directly. The nearest "benchmark" for this task is the IDSSE-based PathCRF test and Borghesi's private-data results.

### Cited Findings
- The DFL / AWS "Bundesliga Data Shootout" ran on Kaggle from 30 July to 13 October 2022 with a US$25,000 prize pool, co-organised with Sportec Solutions. — [bundesliga.com](https://bundesliga.com/en/aws-bundesliga-data-shootout)
- The task was spotting the timing of three event types (Play, Challenge, Throw-In) in match video, scored by average precision over time tolerances. Write-ups found are for 6th place (Fixstars, in Japanese) and other top entries using frame-level event/no-event classifiers. No first-place write-up was found. — [Fixstars Speaker Deck](https://speakerdeck.com/fixstars/kaggle-fixstars-corporation-20230509); [lRomul ball-action-spotting (inspired by the top solutions)](https://github-redirect.dependabot.com/lRomul/ball-action-spotting)
- SoccerNet 2024 Ball Action Spotting (12 classes, 8 teams, 61 submissions): T-DEED won with 73.39 mAP@1, against a baseline of 56.15. — [SoccerNet 2024 Challenges Results, arXiv 2409.10587](https://arxiv.org/html/2409.10587v1)

### Inferences
- The competition lesson that transfers: frame-wise classifiers plus temporal post-processing (peak picking or smoothing) win. Here that means smoothing or Viterbi.

### Gaps
- No winning write-up was found for the DFL Data Shootout. No Metrica, Friends of Tracking or PySport hackathon was found that targets tracking-based team possession labelling. SkillCorner's open-data page links to "PySport competition results", which were not reviewed.
