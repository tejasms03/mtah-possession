# Dead-ball (out-of-play) and restart detection from tracking data, and referee whistle detection from match audio

Context: v12 pipeline, one match, tactical-camera tracking, 25 fps, ball missing 41%, no player IDs. Constraints: about 1.5 days, laptop CPU, deterministic, organisers must be able to rerun it.

## 1. Rule and feature ideas for out-of-play detection from tracking (templates, Vidal-Codina 2022, Lang 2022, Gudmundsson & Wolle)

### Takeaway
The best player-only published model (Lang et al. 2022, AdaBoost, 102 Bundesliga matches) reaches 92% frame accuracy and F1 0.80 on whole stoppages longer than 2 s. Throw-ins are its worst type: 74% recall, mean length 13 s. Start and end times are off by about 1.7 s on average. Vidal-Codina et al. 2022 do not detect dead ball at all. They assume a dead/alive flag is given, then classify the restart with simple player-position "triggers" plus a "pattern" check at the first live frame. On that task they report 89–100% precision and 82–100% recall per restart type. Their triggers can be ported directly as features or rules in our decoder.

### Cited Findings
**Lang, Wild, Isenko & Link 2022 (Sci Rep 12:16291), player-only in-game status**
- Data: 102 matches (82 Bundesliga, 20 Bundesliga 2, season 2017/18), TRACAB optical tracking at 25 fps. In-game status was annotated manually by human observers. Split: train 45, validation 10, test 47 (all Bundesliga 2 matches in the test set). — [Lang et al. 2022](https://doi.org/10.1038/s41598-022-19948-1) ([PMC PDF](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf))
- Features: for each of the 22 players, x, y, speed and acceleration, sorted by the match-sheet order. Second-half coordinates are flipped. The training data is doubled by swapping home and away teams. Each frame also gets the same features from 0.4, 0.8, 2, 4, 20, 40 and 80 s in the past and in the future (14 time shifts), giving 1320 inputs (15 × 4 × 22). — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- Frame-level results on 47 test matches. AdaBoost: accuracy 92.0%, precision 93.4%, recall 93.0%, F1 93.1%. Random forest: 89.3% accuracy. Logistic regression: 86.7% accuracy. Decision tree: 84.6% accuracy. — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- Whole-stoppage results (stoppages ≥2 s, matched by IoU) after a median filter of 91 frames (3.64 s), with the filter width tuned on the validation set. AdaBoost: precision 81.1%, recall 78.9%, F1 79.8%. Random forest: F1 67.9%. — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- Timing error: mean absolute shift 1.79 ± 3.44 s at the start of a stoppage and 1.68 ± 3.47 s at the end. 51% of starts and 58% of ends are within 1 s; 78% and 80% are within 2 s. — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- Per type (n / mean length / recall / mean start shift / mean end shift):

  | Type | n | Mean length | Recall | Mean start shift | Mean end shift |
  |---|---|---|---|---|---|
  | Kick-off | 153 | 50.2 s | 91.6% | 4.80 s | 0.90 s |
  | Penalty | 13 | 61.7 s | 86.7% | — | — |
  | Free kick | 1290 | 26.9 s | 81.3% | 1.80 s | 1.87 s |
  | Corner | 367 | 26.9 s | 80.7% | 1.58 s | 1.81 s |
  | Goal kick | 568 | 23.0 s | 79.8% | 2.00 s | 1.55 s |
  | Throw-in | 1495 | 13.0 s | 74.3% | 1.42 s | 1.54 s |

  — [Lang et al. 2022, Table 2](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- Failure modes the authors name:
  - Short stoppages (2–4 s) of free kicks and throw-ins are missed, because the filter over-smooths or because "the players' behavior shows no difference as the match resumes promptly".
  - Short in-play spells between two stoppages get stitched together.
  - Bursts of player activity inside long stoppages produce false "in play" segments.
  - Throw-ins and corners are often followed by fast toggling of the status.

  — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- Feature importance was highest for time shifts within [−4 s, +4 s]. Near the prediction point, past features mattered more than future ones. — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- The authors expect that adding the ball position would bring accuracy close to 100%, because kick-offs, corners, penalties and throw-ins depend on where the ball is. — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)

**Gudmundsson & Wolle**
- Lang et al. report that Gudmundsson & Wolle achieved F1 0.94 for "ball-out" event detection using both ball and player positions. — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf). Primary paper: "Football analysis using spatio-temporal tools", SIGSPATIAL 2012, pp. 566–569, DOI 10.1145/2424321.2424417 — [researchr](https://researchr.org/publication/GudmundssonW12)

**Vidal-Codina, Evans, El Fakir & Billingham 2022 (Sports Engineering 25:18), restart templates**
- The method assumes a dead-ball flag already exists: "signaled by either a binary boolean or by the absence of ball tracking data". TRACAB provides a boolean; Track160 and Hawk-Eye leave the ball missing while it is dead. — [Vidal-Codina et al., arXiv 2202.00804](https://arxiv.org/abs/2202.00804v3)
- Triggers are player configurations evaluated during the dead-ball interval [d0, dc]:
  - Kick-off: all players in their own halves (tolerance ϵk1) and at least one player within ϵk2 of the centre mark.
  - Penalty: exactly one player on the goal line between the posts; one opponent in a box from ϵp2/4 in front of to 3ϵp2/4 behind the penalty mark; everyone else outside the box and more than 9.15 m from the mark.
  - Goal kick: at least one player inside their own goal area (tolerance box).
  - Corner: at least one player within ϵc of an active corner mark.
  - Throw-in: at least one player beyond the "auxiliary sideline" (the touchline minus ϵt).

  — [Vidal-Codina et al.](https://arxiv.org/abs/2202.00804v3)
- How triggers are used:
  - A trigger must be "complete", meaning it switches on at some frame d1 and stays on until the last dead frame dc.
  - The pattern check then confirms it: at the first live frame, the ball must be within the possession radius Rpz of the triggering player. If several players qualify, the one closest to the ball is the taker.
  - A fixed order breaks ties between triggers. If nothing matches, the restart defaults to a free kick.

  — [Vidal-Codina et al.](https://arxiv.org/abs/2202.00804v3)
- Possession radius Rpz was 0.5 m for one provider and 1 m for the other two; duel radius 1 m. The authors say the method can be applied to "one tactical camera or broadcast footage" data by enlarging the possession radius and re-tuning hyperparameters. The tolerance values are in Online Resource 1, sections S3–S4, which I did not retrieve. — [Vidal-Codina et al.](https://arxiv.org/abs/2202.00804v3)
- Set-piece results against STS annotations (Fig. 7b), with types in the order kick-off, penalty, corner, throw-in, goal kick, free kick. The figure lists two rows of numbers: 100/100/97/92/96/89% and 100/100/97/95/95/82%. The text says corner, throw-in and goal kick have "supra-90% precision and recall", and free kicks are the worst. From the PDF text extraction I could not tell which row is precision and which is recall. — [Vidal-Codina et al.](https://arxiv.org/abs/2202.00804v3)
- Confusions they report:
  - Throw-ins and free kicks near the corner mark labelled as corners.
  - Free kicks near the touchline labelled as throw-ins.
  - Free kicks in the goal area labelled as goal kicks.
  - Offsides labelled as fouls.

  They suggest using the ball's height (z) for throw-ins, since the ball is above head height. Running time was about 3 min per match on 1 vCPU. — [Vidal-Codina et al.](https://arxiv.org/abs/2202.00804v3)
- A later IRI/UPC paper criticises touch-based methods like this one as not robust to missing data, because they "need full visibility of the arrangement of the players on the pitch". — [search snippet via IRI/UPC](https://www.iri.upc.edu/download/scidoc/3004)

### Inferences
- **Throw-in template as a dead-ball feature** (about 3–4 h, low risk). For each frame, compute the distance of the nearest player to each touchline, with a flag for "beyond the touchline minus ϵt". In our data the touchline position must be calibrated, so start with ϵt ≈ 1–1.5 m. Combine with:
  - the last known ball position within about 3 m of the touchline, or the ball missing;
  - no other player within about 2–3 m of that player;
  - a "complete trigger" persistence rule: the trigger stays on until the restart.

  This targets our biggest weakness. Throw-ins average 13 s and have the lowest recall even in Lang's best model (74%). Players do not slow down during throw-ins, but the thrower does stand still at the line. A wing player stationary near the touchline for ≥1–2 s while the ball is missing is a strong cue. Without player IDs, any one player standing near the touchline still works.
- **Corner and goal-kick templates** (about 2 h, low risk):
  - Corner: one player within about 2–3 m of a corner mark, plus many players (≥8–10) inside the penalty area.
  - Goal kick: one player (the goalkeeper) inside the goal area and a defensive back-off.

  These templates also fix timing. The restart is the first frame where the ball (when visible) leaves the taker's radius. This addresses our "goal-kick restarts late" problem: anchor the restart on the taker's ball contact, or on the moment the template stops holding.
- **Lang-style multi-scale time-shift features** (about 3 h, medium risk of overfitting on 10 matches). Add team-level aggregates at ±0.4, 0.8, 2, 4, 20, 40, 80 s rather than per-player values, because we have no IDs. Examples: mean speed, how spread out the team is, the centroid, the minimum distance to a touchline. Lang found the ±4 s window most important, with past features weighted more. Our "stoppage starts several seconds early" error may come from too much look-ahead; consider weighting past features more, or asymmetric windows.
- **Do not median-filter at 3.64 s.** That is what makes Lang miss short throw-ins. Our decoder already does this job with a transition cost. Keep a lower minimum-duration prior for throw-in-type stoppages (around 4–5 s) than for other stoppages.
- Expected ceiling: published player-only F1 is about 0.80 for whole stoppages, with about ±1.7 s timing error. Being within about 1–2 s of the truth is realistic; tighter than that needs the ball or audio.

### Gaps
- The Vidal-Codina tolerance values (ϵt, ϵc, ϵg, ϵk1, ϵk2) were not retrieved. They are in the Springer supplementary PDF ([ESM](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs12283-022-00381-6/MediaObjects/12283_2022_381_MOESM1_ESM.pdf)).
- Which Fig. 7b row is precision and which is recall could not be resolved from the text extraction.
- I could not retrieve the Gudmundsson & Wolle method details. Only the F1 0.94 figure, cited secondhand, is available.
- No published per-type results exist for tactical-camera data with 41% of the ball missing. All the figures above come from multi-camera stadium tracking.

## 2. Referee whistle detection in broadcast or stadium audio (methods, accuracy, self-calibration, audio extraction and sync)

### Takeaway
Raventós et al. 2015 report 93% recall and 88% precision on about 10 h of professional match audio. Their method looks at energy in the 3.5–4.5 kHz band, then rejects frames whose spectrum is spread out (high entropy), then keeps frames with 2–3 spectral peaks. They do not publish threshold values, so we would set them from our own match (self-calibration). Learned detectors exist (RoboCup CNNs; PANNs, a set of general-purpose audio classifiers trained on Google's AudioSet), but their crowd-noise numbers are weaker or unverified. The deciding first step is to check whether the tactical mp4 has an audio track at all.

### Cited Findings
**Raventós, Quijada, Torres & Tarrés 2015 (SpringerPlus)**
- Audio is processed at 48 kHz in 4800-sample frames (100 ms). The DFT bins in the 3.5–4.5 kHz band are computed with the Goertzel algorithm, a cheap way to compute a few frequency bins without a full FFT. — [Raventós et al. 2015, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4485672)
- Three tests are applied to each frame:
  1. Band energy must exceed a threshold.
  2. Treat the normalised band spectrum ρ(k) = X(k)² / Σ X(r)² as a probability distribution and compute its entropy. Frames with high entropy (energy spread across the band) are rejected.
  3. Keep only frames with 2 or 3 spectral peaks above a threshold ("Only audio frames with two or three peaks are finally validated").

  None of the three threshold values is given. — [Raventós et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4485672)
- Results: recall 93%, precision 88%, on five complete matches (three first division, two second division, about 2 h each), annotated with audio and video. The paper argues that earlier band-energy-only detectors give many false positives under cheering. Its two-stage refinement is meant to reject vuvuzelas, horns and supporters' whistles. — [Raventós et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4485672); [arXiv 1411.6496](https://arxiv.org/pdf/1411.6496)

**RoboCup detectors**
- rUNSWift (2015): looks for a "rectangle" in the NumPy FFT spectrogram in the 2000–4000 Hz band lasting at least 250 ms. Detection is only switched on in the "Ready" state, when a whistle is expected. The team reported a perfect whistle record at RoboCup 2015. Another source puts referee whistles at 2.5–3.5 kHz in most cases. — [search summary of rUNSWift 2015 / Springer chapter](https://rd.springer.com/chapter/10.1007/978-3-319-29339-4_6)
- Continuous-kernel CNN on the NAO robot, using 1024-sample FFT windows at 44.1 kHz:
  - Test set: accuracy 98.0%, precision 79.4%, recall 90.4%.
  - Real games while the robots were playing: recall 80%, precision 100%.
  - Kick-off waits ("Ready/Set" state): 100% on both.

  Failure case: whistles after goals are drowned out by crowd cheering. Training data had a 10:1 imbalance (60k no-whistle vs 6k whistle windows). — [arXiv 2411.17347](https://arxiv.org/pdf/2411.17347)
- Other public code: [bit-bots/whistle_detection](https://github.com/bit-bots/whistle_detection) (not inspected).

**PANNs (Kong et al.)**
- The Cnn14 checkpoint `Cnn14_mAP=0.431.pth` is on Zenodo record 3987831. There are also 16 kHz (mAP 0.438) and 8 kHz (mAP 0.416) variants. The models output 527 AudioSet classes. — [PANNs paper arXiv 1912.10211](https://arxiv.org/pdf/1912.10211); [audioset_tagging_cnn README](https://github.com/zanilzanzan/audioset_tagging_cnn/blob/master/README.md)
- I could not find the per-class AP for the AudioSet "Whistle" class in the retrieved text. It is only shown in a figure covering all classes. — [PANNs](https://arxiv.org/pdf/1912.10211)

**Audio/video sync (AAC priming)**
- Apple's players assume an AAC encoder delay of 2112 samples by default. FFmpeg's native AAC encoder produces 1024 samples of delay, and FDK-AAC usually 2048. MP4 edit lists signal how much to trim. Desync bugs happen when an edit list is missing, or is applied twice. — [Apple TN2258](https://developer.apple.com/library/ios/technotes/tn2258/_index.html); [Apple QuickTime Appendix G](https://developer.apple.com/documentation/quicktime-file-format/appendix_g_audio_priming_handling_encoder_delay_in_aac.md); [Vimeo engineering blog](https://medium.com/vimeo-engineering-blog/a-brief-history-of-gapless-audio-and-what-you-can-do-about-it-ea9e1c343215); [FFmpeg trac #11031](https://trac.ffmpeg.org/ticket/11031)

### Inferences
- **Step 0, check for audio (5 min).** Run
  `ffprobe -v error -select_streams a -show_entries stream=index,codec_name,sample_rate,channels,start_time,duration -of json match.mp4`
  An empty `streams` list means there is no audio, and the whole whistle branch is dropped at no cost. Also compare `start_time` of the video stream (`-select_streams v`) with the audio stream.
- **Deterministic extraction.** Run
  `ffmpeg -nostdin -hide_banner -i match.mp4 -map 0:a:0 -vn -ac 1 -ar 48000 -c:a pcm_s16le -bitexact match_48k.wav`
  This decodes to mono 16-bit PCM. The ffmpeg decoder applies the edit list and skips priming by default. To be safe, log the ffprobe `start_time` of each stream, and define audio time t_audio = sample_index/48000 + (audio start_time − video start_time). Then frame = round(t_audio × 25). The worst case for an unhandled priming error is 2112/48000 = 44 ms, about 1 frame at 25 fps, which is negligible compared with the ±1.7 s stoppage timing errors. Pin the ffmpeg version in the README.
- **Raventós-style detector, self-calibrated** (about 4–6 h, numpy/scipy only, fully deterministic):
  1. Short-time FFT with a 100 ms window and 50 ms hop at 48 kHz.
  2. Band energy E_b at 3.5–4.5 kHz. Also try 2–4.5 kHz, because rUNSWift and other sources put whistles at 2–4 kHz. Pick the band by where the top peaks of the band-to-broadband ratio cluster in our match.
  3. Use the ratio E_b / E_total, not raw energy, to normalise for crowd loudness.
  4. Spectral entropy within the band must be below the match's 5th–10th percentile.
  5. 1–3 peaks, and the dominant peak frequency must be stable (±50 Hz) for ≥150–250 ms.

  To self-calibrate:
  - Fit thresholds as match percentiles. Whistles are rare, roughly <1% of frames.
  - Use the 6 kick-off tags as known positives. A whistle is always blown to start play, so the largest band-ratio peak in the ±3 s window around each kick-off restart gives the whistle's peak frequency and level for this referee.
  - Set thresholds at, say, 50% of the median positive level.
  - Use free-kick tags (5) as a second, weaker positive set. A whistle usually comes several seconds before the restart.
- **PANNs as an alternative** (about 3–4 h; medium risk).
  - Benefits: public, pinned and deterministic on CPU (Cnn14 16 kHz, Zenodo DOI pinned, torch in eval mode).
  - Risks:
    - The download is about 300 MB.
    - AudioSet "Whistle" also covers human whistling and other whistles, and its AP on crowd audio is unverified.
    - Running it over 90 min at 1 s hop on a laptop CPU is feasible, but slower than the DSP approach.
  - Recommendation: use PANNs only as a cross-check if the DSP detector is noisy.
- **Expected value: medium–low.** Tactical-camera recordings often have weak or no stadium audio. And even at 88–93% precision/recall, whistles mostly mark fouls, offsides, goals and half-time, not throw-ins (see section 3). So whistles would mainly sharpen start times for free kicks and kick-offs, which are not our weakest category.

### Gaps
- No published precision/recall exists specifically for tactical-camera stadium audio.
- Raventós threshold values are not published.
- The per-class AP of PANNs on "Whistle" was not retrieved, and the licence of the checkpoint files was not verified.
- The rUNSWift code repository was not inspected.
- It is unknown whether our mp4 has audio. ffprobe must be run on it.

## 3. How whistles map to dead-ball starts (fouls, offsides, goals, half-time) vs ball-out events

### Takeaway
Under IFAB guidance, the whistle is needed to stop play for free kicks and penalties (so fouls and offsides), cautions, injuries and substitutions, and to start halves and restart after goals. It is not needed to stop play for an obvious goal kick, corner, throw-in or goal, or to restart from them. So a whistle detector mainly helps with foul and offside stoppages and kick-offs. For throw-ins, corners and goal kicks, the stoppage start is the ball crossing the line, often with no whistle.

### Cited Findings
- Law 9.1: the ball is out of play when "it has wholly passed over the goal line or touchline on the ground or in the air" or when "play has been stopped by the referee". It is also out of play when it touches a match official and certain conditions follow, in which case play restarts with a dropped ball. — [IFAB Law 9](https://www.theifab.com/laws/latest/the-ball-in-and-out-of-play/)
- IFAB "Body language, communication and whistle" guideline, as summarised by search:
  - The whistle is needed to start play (each half, extra time, after a goal) and to stop play for a free kick or penalty, a caution or send-off, an injury, or a substitution.
  - It is not needed to stop play for a clear goal kick, corner, throw-in or goal, or to restart from those, except in tight decisions.

  — [IFAB guideline page](https://www.theifab.com/laws/latest/guidelines/body-language-communication-and-whistle/); [askasoccerreferee: When to use the whistle](https://www.askasoccerreferee.com/?p=1183). I could not fetch the full guideline text, so this rests on a search summary that agrees across both sources.
- Law 5: the referee "allows play to continue until the ball is out of play if a player is only slightly injured". — [IFAB Law 5](https://www.theifab.com/laws/latest/the-referee/)

### Inferences
Mapping for our 60 tagged restarts:

| Restart type | Count | Whistle at stoppage start | Whistle at restart | What the whistle tells us |
|---|---|---|---|---|
| Kick-off | 6 | Usually yes (after a goal or end of half) | Yes | Strong anchor for both ends |
| Free kick | 5 | Yes, at the foul or offside | Only if ceremonial (wall set) | Sharp start time |
| Throw-in | 26 | Usually no | No | Little help; use the touchline template |
| Corner | 12 | Usually no | No | Little help; use the corner template |
| Goal kick | 10 | Usually no | No | Little help; use the goal-area template |

- A whistle not followed by a dead-ball segment within about 1–3 s is either a false positive or advantage play. Use the whistle as a soft prior that lowers the alive-to-dead transition cost in the window [t_w − 0.5 s, t_w + 1.5 s]. Do not use it as a hard label.
- For our "stoppage starts several seconds early" problem: when a stoppage is confirmed and a whistle falls inside it, snap the start to the whistle for free-kick and kick-off stoppages. For throw-ins, corners and goal kicks, snap the start to the last frame before the ball leaves the field, or to the start of the ball-missing run near a line.

### Gaps
- No quantitative data was found on how often referees actually whistle for out-of-play situations in professional matches. Whistling is discretionary in tight cases.

## 4. How ground-truth "dead ball" is defined in DFL and Metrica tracking (BallStatus)

### Takeaway
The DFL open data (Bassek et al. 2025) defines BallStatus as 0 = inactive, 1 = active. It says only that the match state "is inactive when the match is interrupted by the referee, for example after a foul or during a substitution". It does not say whether the flag switches at the ball crossing the line or at the whistle, nor when it switches back. Vidal-Codina's convention treats the dead interval as ending at the first live frame, where the taker has the ball. Lang's ground truth was annotated manually by human observers. I found no Metrica definition.

### Cited Findings
- DFL/IDSSE: the ball data include "the match state (BallStatus; 0 = ball inactive, 1 = ball active)". "The match state is inactive when the match is interrupted by the referee, for example after a foul or during a substitution." The paper does not state whether the flag is set manually or automatically, what triggers each switch, or how the ball position is handled during interruptions. There are 7 matches from the 2022/23 season with 25 Hz cameras. — [Bassek et al. 2025, Sci Data, PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/); [HF mirror README](https://huggingface.co/datasets/pysport/idsse-data/blob/main/README.md)
- Lang et al.'s TRACAB-based ground truth: in-game information was "collected manually by human observers". — [Lang et al. 2022](https://rcastoragev2.blob.core.windows.net/da8383a4790e53f7460e9c4a819d6937/PMC9522646.pdf)
- Providers differ in how they mark dead ball. TRACAB uses an alive/dead boolean; Track160 and Hawk-Eye mark it implicitly by leaving the ball missing. — [Vidal-Codina et al.](https://arxiv.org/abs/2202.00804v3)
- Vidal-Codina et al. also note that manual event timestamps can be more than 10 s off, and that analysts disagree by 10–15% when tagging events. — [Vidal-Codina et al.](https://arxiv.org/abs/2202.00804v3)

### Inferences
- The DFL wording ("interrupted by the referee") suggests the dead state may start at the whistle or decision rather than the instant the ball crosses the line. For ball-out events these are nearly the same time anyway. Our outside model is trained on DFL labels (plus Metrica), so any convention offset is baked into it.
- Cheap check (about 1 h): on the 10 training matches, measure the lag between ball-crosses-line (from ball x/y, before the ball freezes) and the BallStatus 1→0 switch. Also measure the lag between the restart touch (first ball speed above about 2 m/s) and the 0→1 switch. Use the median lags to shift our decoder outputs to match the convention of whichever ground truth the organisers will score against.
- The ball "freezing" at stoppages is the user's observation of the DFL data and is not documented in the paper. Keep excluding ball-motion features for the DFL-trained model.

### Gaps
- No Metrica Sports documentation on how dead ball is defined was found or retrieved, so it is unknown whether Metrica marks dead ball as ball-missing or with a flag.
- DFL does not document the trigger for BallStatus switches (line crossing vs whistle, manual vs automatic) or the end condition (restart touch vs whistle).
- It is unknown which convention the organisers' ground truth for our match uses.
