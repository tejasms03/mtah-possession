# Lessons: what moves spellF1, and what works for other teams

Written 16 Nov 2026 from our 10-match hold-out (7 DFL + 3 Metrica, each labelled by models
that never saw it, tracking damaged to match ours) and from other 2027-edition teams' public
READMEs. Numbers are bench spellF1 unless marked otherwise. Detail and logs: DECISIONS.md.

## 0. To-do, in order of priority and likely reward

Each row: the problem (from the sections below), the fix, the reward we can expect, what it costs
and needs, and where it stands. "Ceiling" = the perfect-information gain for that problem
(section 2); "expected" = a realistic range from our own or other teams' evidence. Every fix is
judged by adoption rule v2 (DECISIONS.md).

| # | Problem | Fix | Ceiling | Expected | Cost / needs | Status |
|---|---|---|---|---|---|---|
| 1 | Wrong team in quick turnovers, ball seen | **Carrier inputs**: per team, is a player moving with the ball (close, same velocity and heading), for how long before and after, did the ball slow on arrival, who reaches it first, who was last / is next in clear control (`features.carrier_extra`) | +0.134 | +0.01 to +0.03 (AndriesDKock +0.035 / +0.027 with control inputs; our probe: the "last team in control" signal is right on ~40% of v19's team errors) | retrain, features built in memory; no new dependency | running (7/10 matches) |
| 2 | Same; per-frame models can't see the shape of a turnover | **Sequence model, tree version**: a second stage reads the first stage's outputs at 15 offsets, -4 s..+4 s (`src/possession/sequence.py`) | +0.134 | +0.007 measured (both sources up, one match -0.004) | no retrain; second stage only | paired check running |
| 3 | Fixes 1 and 2 attack the same errors differently | **Sequence model on top of carrier** (and on top of torl) | +0.134 | +0.01 to +0.03 if the gains add | after 1; minutes | torl version running; carrier version next |
| 4 | Same, at full strength | **Sequence model, network version** (TCN over the per-frame inputs, like Opus-Operandi) | +0.134 | +0.01 to +0.05 (Opus +0.057 from a 0.63 base) | adds PyTorch (pinned, processor only, fixed seeds); bigger build | only if 2-3 show the idea works |
| 5 | Turnover inputs help DFL (+0.014) but cost Metrica (-0.013) | **Split torl**: turnover only vs restart layout only, keep the block that doesn't hurt Metrica | +0.134 | +0.005 to +0.01 | experts from the existing cache (turnover-only 6/10 saved) | queued |
| 6 | Models trained on one random damage per match | **Extra damage seeds** (2-3 per training match, like pranshu-nautiyal / Opus) | all | +0.003 to +0.01 | ~0.5 GB disk per match per seed; feature rebuild | not started |
| 7 | One model copy is noisy | **Seed averaging** (3 copies averaged) | all | 0 to +0.005 (v14's 5-copy average tied on the old bench) | 3x training time, no rebuild | running |
| 8 | Labelling settings tuned for the missing-ball case also swallow real short possessions with the ball seen | **Seen-ball switch settings**: shorter minimum length and no blip merge where the ball is seen | +0.113 (short possessions) | small alone; pays off only with better evidence (1-4) | decoder change, minutes | after 1-4 |
| 9 | Wrong team while the ball is missing | **Sequence model on hidden-ball frames** (Opus used its TCN exactly there); **PFF World Cup data** (20 matches, real occlusions, weight 0.5); **broadcast tracking** to fill ball gaps | +0.048 | sequence: part of 2-4; PFF: unknown, adopted by Opus | PFF: your download and terms, several GB; broadcast: your decision (set aside earlier) | sequence part in progress; PFF and broadcast not started |
| 10 | Live play called dead while the ball is missing | **Restart-layout inputs** for the dead-ball model (part of torl); **whistle** from the tactical video's sound, if it has any | +0.027 | small | rl: existing cache; whistle: video download, audio check | rl in item 5; whistle not started |
| 11 | Each new input could leak the answer (DFL freezes the ball at stoppages) | Keep ball-motion inputs out of the dead-ball model (`model.dead_drop_prefixes`); check every new block before benching | - | protects every gain | review per block | done for to_ / rl_ / ca_ |
| 12 | Rules and decoder tuning on the same signals | **Stop**: no more hand rules or decoder sweeps unless new evidence arrives | - | 0 (13 rules, best +0.0004) | - | closed |
| 13 | Public 5 minutes tempting as a guide | **Correctness check only**; never choose by it | - | - | - | standing rule |

Whatever wins: switch it on in `params.yaml`, run the pipeline on our match, compare with v19
(frames agreeing, possession count, dead share), fresh-clone check, update SUBMISSION_SHA256
and BEST.csv, record in DECISIONS.md, commit.

## 1. How the scorer works, and what follows

1. **spellF1 scores whole possessions, not frames.** A possession counts only if ours overlaps
   the true one well (IoU averaged over 0.3-0.7). v19 gets 90% of frames right and scores
   0.71: one wrong flip splits a possession into three. The possession count and the
   boundaries matter more than frame accuracy.
2. **Match the labels' own conventions.** Merging A-H-A blips shorter than 1.5 s (as the
   training labels do) and giving the ball to the last team to touch it before a stoppage were
   part of v19's +0.012.

## 2. Where the remaining points are (perfect-information test, v19)

Fix one kind of error with the truth on the bench and rescore (`outputs/logs/oracle.py`):

| Perfect on... | spellF1 | gain |
|---|---|---|
| nothing (v19) | 0.710 | |
| dead vs live, every frame | 0.737 | +0.027 |
| **team, every live frame** | **0.910** | **+0.200** |
| team, ball seen | 0.844 | +0.134 |
| team, ball missing | 0.758 | +0.048 |
| team, inside possessions under 4 s (9% of live time) | 0.823 | +0.113 |

So the target is **team calls in quick turnovers, mostly with the ball visible**. Dead-ball
work has little left to give.

## 3. What has worked for us

- Training on damage that copies our tracking (v17): our biggest single jump.
- New information for the models: player-motion inputs (v18, +0.016, every match better).
- Labelling settings matched to the training labels (v19, +0.012).

## 4. What has not, and why

- **Hand-written football rules** (13 tried): best +0.0004. They reuse the same noisy signals
  (nearest team, last sighting, player motion) the models already see.
- **Labelling-step tuning**: used up; every setting is flat.
- **Smart Tagging as evidence**: hurt (v15). Only the kick-off rule helps.
- **Ghost ball (ball guessed from players), gap-end anchoring, peak insertion**: no gain.
- **The public 5 minutes**: 17 possessions; 90% range about 0.3 wide (v15 0.35-0.62, v19
  0.32-0.65). A correctness check only, never for choosing.
- **Leaks**: DFL freezes the ball at every stoppage, so the dead-ball model must never see
  ball-motion inputs (`model.dead_drop_prefixes`).

## 5. What works for other teams (public READMEs, 2027 edition)

| What | Who | Their result | Our status |
|---|---|---|---|
| Sequence model over per-frame inputs (TCN) | Opus-Operandi | +0.057, every seed better | tree version: +0.007, under check |
| Ball-control inputs (reach first, control episodes, touch log) | AndriesDKock | +0.035 / +0.027, 10/10 matches | carrier retrain |
| 11 nearest players per team read one by one by a network | Derekko-web | part of their best ensemble | not tried |
| Several damage seeds per training match | pranshu-nautiyal, Opus-Operandi | their proxy 0.711 | not tried |
| Averaging model copies; tree + network blend (0.4 / 0.6) | jbesnardrodru, Derekko, Opus | small, steady | seed averaging under test |
| PFF World Cup, 20 matches at weight 0.5 (real occlusions) | Opus-Operandi | adopted | needs our download and terms |
| Two model sets (raw, provider artefacts removed), averaged | Derekko-web | adopted | partly: frozen-ball inputs dropped |
| Broadcast tracking aligned to fill ball gaps | AndriesDKock | optional | set aside |
| Automatic decoder tuning (Optuna) | Opus-Operandi | adopted | our settings are flat |

Failed for others too: SkillCorner hurt Opus-Operandi's damaged validation (as for us);
literature-style possession cuts after decoding gave them < +0.01 (our peak insertion: 0);
a Bi-LSTM on raw inputs lost to trees for debjyotydasgupta.

## 6. The pattern

Every real gain, ours and theirs, came from **new information for the models** (motion,
control, time context, better-matched training data). None came from more rules or more
decoder tuning on the same information.
