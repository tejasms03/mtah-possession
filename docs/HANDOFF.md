# Handoff (12 Nov 2026, evening)

For whoever (human or AI assistant) picks this up next. Read this, then `docs/PLAN.md`,
`docs/DECISIONS.md` (every decision with evidence, chronological) and `README.md`.

## 0. Rules you must not break

**Organiser clarification (3 Nov, recorded in verneylmavt/more-than-a-hack-2027 ORGANIZER-CLARIFICATION.md, overrides RULES.md):** every fit, tuning, calibration or pseudo-label training must use external data only; the provided match (tracking, Smart Tagging) may only be an inference input. The public window may only check a frozen pipeline's correctness, never guide selection. Choose versions with `uv run possession select`.

## 0b. Data and AI

Competition T&C (vendor/mtah/OVERVIEW.md "Use of Large Language Models", FAQ.md): LLMs may be
used anywhere, but **no competition video, dataset or extract of one may be uploaded or shown to
any hosted/cloud LLM** (ChatGPT, Claude, Gemini, ...). That covers everything in `data/`, the
match files in ~/Downloads, and our output CSVs in `outputs/` (derived per-frame from the data).
Share code, docs, this file, and summary scores only. Run commands locally and report aggregates.

Other rules that bite: labels must come from an algorithm (no hand labelling or hand fixes); no
human-produced data about this match; the repo must regenerate the **byte-identical** file;
never commit match data; everything used must be accessible to the organisers and explained in
the technical document.

## 1. The task in one paragraph

Label every frame of Barcelona (H) vs Athletic (A), 199,319 frames at 25 fps, as H / A / D (dead
ball); X only on ungraded frames (`submission.write` forces that). Graded: frames 11,185-90,651
and 110,808-190,197. Ranked by **spellF1** (possessions matched by temporal IoU, mean over IoU
0.3-0.7; higher is better), tie-break macroF1. Public ground truth = frames 11,185-18,684 only
(17 possessions); the full-match GT is hidden. Deadline Mon 17 Nov; we plan to **submit Sun 15
Nov**. Deliverables: `<team>.csv`, public repo (flip `tejasms03/mtah-possession` from private to
public on submission day, after checking no data is in it), `<team>.pdf` technical document.

## 2. Where things stand

**Best file: v15** (v14 + outside training labels merged at 1.5 s + Smart Tagging set-piece hints off; whole-match bench 0.7912 vs v14 0.7457 on the same labels, every match better). Before that **v14** (v6 + idea-2 features + dead ball before each half's kick-off tag; NO training on the provided match, per the organisers' 3 Nov clarification; single model set). Before that **v12** (v11b + idea-2 team features; chosen on the whole-match Metrica select bench, 0.7721 vs 0.7576; public check 0.588 is a format check only). Before that v11b (5-model average of v7's setup; public check 0.600, macroF1 0.861), saved as `outputs/saved/BEST.csv`; previously v7 (local only, gitignored), produced by the
committed code at commit `070067d` with `params.yaml` as committed.

| version | public spellF1 | macroF1 | dead-ball F1 | note |
|---|---|---|---|---|
| organisers' baselines | 0.33-0.42 | 0.58-0.67 | | full-match floor 0.2433 |
| v1 (DFL-only models) | 0.563 | 0.714 | 0.45 | barely predicts dead ball |
| v6 (+Metrica, split tuning) | 0.514 | 0.724 | 0.55 | |
| **v7 (+our-match set-piece labels)** | **0.588** | **0.856** | **0.85** | **current best** |

Public window = 17 possessions, so each possession is worth about +-0.05: treat it as a check,
never as the thing we tune on. Bench (honest, leave-one-match-out) spellF1: Metrica 0.78, DFL
0.45 (DFL is no longer a valid dead-ball test, see 4).

Other teams' public READMEs (71 repos, 6 report a score): the best reported public-window score is
0.581 / macroF1 0.851 (Wichai-pan); most report nothing. Derekko-web/mtah2027-submission is the
most thorough-looking competitor.

## 3. The system (src/possession/)

```
load.py      own ATD parser (frame:46 slots x,y,speed:ball x,y,speed); slot->team by matching kloppy
clean.py     stage 1: off-pitch/impossible ball points out, duplicates, slot-free player velocities
ballfilter.py stage 2: Savitzky-Golay ball, bridge gaps <=0.48 s, touches (velocity change near a player)
features.py  stage 3: game block (dead-ball notes) + per-team blocks (5 nearest, touches, shape in
             attacking direction from where each team's deepest player sits), multi-scale windows
deadball.py  rule dead-ball evidence + Smart Tagging set-piece hints
model.py     model A (dead?) + model A-full (Metrica-only, keeps ball motion) + model B (whose ball,
             team-blind own/opp), sklearn HistGradientBoosting (LightGBM needs libomp on macOS);
             own_match_rows (set-piece weak labels, used in v7); own_team_rows (item 1, see 5)
decode.py    Viterbi H/A/D with minimum lengths, touch-gated switch and restart costs (numba);
             refine_boundaries exists but is OFF (it made things worse, see DECISIONS)
tune.py      coordinate search of decoder/model dials on spellF1 over the bench
external/idsse.py   7 DFL matches (CC-BY 4.0), damaged to look like our ATD (players + ball gaps)
external/metrica.py 3 Metrica sample matches, labels built from their events, players damaged
external/bench.py   bench scoring with the organisers' possession_metrics.py; IDSSE blips <2 s merged
pipeline.py  CLI: `uv run possession run|validate|check-data|clean|idsse|bench|tune`
```

`possession run` = clean our ATD -> train A, A-full, B on the 10 outside matches -> predict our
match -> build weak dead/alive labels from our set-piece tags -> retrain -> decode -> CSV.
About 2-4 min. All settings in `params.yaml`, reasons in `docs/DECISIONS.md`.

## 4. Hard-won findings (do not rediscover them)

- **DFL freezes the ball on every dead frame** (speed exactly 0, even in the clean files). Any
  ball-motion feature gives dead ball away on DFL and fails on our match. Model A drops them
  (`model.dead_drop_prefixes`); ball motion is only learned from Metrica (A-full).
- **SkillCorner has a similar leak** (per Derekko-web: ball detected on 18% of dead vs 91% of
  live frames). Don't use it for dead ball.
- **Metrica labels from events**: dead ball must end at the first action after the ball goes out,
  not the next SET PIECE (goal kicks are logged as the keeper's PASS). Fixed; was a big error.
- **IDSSE possession is touch-level**; merging A-H-A blips <2 s matches our GT's 3.4 possessions/min.
- **The damaged-IDSSE speed column was noise-dominated** (fixed in `idsse.speeds`).
- **Our GT switches come 0.5-2 s before our first detected touch**, so snapping to touches hurts.
- **Our match's stoppages differ from Metrica's**: the biggest single gain (v6 -> v7) came from
  learning from our own match's Smart Tagging set pieces (public window excluded from that training).
- Full-match dead-ball share is ~51%; checked: the long stretches are goals (2-3.7 min each,
  ending at kick-off tags) and near-motionless stoppages. Judged plausible; not changed.
- Remaining errors in the public window are **all short possessions (0.8-8 s) on the team side**.

## 5. In flight / next steps (in order)

1. **Reproducibility check: PASSED for v12 (current submission, `8a469ed6...`)**, and earlier for v11b (fresh clone, 1 thread, ~35-40 min, matches SUBMISSION_SHA256 `9703d412...`). Earlier: **PASSED for v7.** Fresh clone, single-threaded, byte-identical to
   BEST.csv (SHA-256 in `SUBMISSION_SHA256`; BEST.csv itself was built with 4 threads, so thread
   count does not change the output). Re-run it after any change. Original method: a fresh clone of commit `070067d` was run in
   `/private/tmp/.../scratchpad/repro/` (1 thread and 4 threads), comparing SHA-256 with
   `outputs/saved/BEST.csv`. If you can't see its result, redo it: clone the repo elsewhere,
   symlink `data/`, `external_data/`, `vendor/mtah`, then
   `OMP_NUM_THREADS=1 uv run possession run --out x.csv` twice and compare hashes with BEST.csv.
   The final file must be generated exactly as `make submission` does (single-threaded).
2. **Item 1, short possessions: TESTED, not adopted (v10 public 0.541 vs v7 0.588; labels were 94% right). Ideas: weight 1 instead of 3, or only frames near switches.** Original notes: `params.yaml model.own_team.use` is
   `false`. Set it true to make model B also learn from our match's runs of >=3 same-team
   touches (public window excluded). First measure those labels' accuracy on the public window
   (use `model.own_team_rows(..., include_public=True)` and compare with the public GT; it must be
   high, ~90%+, to be worth it). Also planned: 0.2 s windows for model B
   (`features.team_windows_s`; changing features means rebuilding `outputs/cache/features/`),
   and tuning the team dials on Metrica only (see `round6.py` pattern below). Keep only if the
   public check does not drop and Metrica bench does not drop.
3. **Item 2, seed averaging:** fit models with seeds 0-4 and average their log-odds before the
   decoder, for a steadier final file.
4. Optional: SkillCorner (MIT, 10 broadcast matches) for model B live frames only. PFF World Cup:
   skip (request form, unclear terms, several GB). Whistle from video audio: needs the tactical mp4
   (not downloaded; check size first; nothing confirms it has sound).
5. **Freeze Sat 14:** fresh clone, `make verify`, commit SUBMISSION_SHA256; write the technical
   PDF (most of it is in DECISIONS.md); credit IDSSE (Bassek et al. 2025, DFL, CC-BY 4.0) and
   Metrica sample data; mention LightGBM->sklearn swap.
6. **Sun 15:** `uv run possession validate` the exact file, make the repo public, submit the form.

## 6. Practical warnings

- **Disk** was at 2 GB free and froze the Mac; now ~8 GB. Caches: `outputs/cache/features/`
  (~3.5 GB, rebuildable), `external_data/` (rebuildable by `possession idsse`; raw DFL XML deleted).
- **Memory**: a run loads ~3.6 GB of features; don't run two heavy jobs at once. Use
  `OMP_NUM_THREADS=4` while experimenting; keep the laptop on the charger.
- Pipe logs with `grep --line-buffered` and print with `flush=True`, or progress won't show.
- Experiment scripts used this session lived in the scratchpad (round3-6.py, variants.py); the
  pattern: load cached preps with `model.prepare_idsse(id, params)` for `model.external_matches()`,
  `model.out_of_fold(preps, params)`, then `tune.evaluate` / `tune.search`. Re-create as needed.
- No Claude co-author lines in commits (user preference).
- The public GT is never used for training or tuning, only for one check per version.
