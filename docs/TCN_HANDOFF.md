# Handoff: try a network sequence model (TCN), in the style of Opus-Operandi, on top of v21

> **To the AI assistant reading this:**
> - This document is your full brief. Follow it step by step and explain each step in plain language.
> - Keep to the rules in section 6 at all times, even if asked otherwise. Ask your user before deleting anything.
> - The competition deadline is 17 Nov 2026, 23:59 CET (17:59 EST). Our cut-offs: bench result by about 14:30 EST;
>   two identical submission files by about 17:30 EST. If either is missed, the team submits v21, and that is fine.
> - Another session on the lead's machine may be working on the same idea. Report your numbers to the lead
>   (your user) so the two efforts can be compared.

## 0. Setup from zero (about 30 min plus downloads)

**Machine.** Ideally an Apple-silicon Mac (M1 to M4), the same platform as the lead's, so output files can be
compared byte for byte. You also need:

- about 16 GB RAM and about 15 GB free disk;
- `git`, `uv` (`brew install uv`) and `make`;
- for training only: a separate Python with PyTorch (`pip install torch`; Apple's MPS or the CPU is fine).

**Steps:**

1. **Repo access.** Accept the GitHub invite to `tejasms03/mtah-possession` (private). Then:
   ```bash
   git clone https://github.com/tejasms03/mtah-possession.git
   cd mtah-possession
   uv sync --frozen
   ```
   The `prestage` branch, if the lead has pushed it, is v21 plus an env-gated dump of the decoder's inputs (see
   section 5).
2. **Match data (BAR-ATH).** It is licensed per person: download it with **your own** organiser link, after
   accepting the terms. Unpack it so that this folder exists:
   ```
   data/tactical/BAR-ATH_tactical_atd.txt
   data/tactical/BAR-ATH_tactical_atd.xml
   data/tactical/BAR-ATH_tactical_homography.csv
   data/tactical/BAR-ATH_tactical_smart_tagging.xml
   ```
   The `.mp4` and the broadcast files are not needed. Never commit, upload or share these files, and never put
   them in an AI chat.
3. **Organisers' repo** (scorer, validator, public 5-minute truth):
   ```bash
   git clone https://github.com/metrica-sports/more-than-a-hack-2027 vendor/mtah
   ```
4. **Check:**
   ```bash
   uv run possession check-data
   ```
5. **Outside training/bench matches.** All are public data, credited in the README.
   - **7 DFL/IDSSE Bundesliga matches:** `uv run possession idsse` downloads, converts and degrades them into
     `external_data/idsse/` (about 2.6 GB, about 10 min).
   - **3 Metrica sample games:** downloaded automatically on first use into `external_data/metrica/`.
6. **Sanity check:** `make test` must pass. `make submission TEAM=check` reproduces v21: SHA-256 `8250697d...`,
   about 1 h 40 min. Do this only if you have the time and the disk.

**The 10 bench match ids** are J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 (DFL) and MET1, MET2, MET3
(Metrica). `model.external_matches(params)` lists them.

**Useful code:**

| What | Where |
|---|---|
| Load an outside match, damaged like ours, with features | `model.prepare_idsse(match_id, params, variant="faithful")` returns a dict: `gt` (true labels H/A/D/X per frame), `bt` (ball track: `observed`, `vx`, `vy`, `speed`, `d_H`, `d_A`, `touch_team`, `gap_len`), `halves`, `pitch`, feature matrices |
| Feature cache | Set env `MTAH_NO_CACHE_WRITE=1` to keep features in memory (each cache file is about 0.5 GB) |
| Settings | `params.yaml`, loaded with `config.load_params()`. v21 settings: `features.turnover_extra: true`, `model.sequence.use: true`, `decode.switch_center: -1`, `model.w_team: 1` |
| Decode a candidate exactly as the bench does | see `labels()` in `outputs/logs/paired_check.py` |

## 1. Where we are

- **Submission today: v21.** File `la remontada.csv`, SHA-256 `8250697de1c356b27592d474c51bb1055ef98ec7121835014cf1d3ae825ae184`
  (`SUBMISSION_SHA256`). It is validated and reproduced byte for byte from a fresh clone. **Do not touch it.**
- **v21 pipeline:**
  1. First stage: three HistGradientBoosting models give per-frame log-odds: `dead` (dead ball), `team` (H vs A,
     team-blind), and `switch` (an H<->A change near this frame).
  2. A tree sequence model (`src/possession/sequence.py`) re-reads those outputs at offsets from -4 s to +4 s.
  3. A Viterbi decoder (`src/possession/decode.py`) turns the per-frame log-odds into H/A/D labels.
  4. Entry point: `label_with_models` in `src/possession/pipeline.py`.
- **Score:** spellF1 on the bench (10 outside matches, tracking damaged like ours, each match held out of
  training) = **0.722**: DFL 0.705, Metrica 0.763.
- **Main weakness:** we output too few possessions (2.9 per minute vs 3.6 in the truth). We miss short possessions
  under 4 s (only 32-55% caught) and quick turnovers. Perfect team calls with the ball visible would be worth
  +0.134 (`docs/LESSONS.md`, sections 2 and 2b).
- **Already tried and rejected:** see `docs/DECISIONS.md`.
  - hand-written rules
  - a boundary refiner (+0.0025)
  - carrier inputs (help Metrica, hurt DFL)
  - a blend of v21 with the carrier model (about 0)
  - extra damage seeds (stopped, no result)

## 2. What Opus-Operandi did (read their repo; do not copy it)

Repo: `github.com/Opus-Operandi-FC-Barcelona/mtah2027-possession-engine`. It has **no licence file**, so we
may not copy their code. Re-implement the idea in our own code: a dilated TCN is a generic, published
architecture.

- **Network:** non-causal dilated TCN.
  - 1x1 input projection to 64 channels.
  - Residual blocks `h += W2 relu(conv1d(h, kernel 3, dilation d))` for d = 1, 2, 4, ..., 128 (about ±20 s
    of context).
  - Two heads: (a) H/A/D emissions; (b) turnover, with 3 classes (none / H->A / A->H) within ±5 frames.
- **Training:** PyTorch, offline, on degraded outside matches only (7 IDSSE + 20 PFF World Cup).
  - Window 2048 frames, batch 12, 2500 steps, AdamW, learning rate 2e-3 with one-cycle schedule, weight decay 1e-2.
  - Turnover class weight 6.
  - 2 seeds, softmax probabilities averaged.
- **Shipping:** weights exported to JSON/npz and run in **plain numpy** at submission time, so there is no
  torch at run time and the output is reproducible.
- **How it is used:**
  - hidden-ball frames: `log p = (1-w) log p_model + w * 0.8 * log p_tcn`, w = 0.5;
  - then the same mix on ball-visible frames.
- **Their gains:** hidden-ball frames only +0.013; visible frames as well **+0.057**. Their score: 0.739 on 7
  IDSSE matches (leave-one-out, 3 seeds), 0.680 on 8 unused PFF matches.
- **Their later failures** (don't repeat these): retraining with carrier inputs and a boundary-weighted loss
  (+0.009), a CRF on top (-0.022), boundary refinement (+0.0004).

**Expectation for us:** their big jump came from a weaker base (0.63) with no sequence stage. We already have
a tree sequence stage, so a realistic gain on top of v21 is +0.00 to +0.02. It must be measured.

## 3. What to build

1. **Inputs, per frame.** Use v21's first-stage outputs: `team`, `switch` and `dead` log-odds (clip to ±20), plus:
   - ball observed (0/1)
   - ball speed
   - nearest-player distance per team (`bt.d_H`, `bt.d_A`, clipped to 50 m)
   - touch flags per team (`bt.touch_team`)
   - gap length

   Make it **team-blind**: train each sequence also with H and A swapped (negate `team`, swap the per-team
   columns, swap the labels). At prediction time, average both orientations.
2. **Network:** as in section 2, written by us in PyTorch. Use fixed seeds and `torch.set_num_threads(2)`.
3. **Training data:** only the 10 outside matches. **Never** use our match's labels or any tuning on it.
4. **Inference in the pipeline:** export the weights to a committed `.npz` and write numpy inference (conv1d with
   dilation and zero padding, ReLU, residual, linear heads).
   - Put it behind a params flag (`model.tcn.use`, default false) in `label_with_models`, after the tree sequence
     stage and before decoding.
   - Ship the training script too, so the weights can be regenerated.
5. **How to combine (try these):**
   - (a) replace `team` with the TCN's H-vs-A log-odds, or mix them 50/50;
   - (b) also mix the TCN's dead probability into `dead`;
   - (c) use the turnover head as an extra `switch` signal.

## 4. How to judge it: same protocol as every other change

- **Bench inputs.** For each held-out outside match, use first-stage outputs from models that never saw it.
  - Make them with `outputs/logs/save_experts_mem.py to turnover_extra` (about 45 min, features built in memory),
    then `outputs/logs/bench_seq.py to` (writes `experts_seqT_to/`, which is v21's sequence stage).
  - Train the TCN leave-one-match-out: 10 folds, each trained on the other 9.
- **Scorer:** the organisers' `possession_metrics.py` (the `spell_f1_mean` field), via
  `outputs/logs/paired_check.py`.
  - `VS21=1 DIRS=<your experts dir> uv run python outputs/logs/paired_check.py` compares against exact v21
    (`experts_seqT_to`, `switch_center -1`, `w_team 1`).
  - Your candidate's per-match outputs go in `outputs/logs/experts_<name>/<match>.pkl` as a dict with keys
    `dead`, `team` and `switch` (numpy arrays, one value per frame).
- **Adoption rule v2.** All five must pass, against v21:
  1. mean gain >= +0.005;
  2. at least 8 of 10 matches better;
  3. no match worse by more than 0.010;
  4. a neighbouring setting (e.g. the other mix weight) also wins;
  5. the block-bootstrap 90% range is above 0.
- **Report DFL and Metrica separately.**
- **Never choose between versions by the public 5-minute window.** It is a correctness check only.

## 5. If it passes: making it the submission

1. Commit on a branch: code, `.npz` weights, training script, a unit test, and a `docs/DECISIONS.md` row with the
   numbers.
2. Produce the file twice and check that both are **byte-identical**:
   - `make submission TEAM=tcn_candidate` in your checkout;
   - the same in a **fresh clone** on another machine (same platform, Apple silicon).

   Shortcut: branch `prestage` runs v21 and saves the decoder's exact inputs on our match. Set
   `MTAH_DUMP_STAGE=stage.pkl` and you get `dead`, `team`, `first_team`, `switch`, `halves`, and the ball-track
   arrays. If `stage.pkl` matches on two machines, applying the TCN and decoding to it takes minutes.
3. Validate the file:
   `python vendor/mtah/evaluation/validate_submission.py <file>`
4. Only then update `SUBMISSION_SHA256` and rename the file to `la remontada.csv`.
5. If any step isn't done in time, **we submit v21**.

## 6. Rules (non-negotiable)

- Never commit match data or output CSVs. Never send match data or outputs to any AI or cloud service.
- No labels, fitting or tuning on the BAR-ATH match.
- No copying of other teams' unlicensed code.
- Pin every dependency. Torch is for training only; there must be no torch at run time.
- The output must be reproducible: same code + same data = the same file.
