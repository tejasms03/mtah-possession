# Submitting v22 (TCN candidate) — checklist and text for the technical document

v22 = v21 + the network sequence model (`src/possession/tcn.py`, `model.tcn.use: true`, decoder
blend weight 0.5). Bench on the 10 outside matches vs exact v21 (0.7220): **0.739 (+0.017)**, 10/10
matches better, DFL +0.017, Metrica +0.017. Details: DECISIONS.md, row "Network sequence model (TCN)".

## 1. Checks before anything is replaced (done on branch `tcn-candidate`)

- [x] `make submission TEAM=tcn_candidate` in the worktree and in a fresh `git clone` (raw data
      symlinked, `uv sync --frozen`, `MTAH_NO_CACHE_WRITE=1`) give the **same SHA-256**:
      `b886ae5807d19e3bea7d79f949655e890538b1d891bc2eb5b55bfe96298c7158`
- [x] `python vendor/mtah/evaluation/validate_submission.py <file>` passes on both
- [x] public 5-minute window through the organisers' scorer: correctness check only (not tuning): spellF1 0.503 (v21 0.463)
- [x] frame-by-frame comparison with v21's `la remontada.csv`: 5.4% of graded frames differ; possessions/min 2.58 -> 2.79
- [x] `make test` passes; no match data or output CSV tracked (`git ls-files`)

## 2. Switching the submission to v22 (main session, with the user's OK — these overwrite v21)

1. Merge `tcn-candidate` into `main` (main has newer doc commits on origin: merge, don't rebase).
2. Copy the verified file to `outputs/la_remontada.csv`; keep v21's file as `outputs/saved/v21_la_remontada.csv`
   before replacing `la remontada.csv` and `outputs/saved/BEST.csv`.
3. Write the new hash into `SUBMISSION_SHA256`; README and the technical document quote it.
4. `make verify` on main must print `byte-identical`.
5. **Make the GitHub repo public** (`tejasms03/mtah-possession` is private now; the organisers require
   a public URL). Check once more that no match data is in its history.
6. Push main.

## 3. The form (the user does this)

| field | what |
|---|---|
| Team Name | La Remontada |
| Possession file | `la remontada.csv` (the v22 file, ~1.9 MB) — validate this exact file right before upload |
| Public code repository URL | https://github.com/tejasms03/mtah-possession (after making it public) |
| Short technical document | `la remontada.md` (or exported `la remontada.pdf`) |

One response per team; it can be edited until the deadline; the last response counts.

## 4. Text to add to the technical document (`docs/la remontada.md`)

**Section 1, new step between 2 and 3:**

> **A network sequence model (TCN)** re-reads the first-stage outputs (team, change-detector and
> dead log-odds) together with ball visibility, ball speed, each team's distance to the ball and
> touches, over about ±20 s. It is a small non-causal dilated temporal convolution network (64 channels,
> kernel 3, dilations 1 to 256, residual blocks; about 150k weights) with an H/A/D head and a turnover
> head, team-blind (run on the inputs and with H and A swapped, answers averaged). Its H/A/D
> log-probabilities are blended 50/50 into the decoder's per-frame evidence. Trained offline in PyTorch
> on the 10 outside matches only (`tools/train_tcn.py`); the weights are committed (`models/tcn.npz`)
> and the pipeline runs them with plain numpy, so the submission needs no PyTorch and is byte-identical.

**Section 5, results row:**

> v22 = v21 + TCN blend: 10-match hold-out 0.739 vs 0.722 (+0.017), better on 10/10 matches (DFL +0.017,
> Metrica +0.017; 90% block bootstrap +0.013 to +0.023; a second training seed gives the same).
> The gain is almost all live vs dead ball (live/dead mix-ups where the ball is seen fall from 2.3% to
> 0.6% of frames); wrong-team frames and quick turnovers are unchanged. Using the turnover head as the
> change signal, or blending only where the ball is hidden, added nothing and is off.

**Section 7, disclosure:**

> The TCN design follows a generic published architecture (dilated residual temporal convolutions).
> We were prompted to try it by another team's public write-up (Opus-Operandi); we did not use their
> code. All TCN code is ours.

**Section 8, reproduction:** unchanged command; add "retraining the TCN weights (optional) needs
PyTorch 2.x: see README".
