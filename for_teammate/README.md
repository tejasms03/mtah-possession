# Comparing the two pipelines

Files from the rules-only pipeline built in a separate local repo (`possession`, branch `dev`).

| File | What it is |
|---|---|
| `possession-dev.bundle` | That whole repo as one file: code, config, `DECISIONS.md`. No match data. |
| `export_metrica_clean.py` | Run in this repo: writes the three clean Metrica bench matches and labels to a folder. |
| `ours_on_metrica_fair.py` | Run in the bundled repo: scores its rules pipeline and the organisers' baseline on those matches. |

The submission CSV is not here on purpose: output CSVs are never committed. It is sent directly.

## Like-for-like score on the Metrica bench

```bash
git clone for_teammate/possession-dev.bundle ../possession-ours
cd ../possession-ours && git switch dev && uv sync --frozen
mkdir -p /tmp/metrica_bench
cd ../mtah-possession && uv run python for_teammate/export_metrica_clean.py /tmp/metrica_bench
cd ../possession-ours && MTAH_UPSTREAM=../mtah-possession/vendor/mtah \
  uv run python ../mtah-possession/for_teammate/ours_on_metrica_fair.py /tmp/metrica_bench
```

Expected: rules pipeline mean spellF1 0.578 (macroF1 0.809, dead-ball F1 0.824), baseline 0.383,
on the 1.5 s-merged labels. Compare with v16 on the same three matches (0.810 in DECISIONS.md).

One difference to remove for an exact comparison: this script fragments the Metrica players
with no added position noise, because the real tracking is that smooth (measured jitter 0.002 m
per frame). The bench in this repo adds 0.15 m (`idsse_degrade.pos_noise_m`). With that noise the
rules pipeline's slowdown clue breaks and it scores 0.446. Re-running the whole-match bench with
`pos_noise_m: 0` (feature cache rebuilt) puts both numbers on identical inputs.

## Frame-by-frame comparison

```bash
uv run possession compare /path/to/rules_v3.csv outputs/<v16 file>.csv
```

Known weakness of the rules pipeline: it over-calls dead ball (36-46% of frames on the Metrica
matches against a true 27-34%), so expect most disagreement as "dead only in the rules file".
