# vendor/ (organisers' repository)

Clone the organisers' challenge repository here as `vendor/mtah`. It holds the official scorer
(`evaluation/`), the public ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`),
`games.py` and the starter notebooks.

```bash
git clone <organisers-repo-url> vendor/mtah
```

It is gitignored on purpose: we use their scorer **unchanged**, and it keeps their files out of our
public repository. Our code reads the public GT from
`vendor/mtah/evaluation/gt/BAR-ATH_possession_gt_public.csv`.
