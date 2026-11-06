# data/ (not committed)

The match assets are licensed for the competition only. Everything in this folder except this
README is gitignored. **Do not commit, upload or share these files**; each team member downloads
them with their own link after accepting the terms.

Unpack the bundle here, keeping the names it arrives with:

```
data/
├─ tactical/
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml              tracking metadata
│  ├─ BAR-ATH_tactical_atd.txt              tracking, players and ball
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

To keep the data somewhere else, set `MTAH_DATA_DIR=/path/to/bundle`. Check with:

```bash
uv run possession check-data
```
