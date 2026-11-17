#!/bin/zsh
# bench grids for the rule fixes (src/possession/rules.py), on v18 experts with v19 params
S=(uv run python outputs/logs/decode_sweep.py)
export EXP=mot BASE=mot
echo "### 1 restart team (location)"; $S[@] 'decode.rt_s=1,2,3' 'decode.rt_line_m=1,2,3' 'decode.rt_foul=0,1' 2>&1 | grep -v '^\['
$S[@] 'decode.rt_s=2' 'decode.rt_fallback_s=0.5,1' 2>&1 | grep -v '^\['
echo "### 2 restart team with stand-in tags (MET only)"; TAGS=1 $S[@] 'decode.rt_s=2,3' 'decode.rt_tags=0,1' 'decode.rt_foul=0,1' 2>&1 | grep -v '^\['
echo "### 3 nearest at last sighting"; $S[@] 'decode.ln_s=1,2,3' 'decode.ln_m=1,2,3' 'decode.ln_gap_m=0.5,1,2' 2>&1 | grep -v '^\['
echo "### 4 team flip"; $S[@] 'decode.tf_s=0.4,0.8,1.5' 'decode.tf_m=1,1.5,2' 'decode.tf_gap_m=1,2,3' 2>&1 | grep -v '^\['
echo "### 5 dead check"; $S[@] 'decode.dc_on=1' 'decode.dc_miss=0.5,0.8,0.95' 'decode.dc_spd=0,1,1.5,2' 'decode.dc_block=0,0.5,1' 2>&1 | grep -v '^\['
echo "### 6 dead needs a reason"; $S[@] 'decode.nr_s=1,2,4' 'decode.nr_m=2,4,8' 2>&1 | grep -v '^\['
echo "### 7 gap ends"; $S[@] 'decode.gr_mode=1,2' 'decode.gr_min_s=0.5,1,2,4' 'decode.gr_m=1.5,3' 'decode.gr_gap_m=0.5,1.5' 2>&1 | grep -v '^\['
echo "### 8 snap to ball speed"; $S[@] 'decode.fs_v=2,4,6' 'decode.fs_s=0.5,1,2' 'decode.fs_stop=0,1' 2>&1 | grep -v '^\['
echo "### 9 snap to tag (MET only, stand-ins: circular)"; TAGS=1 $S[@] 'decode.ts_off_s=1,2,3,4' 'decode.ts_w_s=2,4' 2>&1 | grep -v '^\['
echo "### existing dials"; $S[@] 'decode.last_touch_s=2,3,4' 2>&1 | grep -v '^\['
$S[@] 'decode.min_possession_frames=5,8,10' 2>&1 | grep -v '^\['
$S[@] 'decode.refine=1' 'decode.snap_s=0.3,0.5' 2>&1 | grep -v '^\['
echo "### done"
