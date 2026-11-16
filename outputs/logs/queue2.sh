#!/bin/zsh
cd "$(dirname $0)/../.."
L=outputs/logs
export OMP_NUM_THREADS=4
say() { echo "$(date +%H:%M) $*" >> $L/queue2.log; }
say "B start"
uv run python $L/save_experts.py mot features.motion_extra=1 --tmpcache > $L/q_B_save.log 2>&1 && \
  EXP=mot uv run python $L/decode_sweep.py > $L/q_B.log 2>&1 || say "B FAILED: $(grep -m1 Error $L/q_B_save.log)"
rm -f outputs/cache/features/*_mot_xt.pkl; say "B end"
say "D start"
uv run python $L/save_experts.py gho features.ghost_extra=1 --tmpcache > $L/q_D_save.log 2>&1 && \
  EXP=gho uv run python $L/decode_sweep.py > $L/q_D.log 2>&1 || say "D FAILED: $(grep -m1 Error $L/q_D_save.log)"
rm -f outputs/cache/features/*_gho_xt.pkl; say "D end"; say "ALL DONE"
