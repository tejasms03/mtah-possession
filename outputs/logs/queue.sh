#!/bin/zsh
# Runs every missing-ball idea one after another, cheapest first. Each step's table goes to its own
# log; queue.log gets one summary line per step. All deltas are against v17 on the 10-match hold-out.
cd "$(dirname $0)/../.."
L=outputs/logs
export OMP_NUM_THREADS=4
say() { echo "$(date +%H:%M) $*" | tee -a $L/queue.log; }
until grep -qE "^gap |Traceback" $L/gap_bench.log 2>/dev/null; do sleep 30; done
sleep 20
say "start (gap bench finished)"

say "F: missing-ball-only dead bias (decoder only)"
uv run python $L/decode_sweep.py 'decode.dead_bias_miss=-1,-0.5,-0.25,0.25' > $L/q_F.log 2>&1
say "E: pool dead evidence over each missing-ball run (decoder only)"
uv run python $L/decode_sweep.py 'decode.gap_pool=0.25,0.5,0.75,1' > $L/q_E.log 2>&1

say "C: missing-ball frames weighted x3 in model A training"
uv run python $L/save_experts.py mw3 model.miss_weight=3 > $L/q_C_save.log 2>&1 && \
  EXP=mw3 uv run python $L/decode_sweep.py > $L/q_C.log 2>&1
say "C done"

say "B: motion features"
uv run python $L/save_experts.py mot features.motion_extra=1 --tmpcache > $L/q_B_save.log 2>&1 && \
  EXP=mot uv run python $L/decode_sweep.py > $L/q_B.log 2>&1
rm -f outputs/cache/features/*_mot_xt.pkl; say "B done"

say "D: ghost ball (train ghost models, then features)"
uv run python $L/train_ghost.py > $L/q_D_ghost.log 2>&1 && \
  uv run python $L/save_experts.py gho features.ghost_extra=1 --tmpcache > $L/q_D_save.log 2>&1 && \
  EXP=gho uv run python $L/decode_sweep.py > $L/q_D.log 2>&1
rm -f outputs/cache/features/*_gho_xt.pkl; say "D done"

say "C neighbours: weight x2 and x5"
for w in 2 5; do
  uv run python $L/save_experts.py mw$w model.miss_weight=$w > $L/q_C${w}_save.log 2>&1 && \
    EXP=mw$w uv run python $L/decode_sweep.py > $L/q_C$w.log 2>&1
done
say "ALL DONE"; df -h . | tail -1 | tee -a $L/queue.log
