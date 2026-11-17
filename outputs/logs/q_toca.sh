#!/bin/zsh
S=(uv run python outputs/logs/decode_sweep.py 'model.w_team=1,1.5' 'decode.switch_center=-1,-0.5')
echo "### carrier + turnover retrain $(date)"; MTAH_NO_CACHE_WRITE=1 uv run python outputs/logs/save_experts_mem.py toca turnover_extra carrier_extra 2>&1 | grep -v "Warn\|nanmean\|nanmax\|nanmin\|^\[\|^  "
EXP=toca BASE=mot $S[@] 2>&1 | grep -v "Warn\|nanmean\|^ \|^\[" | sed 's/^/toca /'
echo "### sequence $(date)"; uv run python outputs/logs/bench_seq.py toca 2>&1 | grep -v "Warn\|nanmean"
EXP=seqT_toca BASE=mot $S[@] 2>&1 | grep -v "Warn\|nanmean\|^ \|^\[" | sed 's/^/seqT_toca /'
echo "### done $(date)"
