#!/bin/bash
# Append a compact status line to results/STATUS.log every 5 min. Run via nohup.
cd /gscratch/raivn/andy132/dawg
while true; do
  d=$(ls -d data/syntheticqa_deep/*/page*/ 2>/dev/null|wc -l)
  ds=$(ls -d data/syntheticqa_deep/*/ 2>/dev/null|wc -l)
  al=$(pgrep -f syntheticqa_deep >/dev/null && echo ALIVE || echo done)
  j=$(squeue -u andy132 -h 2>/dev/null|wc -l)
  l1=$(cat results/l1_dataset/shard*.jsonl results/l1_dataset_new/shard*.jsonl 2>/dev/null|wc -l)
  echo "$(date '+%m-%d %H:%M')  deep=${ds}sites/${d}pages[$al]  gpu_jobs=$j  L1_pairs=$l1" >> results/STATUS.log
  sleep 300
done
