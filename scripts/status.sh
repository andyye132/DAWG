#!/bin/bash
# On-demand DAWG pipeline status snapshot.
cd /gscratch/raivn/andy132/dawg
echo "===== DAWG STATUS  $(date) ====="
echo "[downloads]"
df=$(pgrep -f syntheticqa_deep >/dev/null && echo ALIVE || echo done/stopped)
echo "  syntheticqa_deep: $(ls -d data/syntheticqa_deep/*/ 2>/dev/null|wc -l) sites / $(ls -d data/syntheticqa_deep/*/page*/ 2>/dev/null|wc -l) pages  [$df]  $(tail -1 results/fetch_deep.log 2>/dev/null)"
echo "  syntheticqa_full (canonical): $(ls -d data/syntheticqa_full/*/ 2>/dev/null|wc -l) sites / $(ls -d data/syntheticqa_full/*/page*/ 2>/dev/null|wc -l) pages"
echo "[gpu jobs]"
squeue -u andy132 -o "  %.12i %.14j %.10P %.3t %.11M %R" -h 2>/dev/null || echo "  (none)"
echo "[L1 dataset]"
r1=$(cat results/l1_dataset/shard*.jsonl 2>/dev/null|wc -l); r2=$(cat results/l1_dataset_new/shard*.jsonl 2>/dev/null|wc -l)
b1=$(cat results/l1_dataset/shard*.jsonl results/l1_dataset_new/shard*.jsonl 2>/dev/null|grep -c '"success": true')
echo "  pairs: wave1=$r1 wave2=$r2 total=$((r1+r2))  broke=$b1"
echo "  adv screenshots: $(ls results/l1_dataset/*/*/adv_q*.png results/l1_dataset_new/*/*/adv_q*.png 2>/dev/null|wc -l)"
