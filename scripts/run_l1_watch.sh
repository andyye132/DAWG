#!/bin/bash
# Detached login-node watcher: wait for the 3 L1 momentum jobs, write final stats.
cd /gscratch/raivn/andy132/dawg
while [ -n "$(squeue -j 35900136,35900137,35900138 -h 2>/dev/null)" ]; do sleep 180; done
{
  echo "L1 momentum jobs done $(date)"
  tot=0; bro=0
  for d in l1_single10 l1_two5 l1_three33; do
    r=$(cat results/$d/shard*.jsonl 2>/dev/null | wc -l)
    b=$(cat results/$d/shard*.jsonl 2>/dev/null | grep -c '"success": true')
    pct=$(( r>0 ? b*100/r : 0 ))
    echo "  $d: $r pairs, $b broke (${pct}%)"
    tot=$((tot+r)); bro=$((bro+b))
  done
  pct=$(( tot>0 ? bro*100/tot : 0 ))
  echo "  L1 TOTAL: $tot pairs, $bro successful (${pct}%)"
} > results/L1_FINAL.txt 2>&1
