#!/bin/bash
# Autonomous overnight orchestrator: wait for the fetch to finish, partition
# sites into L1/L2/L3, then launch wave-2 L1 data-gen over L1's NEW sites.
# Writes a separate output dir so it never collides with the still-running
# wave-1 shards. Logs to results/wave2_orchestrator.log.
cd /gscratch/raivn/andy132/dawg
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src

echo "[orch] $(date) waiting for fetch to finish..."
while pgrep -f 10_fetch_syntheticqa >/dev/null; do sleep 120; done
echo "[orch] $(date) fetch finished. sites on disk: $(ls -d data/syntheticqa_full/*/ | wc -l)"
tail -3 results/fetch_wave2.log

echo "[orch] partitioning sites L1/L2/L3 ..."
python scripts/orchestrate_wave2.py

echo "[orch] launching wave-2 L1 data-gen (L1 new sites -> results/l1_dataset_new)"
sbatch --export=ALL,OUT=/gscratch/raivn/andy132/dawg/results/l1_dataset_new,PAGES_FILE=/gscratch/raivn/andy132/dawg/data/pagelist_l1_wave2.txt \
       --array=0-7 scripts/sbatch_l1data.sh
echo "[orch] $(date) done. wave-2 submitted."
