#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=dawg-l1data
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/l1data-%A_%a.out
#SBATCH --requeue
#
# L1 adversarial dataset generation. FEW shards, each boots MolmoWeb ONCE and
# grinds many pages (page i -> shard i%num_shards). Resumable (per-row JSONL).
#   sbatch --array=0-7 scripts/sbatch_l1data.sh            # 8 shards, full
#   QA=1 ITERS=10 OUT=.../results/l1_smoke NUM_SHARDS=300 sbatch --array=0-0 scripts/sbatch_l1data.sh  # smoke

source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg

python scripts/26_generate_l1_data.py \
  --root "${ROOT:-/gscratch/raivn/andy132/dawg/data/syntheticqa_full}" \
  --out  "${OUT:-/gscratch/raivn/andy132/dawg/results/l1_dataset}" \
  --pages-file "${PAGES_FILE:-/gscratch/raivn/andy132/dawg/data/pagelist_l1_wave1.txt}" \
  --qa-per-page "${QA:-5}" --npatch 3 --total-area 10 --eps-min 12 --eps-max 24 \
  --iters "${ITERS:-50}" \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-8}"
