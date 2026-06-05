#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=dawg-frontier
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/frontier-%A_%a.out
#SBATCH --requeue

source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg

python scripts/22_frontier.py \
  --root "${ROOT:-/gscratch/raivn/andy132/dawg/data/syntheticqa_full}" \
  --out  "${OUT:-/gscratch/raivn/andy132/dawg/results/frontier_v1}" \
  --pages "${PAGES:-auto:10}" --sizes "${SIZES:-1,2,4,8}" --eps "${EPS:-4,8,12,16,24}" \
  --npatch "${NPATCH:-1}" --iters "${ITERS:-50}" \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-10}"
