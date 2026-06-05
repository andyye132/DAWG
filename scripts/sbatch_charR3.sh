#!/bin/bash
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --job-name=dawg-charR3
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/charR3-%A_%a.out
#SBATCH --requeue
#
# Characterization run: R=3 best-of-k, 3 arms (single10 / multi_even / multi_random),
# eps 4..24, ~18 unbiased pages. Gives the rigorous single-vs-multi evidence.
# FEW shards, each boots MolmoWeb once. eps baked in (no comma-list via --export).
#   sbatch --array=0-5 scripts/sbatch_charR3.sh

source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg

python scripts/24_scale_sweep.py \
  --root /gscratch/raivn/andy132/dawg/data/syntheticqa_full \
  --out  /gscratch/raivn/andy132/dawg/results/char_r3 \
  --pages "${PAGES:-file:/gscratch/raivn/andy132/dawg/data/pagelist_char18.txt}" --total-area 10 --npatch 3 \
  --eps 4,8,12,16,24 --iters 50 --restarts 3 \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-6}"
