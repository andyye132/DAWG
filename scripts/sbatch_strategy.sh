#!/bin/bash
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=dawg-strat
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/strat-%A_%a.out
#SBATCH --requeue
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg
python scripts/29_strategy_compare.py --pages "${PAGES:-rand:60:7}" --eps 8,16 \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-6}"
