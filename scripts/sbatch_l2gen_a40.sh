#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=dawg-l2gen
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/l2gen-%A_%a.out
#SBATCH --requeue
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg
python -u scripts/32_generate_l2_data.py --pages-file data/chunk_l2.txt --out results/l2_dataset \
  --eps 20 --iters 100 --restarts 1 --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards 16
