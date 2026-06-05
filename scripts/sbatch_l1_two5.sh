#!/bin/bash
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=dawg-l1two5
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/l1two5-%A_%a.out
#SBATCH --requeue
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1 PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg
python scripts/26_generate_l1_data.py --pages-file data/chunk_l1_b_two5.txt --out results/l1_two5 \
  --qa-per-page 1 --npatch 2 --placement topk --total-area 10 --eps-min 12 --eps-max 24 --iters 50 --optim momentum \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards 6
