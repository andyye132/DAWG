#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=dawg-l3gen
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/l3gen-%A_%a.out
#SBATCH --requeue
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
export HF_HOME=/gscratch/raivn/andy132/.cache/huggingface
cd /gscratch/raivn/andy132/dawg
python -u scripts/41_generate_l3_data.py --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-16}" \
  --n "${N:-200}" --eps "${EPS:-24}" --iters "${ITERS:-80}" --restarts "${RESTARTS:-1}"
