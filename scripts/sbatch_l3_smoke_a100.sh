#!/bin/bash
#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=2:00:00
#SBATCH --job-name=dawg-l3smk100
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/l3smoke-%j.out
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
export HF_HOME=/gscratch/raivn/andy132/.cache/huggingface
cd /gscratch/raivn/andy132/dawg
python -u scripts/38_l3_smoke.py --n "${N:-2}" --eps "${EPS:-24}" --iters "${ITERS:-120}" \
  --restarts "${RESTARTS:-2}" --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-12}"
