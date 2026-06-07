#!/bin/bash
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --job-name=dawg-l2smoke
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/l2smoke-%j.out
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME=/gscratch/raivn/andy132/.cache/huggingface
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg
python -u scripts/31_l2_smoke.py --n "${N:-10}" --eps "${EPS:-20}" --iters "${ITERS:-120}" --restarts "${RESTARTS:-3}"
