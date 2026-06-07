#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=dawg-disceval
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/disceval-%j.out
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
export HF_HOME=/gscratch/raivn/andy132/.cache/huggingface
cd /gscratch/raivn/andy132/dawg
python -u scripts/46_disc_eval_dump.py
