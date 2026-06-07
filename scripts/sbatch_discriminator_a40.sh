#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --job-name=dawg-disc
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/disc-%j.out
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
export HF_HOME=/gscratch/raivn/andy132/.cache/huggingface
cd /gscratch/raivn/andy132/dawg
python -u scripts/40_train_discriminator.py --epochs "${EPOCHS:-14}" --img "${IMG:-512}" --bs "${BS:-24}"
