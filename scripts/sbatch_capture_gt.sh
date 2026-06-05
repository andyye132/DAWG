#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=dawg-gt
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/overnight_run_01/slurm-logs/gt-%j.out
#SBATCH --error=/gscratch/raivn/andy132/dawg/results/overnight_run_01/slurm-logs/gt-%j.err

source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb

python /gscratch/raivn/andy132/dawg/scripts/04_capture_ground_truth.py
