#!/bin/bash
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=dawg-placesmoke
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/placesmoke-%j.out
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg
python scripts/28_placement_smoke.py --eps "${EPS:-16}"
