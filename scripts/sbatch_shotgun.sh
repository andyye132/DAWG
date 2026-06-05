#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=dawg-shot
#SBATCH --array=0-4
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/overnight_run_01/slurm-logs/shot-%A-%a.out
#SBATCH --error=/gscratch/raivn/andy132/dawg/results/overnight_run_01/slurm-logs/shot-%A-%a.err

source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb

SITES=(seattle_times fidelity microsoft amazon sweet_alchemy)
SITE=${SITES[$SLURM_ARRAY_TASK_ID]}

echo "[shotgun] array task $SLURM_ARRAY_TASK_ID -> site=$SITE"
python /gscratch/raivn/andy132/dawg/scripts/05_run_shotgun.py --site $SITE
