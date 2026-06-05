#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=dawg-scale
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/frontier_logs/scale-%A_%a.out
#SBATCH --requeue
#
# Scale / placement sweep (scripts/24). Each array index = ONE page (rand:N:0,
# num-shards = N), runs 3 arms x 7 eps x RESTARTS restarts. ~2.3 min/cell, so
# ~2.5h/page at RESTARTS=3.  Params are passed as explicit CLI args (NOT relying
# on --export of EPS/NPATCH, which silently failed for the multipatch job).
#
# Submit the full sweep (12 pages, 1 page/shard, R=1):
#   sbatch --array=0-11 scripts/sbatch_scale.sh
# Fast 1-page smoke to a throwaway dir (scalar overrides only -- never put a
# comma-list like EPS through --export, it splits on the commas):
#   sbatch --export=ALL,ITERS=10,OUT=/gscratch/raivn/andy132/dawg/results/scale_smoke --array=0-0 scripts/sbatch_scale.sh

source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg

python scripts/24_scale_sweep.py \
  --root "${ROOT:-/gscratch/raivn/andy132/dawg/data/syntheticqa_full}" \
  --out  "${OUT:-/gscratch/raivn/andy132/dawg/results/scale_v1}" \
  --pages "${PAGES:-rand:12:0}" \
  --total-area "${TOTAL_AREA:-10}" --npatch "${NPATCH:-3}" \
  --eps "${EPS:-2,4,8,12,16,24,32}" --iters "${ITERS:-50}" --restarts "${RESTARTS:-1}" \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-12}"
