#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --job-name=dawg-batch
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/discriminator_v1/batch-%A_%a.out
#SBATCH --requeue

source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb
cd /gscratch/raivn/andy132/dawg

# Page tree root, output dir, and number of shards (override via --export env vars;
# NUM_SHARDS must match the --array range in the sbatch call).
ROOT="${DAWG_ROOT_OVERRIDE:-/gscratch/raivn/andy132/dawg/data}"
OUT="${OUT_OVERRIDE:-/gscratch/raivn/andy132/dawg/results/discriminator_v1}"
python scripts/18_batch_attack.py --root "$ROOT" --out "$OUT" \
  --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-4}" \
  --eps-min "${EPS_MIN:-4}" --eps-max "${EPS_MAX:-16}" \
  --area-min "${AREA_MIN:-0.005}" --area-max "${AREA_MAX:-0.03}" \
  --iters "${ITERS:-40}"
