#!/bin/bash
#SBATCH --account=raivn
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=dawg-pgd-sanity
#SBATCH --output=/gscratch/raivn/andy132/dawg/results/pgd_sanity/sanity-%j.out
#SBATCH --error=/gscratch/raivn/andy132/dawg/results/pgd_sanity/sanity-%j.out

source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export PYTHONNOUSERSITE=1
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb

PAGE="${1:-/gscratch/raivn/andy132/dawg/data/syntheticQA/adidas/page000}"
echo "[sbatch] host=$(hostname) page=$PAGE"
python /gscratch/raivn/andy132/dawg/scripts/13_pixel_pgd_sanity.py --page "$PAGE"
echo "[sbatch] exit=$?"
