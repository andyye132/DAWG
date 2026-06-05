#!/bin/bash
cd /gscratch/raivn/andy132/dawg
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export HF_HOME=/gscratch/raivn/andy132/.cache/huggingface
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src
export PYTHONUNBUFFERED=1
exec python -u scripts/10_fetch_syntheticqa.py \
  --out data/syntheticqa_deep --n-sites 500 --pages-per-site 50 --max-scan 2000000
