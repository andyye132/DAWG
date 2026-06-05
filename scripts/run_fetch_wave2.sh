#!/bin/bash
# Persistent login-node fetch of MolmoWeb-SyntheticQA sites (wave 2).
# Streams alphabetically, ~5 pages/site, into data/syntheticqa_full (re-fetches
# the existing front sites cheaply, then adds new ones). Stop with:
#   pkill -f 10_fetch_syntheticqa
cd /gscratch/raivn/andy132/dawg
source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg
export HF_HOME=/gscratch/raivn/andy132/.cache/huggingface
export PYTHONPATH=/gscratch/raivn/andy132/dawg/src
export PYTHONUNBUFFERED=1
exec python -u scripts/10_fetch_syntheticqa.py \
  --out data/syntheticqa_full --n-sites 500 --pages-per-site 5 --max-scan 400000
