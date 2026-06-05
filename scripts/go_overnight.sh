#!/bin/bash
# Overnight DAWG pipeline orchestrator. Runs once ssh to hyak is back.
#
# Sequence (idempotent — skips completed stages):
#   1. sync local code to cluster (eval, attacks, scripts, mock_sites)
#   2. install sentence-transformers + pre-download MPNet (skipped if already installed)
#   3. submit Phase C smoke test via srun (blocking — waits up to 30 min)
#   4. if smoke passed → submit Phase B sbatch (capture ground truth)
#   5. wait for Phase B completion
#   6. submit Phase D sbatch array (shotgun attacks)
#   7. report jobids; we monitor separately
#
# Hard rule reminder: NEVER rm on cluster. This script does not delete anything.
#
# Usage:
#   bash scripts/go_overnight.sh
set -uo pipefail

LOCAL_ROOT="/Users/andy132/projects/research/raivn/dawg"
CLUSTER_ROOT="/gscratch/raivn/andy132/dawg"
HYAK="hyak"
ENV_PATH="/mmfs1/gscratch/raivn/andy132/conda_envs/dawg"
RESULTS="$CLUSTER_ROOT/results/overnight_run_01"

echo "=== [1/6] verify ssh ==="
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$HYAK" 'echo connected' >/dev/null 2>&1; then
  echo "FAIL: ssh hyak refused. Re-auth and retry."
  exit 1
fi
echo "ok"

echo
echo "=== [2/6] sync local code → cluster ==="
scp -q -r "$LOCAL_ROOT/src/dawg/eval" "$HYAK:$CLUSTER_ROOT/src/dawg/"
scp -q -r "$LOCAL_ROOT/src/dawg/attacks" "$HYAK:$CLUSTER_ROOT/src/dawg/"
# Re-sync mock_sites (bbox normalization changed prompts.json files)
for site in seattle_times fidelity microsoft amazon sweet_alchemy; do
  scp -q "$LOCAL_ROOT/src/dawg/mock_sites/sites/$site/prompts.json" \
    "$HYAK:$CLUSTER_ROOT/src/dawg/mock_sites/sites/$site/prompts.json"
done
scp -q "$LOCAL_ROOT/scripts/04_capture_ground_truth.py" \
       "$LOCAL_ROOT/scripts/05_run_shotgun.py" \
       "$LOCAL_ROOT/scripts/06_smoke_attack.py" \
       "$LOCAL_ROOT/scripts/sbatch_capture_gt.sh" \
       "$LOCAL_ROOT/scripts/sbatch_shotgun.sh" \
       "$HYAK:$CLUSTER_ROOT/scripts/"
ssh "$HYAK" "chmod +x $CLUSTER_ROOT/scripts/sbatch_*.sh"
echo "sync ok"

echo
echo "=== [3/6] install sentence-transformers + pre-download MPNet ==="
ssh "$HYAK" "bash -lc '
  source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
  conda activate $ENV_PATH
  python -c \"from sentence_transformers import SentenceTransformer\" 2>/dev/null || pip install \"sentence-transformers<5\"
  python -c \"from sentence_transformers import SentenceTransformer; m=SentenceTransformer(\\\"all-mpnet-base-v2\\\"); print(\\\"MPNet ok dim\\\", m.get_sentence_embedding_dimension())\"
'"
if [ $? -ne 0 ]; then echo "FAIL: sentence-transformers/MPNet setup."; exit 2; fi

echo
echo "=== [4/6] Phase C smoke test (srun, ~10 min) ==="
ssh "$HYAK" "srun --account=raivn --partition=gpu-a40 --gres=gpu:1 --mem=32G --time=00:30:00 bash -lc '
  source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh
  conda activate $ENV_PATH
  export PYTHONPATH=$CLUSTER_ROOT/src:$CLUSTER_ROOT/external/molmoweb
  python $CLUSTER_ROOT/scripts/06_smoke_attack.py
'"
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then
  echo "FAIL: smoke test rc=$SMOKE_RC. Check $RESULTS/DONE_BLOCKED.md and $RESULTS/smoke/."
  exit 3
fi
echo "smoke ok"

echo
echo "=== [5/6] submit Phase B (capture ground truth) sbatch ==="
GT_JOB=$(ssh "$HYAK" "cd $CLUSTER_ROOT && sbatch --parsable scripts/sbatch_capture_gt.sh")
if [ -z "$GT_JOB" ]; then echo "FAIL: sbatch capture_gt did not return jobid."; exit 4; fi
echo "Phase B submitted: jobid=$GT_JOB"
echo "  (waiting for completion before Phase D — could take ~1 hr)"
while true; do
  STATE=$(ssh "$HYAK" "sacct -j $GT_JOB --format=State --noheader --parsable2" | head -1)
  if [ -z "$STATE" ] || echo "$STATE" | grep -qE "RUNNING|PENDING|CONFIGURING"; then
    sleep 60
    continue
  fi
  echo "Phase B final state: $STATE"
  break
done
if ! echo "$STATE" | grep -q "COMPLETED"; then
  echo "FAIL: Phase B ended in state $STATE. See $RESULTS/slurm-logs/gt-$GT_JOB.{out,err}"
  exit 5
fi

echo
echo "=== [6/6] submit Phase D shotgun array (5 sites in parallel) ==="
SHOT_JOB=$(ssh "$HYAK" "cd $CLUSTER_ROOT && sbatch --parsable scripts/sbatch_shotgun.sh")
if [ -z "$SHOT_JOB" ]; then echo "FAIL: sbatch shotgun did not return jobid."; exit 6; fi
echo "Phase D submitted: jobarrayid=$SHOT_JOB (--array=0-4)"

echo
echo "=== orchestrator done ==="
echo "Monitor with:"
echo "  ssh $HYAK 'squeue -u andy132'"
echo "  ssh $HYAK 'tail -f $RESULTS/slurm-logs/shot-$SHOT_JOB-*.out'"
echo "Results land in $RESULTS/{attacks,summary.csv}"
