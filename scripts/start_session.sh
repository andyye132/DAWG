#!/usr/bin/env bash
# Spin up the full DAWG playground stack in a single tmux session.
# Expects to be run from a GPU node (after srun). Creates session "dawg" with
# three panes: MolmoWeb server, playground, cloudflared tunnel.
#
# Required env vars:
#   DAWG_PASS         playground HTTP Basic Auth password (no default; you must set it)
#
# Optional env vars:
#   DAWG_USER         playground username (default: dawg)
#   TUNNEL_MODE       "ephemeral" (default) or "named". "named" requires
#                     `cloudflared tunnel create dawg` and DNS route to have
#                     been set up once; serves at https://dawg.andyye.bio.
#
# Usage:
#   export DAWG_PASS=Cheeseball5
#   bash /gscratch/raivn/andy132/dawg/scripts/start_session.sh

set -euo pipefail

SESSION="dawg"
DAWG_USER="${DAWG_USER:-dawg}"
TUNNEL_MODE="${TUNNEL_MODE:-ephemeral}"

if [[ -z "${DAWG_PASS:-}" ]]; then
  echo "ERROR: DAWG_PASS not set. Run: export DAWG_PASS=<a-password>" >&2
  exit 1
fi

if [[ "$(hostname)" == klone-login* ]]; then
  echo "ERROR: you're on a login node. Allocate a GPU node first:" >&2
  echo "  srun --account=raivn --partition=gpu-a40 --gres=gpu:1 --mem=32G --time=8:00:00 --pty bash" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists." >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  echo "Or kill with: tmux kill-session -t $SESSION" >&2
  exit 1
fi

# Common preamble used in every pane
ENV_PREAMBLE='source /mmfs1/home/andy132/miniconda3/etc/profile.d/conda.sh && \
conda activate /mmfs1/gscratch/raivn/andy132/conda_envs/dawg && \
export PYTHONNOUSERSITE=1 && \
export no_proxy="127.0.0.1,localhost,hyak.local,hyakm.washington.edu" && \
export NO_PROXY="$no_proxy"'

# Choose the tunnel command
case "$TUNNEL_MODE" in
  ephemeral)
    TUNNEL_CMD='~/bin/cloudflared tunnel --url http://localhost:8000'
    ;;
  named)
    TUNNEL_CMD='~/bin/cloudflared tunnel run dawg --url http://localhost:8000'
    ;;
  *)
    echo "ERROR: TUNNEL_MODE must be 'ephemeral' or 'named' (got '$TUNNEL_MODE')" >&2
    exit 1
    ;;
esac

echo "Starting DAWG session on $(hostname)..."
echo "  mode:       $TUNNEL_MODE"
echo "  user:       $DAWG_USER"
echo "  pass:       (set)"
echo ""

# --- Create detached tmux session with first pane ---
tmux new-session -d -s "$SESSION" -n main

# --- Pane 0: MolmoWeb model server ---
tmux send-keys -t "$SESSION:0.0" "
$ENV_PREAMBLE
cd /gscratch/raivn/andy132/dawg/external/molmoweb
export CKPT='./checkpoints/MolmoWeb-4B-Native'
export PREDICTOR_TYPE='native'
echo '[pane 0] Loading MolmoWeb model server (~130s)...'
python -m uvicorn agent.fastapi_model_server:app --host 0.0.0.0 --port 8001
" Enter

# --- Pane 1: playground (waits for MolmoWeb) ---
tmux split-window -t "$SESSION:0" -v
tmux send-keys -t "$SESSION:0.1" "
$ENV_PREAMBLE
export DAWG_USER='$DAWG_USER'
export DAWG_PASS='$DAWG_PASS'
echo '[pane 1] Waiting for MolmoWeb to be ready on port 8001...'
until curl --noproxy '*' -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/docs 2>/dev/null | grep -q '200'; do
  sleep 5
done
echo '[pane 1] MolmoWeb ready. Starting playground.'
bash /gscratch/raivn/andy132/dawg/scripts/run_playground.sh
" Enter

# --- Pane 2: cloudflared (waits for playground) ---
tmux split-window -t "$SESSION:0" -v
tmux send-keys -t "$SESSION:0.2" "
export no_proxy='127.0.0.1,localhost,hyak.local,hyakm.washington.edu'
export NO_PROXY=\"\$no_proxy\"
echo '[pane 2] Waiting for playground to be ready on port 8000...'
until curl --noproxy '*' -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/healthz -u '$DAWG_USER:$DAWG_PASS' 2>/dev/null | grep -q '200'; do
  sleep 3
done
echo '[pane 2] Playground ready. Starting tunnel.'
$TUNNEL_CMD
" Enter

# Make the panes roughly equal height for readability
tmux select-layout -t "$SESSION:0" even-vertical

echo "Session created. Attaching now... (Ctrl-b d to detach)"
sleep 1
tmux attach -t "$SESSION"
