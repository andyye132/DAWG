#!/usr/bin/env bash
# Launch the DAWG playground server. Expects:
#   - dawg conda env activated
#   - MolmoWeb model server already running on 127.0.0.1:8001
#   - Env vars DAWG_USER / DAWG_PASS set (or accept defaults)
set -euo pipefail

cd /gscratch/raivn/andy132/dawg

export PYTHONPATH="${PYTHONPATH:-}:/gscratch/raivn/andy132/dawg/src:/gscratch/raivn/andy132/dawg/external/molmoweb"
export DAWG_USER="${DAWG_USER:-dawg}"
export DAWG_PASS="${DAWG_PASS:-changeme}"
export MOLMOWEB_URL="${MOLMOWEB_URL:-http://127.0.0.1:8001}"

echo "DAWG playground"
echo "  user:        $DAWG_USER"
echo "  molmoweb:    $MOLMOWEB_URL"
echo "  listening:   http://0.0.0.0:8000"
echo ""

uvicorn dawg.playground.server:app --host 0.0.0.0 --port 8000
