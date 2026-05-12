#!/usr/bin/env bash
# Update docs/status.json with the current playground URL and push to GitHub.
# Within ~30s GitHub Pages rebuilds and the site reflects the new status.
#
# Usage:
#   bash scripts/update_url.sh https://something.trycloudflare.com
#   bash scripts/update_url.sh offline                 # mark playground down
set -euo pipefail

REPO_ROOT="/gscratch/raivn/andy132/dawg"
STATUS_FILE="$REPO_ROOT/docs/status.json"

URL="${1:-}"
if [[ -z "$URL" ]]; then
  echo "usage: $0 <trycloudflare-url | offline>"
  exit 1
fi

cd "$REPO_ROOT"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$URL" == "offline" ]]; then
  cat > "$STATUS_FILE" <<EOF
{
  "is_live": false,
  "playground_url": null,
  "last_updated": "$NOW"
}
EOF
  MSG="status: offline"
else
  cat > "$STATUS_FILE" <<EOF
{
  "is_live": true,
  "playground_url": "$URL",
  "last_updated": "$NOW"
}
EOF
  MSG="status: live at $URL"
fi

git add "$STATUS_FILE"
git commit -m "$MSG" -q
git push -q
echo "$MSG"
echo "Site will reflect within ~30s: https://andyye132.github.io/DAWG/"
