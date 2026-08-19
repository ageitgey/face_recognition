#!/usr/bin/env bash
# Deploy the Face2AI Hermes plugin:
#   - Python half + dashboard API  -> the Hermes host (VPS) at $HERMES_HOME/plugins/face2ai
#   - desktop half (desktop/plugin.js) -> this Mac's ~/.hermes/plugins/face2ai (the desktop app scans locally)
# Usage: ./deploy.sh [ssh-host]   (default: hermes-brain)
set -euo pipefail
HOST="${1:-hermes-brain}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/face2ai"

echo "== VPS ($HOST): plugin folder"
rsync -az --delete --exclude '__pycache__' "$SRC/" "$HOST:~/.hermes/plugins/face2ai/"
ssh "$HOST" 'bash -l -s' <<'REMOTE'
set -e
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
PY="$HOME/.hermes/hermes-agent/venv/bin/python"
"$PY" -c "import httpx" 2>/dev/null || "$PY" -m pip install -q httpx
hermes plugins enable face2ai >/dev/null 2>&1 || true
hermes config set plugins.entries.face2ai.settings.events_url http://127.0.0.1:8765 >/dev/null 2>&1 || true
echo "-- plugins list:"; hermes plugins list 2>/dev/null | grep -i face2ai || echo "(face2ai not listed yet)"
echo "-- restarting gateway + dashboard"
if systemctl --user is-active hermes-gateway >/dev/null 2>&1; then systemctl --user restart hermes-gateway; else systemctl restart hermes-gateway 2>/dev/null || true; fi
systemctl restart hermes-dashboard 2>/dev/null || systemctl --user restart hermes-dashboard 2>/dev/null || true
REMOTE

echo "== Mac: desktop half"
mkdir -p "$HOME/.hermes/plugins/face2ai/desktop"
cp "$SRC/plugin.yaml" "$HOME/.hermes/plugins/face2ai/plugin.yaml"
cp "$SRC/desktop/plugin.js" "$HOME/.hermes/plugins/face2ai/desktop/plugin.js"
echo "done. In the desktop app: Settings → Plugins → enable 'Face2AI presence' (or ⌘K → Reload desktop plugins)."
