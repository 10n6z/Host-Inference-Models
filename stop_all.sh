#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
for s in model-gateway-legacy flux2-server magi-server framepack-server; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"
    echo "killed $s"
  fi
done
docker compose down --remove-orphans || true
