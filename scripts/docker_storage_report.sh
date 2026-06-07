#!/usr/bin/env bash
set -euo pipefail

echo "== docker system df =="
docker system df

echo
echo "== docker compose images =="
docker compose images

echo
echo "== docker images =="
docker images

cat <<'EOF'

Optional layer inspection:
  docker history --human IMAGE_NAME
  dive IMAGE_NAME
EOF
