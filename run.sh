#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ---- Load environment variables ----
if [ -f ".env" ]; then
  set -a
  # shellcheck source=.env
  source .env
  set +a
fi

# ---- JS dependencies ----
npm install --silent

# ---- One-time setup (idempotent) ----
bash setup.sh

# ---- Activate Python venv ----
if [ ! -d ".venv" ]; then
  echo "Error: .venv not found. Run 'bash setup.sh' first."
  exit 1
fi
source .venv/bin/activate

# ---- Start server ----
exec python3 server.py
