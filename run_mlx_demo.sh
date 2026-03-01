#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/.venv/bin/activate"

export COACH_BACKEND="${COACH_BACKEND:-mlx}"
export MLX_MODEL="${MLX_MODEL:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"
export MLX_MAX_TOKENS="${MLX_MAX_TOKENS:-20}"
export COACH_MIN_INTERVAL_S="${COACH_MIN_INTERVAL_S:-0.35}"
export COACH_REPEAT_COOLDOWN_S="${COACH_REPEAT_COOLDOWN_S:-5.0}"
export COACH_LOG_PATH="${COACH_LOG_PATH:-$ROOT/logs/coach_events.jsonl}"

if [[ -n "${MLX_ADAPTER_PATH:-}" ]]; then
  export MLX_ADAPTER_PATH
fi

exec python server.py
