#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-mlx-community/gemma-3-1b-it-4bit}"
export MLX_MAX_TOKENS="${MLX_MAX_TOKENS:-48}"
export COACH_MIN_INTERVAL_S="${COACH_MIN_INTERVAL_S:-0.35}"
export COACH_REPEAT_COOLDOWN_S="${COACH_REPEAT_COOLDOWN_S:-1.5}"
export COACH_LOG_PATH="${COACH_LOG_PATH:-$ROOT/logs/coach_events.jsonl}"

exec python3 "$ROOT/server.py"
