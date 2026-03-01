#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "========================================="
echo "  Squat Coach — Local Setup"
echo "  Run once with internet, then go offline"
echo "========================================="
echo ""

# ---- Python deps ----
echo "[1/5] Installing Python dependencies..."
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt --quiet

# ---- MediaPipe WASM runtime ----
MEDIAPIPE_VERSION="0.10.32"
WASM_DIR="static/mediapipe/wasm"
JS_DIR="static/mediapipe"
NODE_MEDIAPIPE_DIR="node_modules/@mediapipe/tasks-vision"
mkdir -p "$WASM_DIR"
mkdir -p "$JS_DIR"
mkdir -p "models"

echo "[2/5] Preparing MediaPipe WASM runtime..."
WASM_BASE="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm"
WASM_FILES=(
  "vision_wasm_internal.js"
  "vision_wasm_internal.wasm"
  "vision_wasm_nosimd_internal.js"
  "vision_wasm_nosimd_internal.wasm"
)
for f in "${WASM_FILES[@]}"; do
  if [ ! -f "$WASM_DIR/$f" ]; then
    if [ -f "$NODE_MEDIAPIPE_DIR/wasm/$f" ]; then
      echo "  ✓ $f (from node_modules)"
      cp "$NODE_MEDIAPIPE_DIR/wasm/$f" "$WASM_DIR/$f"
    else
      echo "  ↓ $f"
      curl -fLsS "$WASM_BASE/$f" -o "$WASM_DIR/$f"
    fi
  else
    echo "  ✓ $f (cached)"
  fi
done

# ---- MediaPipe JS bundle ----
echo "[3/5] Preparing MediaPipe JS bundle..."
if [ ! -f "$JS_DIR/vision_bundle.mjs" ]; then
  if [ -f "$NODE_MEDIAPIPE_DIR/vision_bundle.mjs" ]; then
    cp "$NODE_MEDIAPIPE_DIR/vision_bundle.mjs" "$JS_DIR/vision_bundle.mjs"
    echo "  ✓ vision_bundle.mjs (from node_modules)"
  else
    curl -fLsS "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs" \
      -o "$JS_DIR/vision_bundle.mjs"
    echo "  ✓ vision_bundle.mjs"
  fi
else
  echo "  ✓ vision_bundle.mjs (cached)"
fi

# ---- Pose model ----
echo "[4/5] Downloading pose landmarker model (~4MB)..."
MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/pose_landmarker_lite.task"
if [ ! -f "$MODEL_FILE" ]; then
  curl -fLsS "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
    -o "$MODEL_FILE"
  echo "  ✓ pose_landmarker_lite.task"
else
  echo "  ✓ pose_landmarker_lite.task (cached)"
fi

# ---- MLX model pre-download ----
echo "[5/5] Pre-downloading MLX coaching model..."
MLX_MODEL="${MODEL:-mlx-community/gemma-3-1b-it-4bit}"
python3 -c "
from mlx_lm import load
import sys
model_id = '$MLX_MODEL'
print(f'  Fetching {model_id} ...')
load(model_id)
print(f'  ✓ {model_id} (cached)')
"

echo ""
echo "========================================="
echo "  Setup complete!"
echo ""
echo "  Start the coach:"
echo "    source .venv/bin/activate && python3 server.py"
echo ""
echo "  To use a fine-tuned adapter:"
echo "    MLX_ADAPTER_PATH=adapters/ python3 server.py"
echo ""
echo "  Then open:"
echo "    http://localhost:8420"
echo ""
echo "  You can now disconnect from the internet."
echo "========================================="
