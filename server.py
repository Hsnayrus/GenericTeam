"""
Squat Coach — Local Server
Serves everything from disk. Zero cloud dependencies at runtime.
"""
import json
import os
import time
import uvicorn
from pathlib import Path
from urllib import error, request
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from prompt_templates import LIVE_FEEDBACK_SCHEMA, build_live_feedback_messages, rule_live_feedback, sanitize_live_feedback

app = FastAPI(title="Squat Coach")

BASE = Path(__file__).parent
COACH_BACKEND = os.environ.get("COACH_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0"))
OLLAMA_TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.9"))
OLLAMA_TOP_K = int(os.environ.get("OLLAMA_TOP_K", "40"))
OLLAMA_MIN_P = float(os.environ.get("OLLAMA_MIN_P", "0"))
OLLAMA_REPEAT_PENALTY = float(os.environ.get("OLLAMA_REPEAT_PENALTY", "1.15"))
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "48"))
OLLAMA_STOP = [token for token in os.environ.get("OLLAMA_STOP", "").split("||") if token]
OLLAMA_SEED_RAW = os.environ.get("OLLAMA_SEED")
OLLAMA_SEED = int(OLLAMA_SEED_RAW) if OLLAMA_SEED_RAW else None
DEPTH_TARGET_KNEE_DEG = float(os.environ.get("COACH_DEPTH_TARGET_KNEE_DEG", "100"))
TORSO_WARN_DEG = float(os.environ.get("COACH_TORSO_WARN_DEG", "45"))
VALGUS_RATIO = float(os.environ.get("COACH_VALGUS_RATIO", "0.82"))
COACH_MIN_INTERVAL_S = float(os.environ.get("COACH_MIN_INTERVAL_S", "0.35"))
COACH_REPEAT_COOLDOWN_S = float(os.environ.get("COACH_REPEAT_COOLDOWN_S", "5.0"))
COACH_LOG_PATH = Path(os.environ.get("COACH_LOG_PATH", str(BASE / "logs" / "coach_events.jsonl")))
MLX_MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
MLX_ADAPTER_PATH = os.environ.get("MLX_ADAPTER_PATH") or None
MLX_MAX_TOKENS = int(os.environ.get("MLX_MAX_TOKENS", "20"))
MLX_MODEL_HANDLE = None
MLX_TOKENIZER_HANDLE = None

CANONICAL_LIVE_CUES = (
    "Chest up.",
    "Go deeper.",
    "Knees out.",
    "Good depth.",
    "Step back so I can see you.",
    "Move closer so I can see you.",
    "Turn sideways to the camera.",
    "Fix your camera setup.",
)


def deterministic_setup_feedback(snapshot):
    if snapshot.get("is_setup"):
        if not snapshot.get("side_view_ok", True):
            return {"feedback": "Turn sideways to the camera.", "severity": "info", "speak": True}
        if not snapshot.get("head_visible", True) or not snapshot.get("feet_visible", True):
            return {"feedback": "Fix your camera setup.", "severity": "info", "speak": True}
        if snapshot.get("too_close"):
            return {"feedback": "Step back so I can see you.", "severity": "info", "speak": True}
        if snapshot.get("too_far"):
            return {"feedback": "Move closer so I can see you.", "severity": "info", "speak": True}
    return None


def log_event(payload):
    COACH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COACH_LOG_PATH.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def canonicalize_feedback(text):
    value = (text or "").strip()
    lowered = value.lower()
    for cue in CANONICAL_LIVE_CUES:
        if lowered == cue.lower():
            return cue
    mapping = (
        (("chest", "lean"), "Chest up."),
        (("chest",), "Chest up."),
        (("knees",), "Knees out."),
        (("knee",), "Knees out."),
        (("deeper",), "Go deeper."),
        (("depth", "good"), "Good depth."),
        (("good", "depth"), "Good depth."),
        (("step back",), "Step back so I can see you."),
        (("move closer",), "Move closer so I can see you."),
        (("turn sideways",), "Turn sideways to the camera."),
        (("camera setup",), "Fix your camera setup."),
    )
    for needles, cue in mapping:
        if all(needle in lowered for needle in needles):
            return cue
    return value


def apply_runtime_controls(snapshot, result, state):
    now = time.monotonic()
    result["feedback"] = canonicalize_feedback(result.get("feedback", ""))

    last_emit_at = state.get("last_emit_at", 0.0)
    last_feedback = state.get("last_feedback", "")
    last_spoken_at = state.get("last_spoken_at", 0.0)
    last_spoken_feedback = state.get("last_spoken_feedback", "")
    recent = now - last_emit_at
    recent_spoken = now - last_spoken_at
    same_feedback = result["feedback"] == last_feedback
    same_spoken_feedback = result["feedback"] == last_spoken_feedback

    if recent < COACH_MIN_INTERVAL_S:
        cached = dict(state.get("last_result") or result)
        cached["source"] = f"{cached.get('source', 'cached')}:cadence_gate"
        cached["speak"] = False
        return cached

    if result.get("severity") == "good":
        result["speak"] = False

    if same_feedback and recent < COACH_REPEAT_COOLDOWN_S:
        result["speak"] = False
        result["source"] = f"{result.get('source', 'model')}:repeat_gate"

    if result.get("speak") and same_spoken_feedback and recent_spoken < COACH_REPEAT_COOLDOWN_S:
        result["speak"] = False
        result["source"] = f"{result.get('source', 'model')}:speak_repeat_gate"

    state["last_emit_at"] = now
    state["last_feedback"] = result["feedback"]
    if result.get("speak"):
        state["last_spoken_at"] = now
        state["last_spoken_feedback"] = result["feedback"]
    state["last_result"] = dict(result)
    return result


def load_mlx_backend():
    global MLX_MODEL_HANDLE, MLX_TOKENIZER_HANDLE
    if MLX_MODEL_HANDLE is not None and MLX_TOKENIZER_HANDLE is not None:
        return MLX_MODEL_HANDLE, MLX_TOKENIZER_HANDLE
    from mlx_lm import load

    MLX_MODEL_HANDLE, MLX_TOKENIZER_HANDLE = load(MLX_MODEL, adapter_path=MLX_ADAPTER_PATH)
    return MLX_MODEL_HANDLE, MLX_TOKENIZER_HANDLE

def build_ollama_options():
    options = {
        "temperature": OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
        "top_k": OLLAMA_TOP_K,
        "min_p": OLLAMA_MIN_P,
        "repeat_penalty": OLLAMA_REPEAT_PENALTY,
        "num_predict": OLLAMA_NUM_PREDICT,
    }
    if OLLAMA_STOP:
        options["stop"] = OLLAMA_STOP
    if OLLAMA_SEED is not None:
        options["seed"] = OLLAMA_SEED
    return options

def ollama_chat(snapshot):
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": build_live_feedback_messages(snapshot),
            "stream": False,
            "format": LIVE_FEEDBACK_SCHEMA,
            "options": build_ollama_options(),
        }
    ).encode()
    req = request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=60) as response:
        parsed = json.loads(response.read().decode())
    return sanitize_live_feedback(snapshot, json.loads(parsed["message"]["content"]))


def mlx_chat(snapshot):
    from mlx_lm import generate

    model, tokenizer = load_mlx_backend()
    messages = build_live_feedback_messages(snapshot)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = messages[-1]["content"]
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=False,
        max_tokens=MLX_MAX_TOKENS,
    )
    return sanitize_live_feedback(snapshot, json.loads(text))


def model_chat(snapshot):
    if COACH_BACKEND == "mlx":
        return mlx_chat(snapshot), f"mlx:{MLX_MODEL}"
    result = ollama_chat(snapshot)
    return result, f"ollama:{OLLAMA_MODEL}"

# ---- Serve model file with correct MIME ----
@app.get("/models/{filename}")
async def serve_model(filename: str):
    path = BASE / "models" / filename
    if not path.exists():
        return Response(status_code=404, content="Model not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "public, max-age=31536000",
            "Access-Control-Allow-Origin": "*",
        }
    )

# ---- Serve WASM files with correct MIME types ----
@app.get("/mediapipe/wasm/{filename}")
async def serve_wasm(filename: str):
    path = BASE / "static" / "mediapipe" / "wasm" / filename
    if not path.exists():
        return Response(status_code=404, content="WASM file not found")
    
    mime = "application/wasm" if filename.endswith(".wasm") else "application/javascript"
    return FileResponse(
        path,
        media_type=mime,
        headers={
            "Cache-Control": "public, max-age=31536000",
            "Access-Control-Allow-Origin": "*",
        }
    )

# ---- Serve MediaPipe JS with correct MIME ----
@app.get("/mediapipe/{filename}")
async def serve_mediapipe_js(filename: str):
    path = BASE / "static" / "mediapipe" / filename
    if not path.exists():
        return Response(status_code=404, content="File not found")
    return FileResponse(
        path,
        media_type="application/javascript",
        headers={
            "Cache-Control": "public, max-age=31536000",
            "Access-Control-Allow-Origin": "*",
        }
    )

# ---- WebSocket for local Ollama integration ----
@app.websocket("/ws/coach")
async def coaching_ws(websocket: WebSocket):
    await websocket.accept()
    state = {
        "last_emit_at": 0.0,
        "last_feedback": "",
        "last_spoken_at": 0.0,
        "last_spoken_feedback": "",
        "last_result": None,
    }
    try:
        while True:
            snapshot = await websocket.receive_json()
            setup_result = deterministic_setup_feedback(snapshot)
            if setup_result is not None:
                setup_result["source"] = "rules_setup_gate"
                final = apply_runtime_controls(snapshot, setup_result, state)
                log_event({"ts": time.time(), "snapshot": snapshot, "result": final})
                await websocket.send_json(final)
                continue
            try:
                result, source = model_chat(snapshot)
                result["source"] = source
            except (ImportError, ModuleNotFoundError, error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                result = rule_live_feedback(snapshot)
                result["source"] = f"rules_fallback:{type(exc).__name__}"
            final = apply_runtime_controls(snapshot, result, state)
            log_event({"ts": time.time(), "snapshot": snapshot, "result": final})
            await websocket.send_json(final)
    except Exception:
        pass

# ---- Serve the main app ----
@app.get("/")
async def index():
    return FileResponse(BASE / "static" / "index.html")

# ---- Static files fallback ----
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

if __name__ == "__main__":
    print("\n  🏋️  Squat Coach — Fully Local")
    print("  ➜  http://localhost:8420")
    if COACH_BACKEND == "mlx":
        print(f"  ➜  Coach backend: mlx ({MLX_MODEL})")
        if MLX_ADAPTER_PATH:
            print(f"  ➜  MLX adapter: {MLX_ADAPTER_PATH}")
        print(f"  ➜  MLX max tokens: {MLX_MAX_TOKENS}")
    else:
        print(f"  ➜  Coach backend: ollama ({OLLAMA_MODEL} @ {OLLAMA_BASE_URL})")
        print(
            "  ➜  Decoding:"
            f" temp={OLLAMA_TEMPERATURE} top_p={OLLAMA_TOP_P} top_k={OLLAMA_TOP_K}"
            f" min_p={OLLAMA_MIN_P} repeat_penalty={OLLAMA_REPEAT_PENALTY}"
            f" num_predict={OLLAMA_NUM_PREDICT}"
            + (f" seed={OLLAMA_SEED}" if OLLAMA_SEED is not None else "")
        )
    print(f"  ➜  Log file: {COACH_LOG_PATH}")
    print("  ➜  Disconnect from internet anytime\n")
    uvicorn.run(app, host="0.0.0.0", port=8420)
