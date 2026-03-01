"""
Squat Coach — Local Server
Serves everything from disk. Zero cloud dependencies at runtime.

LLM coaching via MLX (in-process, Apple Silicon). Swap models with MODEL env var:
    MODEL=mlx-community/gemma-3-4b-it-4bit python3 server.py

To use LoRA adapters from fine-tuning:
    MODEL=mlx-community/gemma-3-1b-it-4bit MLX_ADAPTER_PATH=adapters/ python3 server.py
"""
import asyncio
import json
import os
import re
import sys
import time
import uvicorn
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from loguru import logger
from mlx_lm import load, generate

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
)
logger.add(
    "logs/squat_coach.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    serialize=True,
)

MLX_MODEL = os.environ.get("MODEL", "mlx-community/gemma-3-1b-it-4bit")
MLX_ADAPTER_PATH = os.environ.get("MLX_ADAPTER_PATH", None) or None
MLX_MAX_TOKENS = int(os.environ.get("MLX_MAX_TOKENS", "48"))

DEPTH_TARGET_KNEE_DEG = float(os.environ.get("COACH_DEPTH_TARGET_KNEE_DEG", "100"))
TORSO_WARN_DEG = float(os.environ.get("COACH_TORSO_WARN_DEG", "45"))
VALGUS_RATIO = float(os.environ.get("COACH_VALGUS_RATIO", "0.82"))
COACH_MIN_INTERVAL_S = float(os.environ.get("COACH_MIN_INTERVAL_S", "0.35"))
COACH_REPEAT_COOLDOWN_S = float(os.environ.get("COACH_REPEAT_COOLDOWN_S", "1.5"))
COACH_LOG_PATH = Path(os.environ.get("COACH_LOG_PATH", str(Path("logs") / "coach_events.jsonl")))

COACH_SYSTEM_PROMPT = """You are a real-time squat coach. Return only valid JSON, no markdown, no explanation.

Output schema:
{"feedback": "<one short coaching cue, max 15 words>", "severity": "<good|warn|bad|info>", "speak": <true|false>}

Severity rules:
- good: correct form or depth achieved
- warn: minor issue, cue to correct
- bad: significant form fault requiring immediate correction
- info: neutral observation

speak: true for warn and bad, false for good and info.
Do not mention raw variable names. Output only the JSON object, nothing else."""

app = FastAPI(title="Squat Coach")

BASE = Path(__file__).parent

CANONICAL_LIVE_CUES = (
    "Chest up.",
    "Go deeper.",
    "Good depth.",
    "Step back so I can see you.",
    "Move closer so I can see you.",
    "Turn sideways to the camera.",
    "Fix your camera setup.",
)

# ---- Load MLX model once at startup ----
try:
    _mlx_model, _mlx_tokenizer = load(MLX_MODEL, adapter_path=MLX_ADAPTER_PATH)
    logger.info(f"MLX model loaded: {MLX_MODEL}")
    if MLX_ADAPTER_PATH:
        logger.info(f"LoRA adapters: {MLX_ADAPTER_PATH}")
except Exception as _load_exc:
    _mlx_model, _mlx_tokenizer = None, None
    logger.critical(f"MLX model failed to load — fallback rules only: {_load_exc}")


def _mlx_infer_sync(squat_state: dict) -> dict:
    """Run MLX inference synchronously for one squat state snapshot.

    Formats the input as a chat prompt, calls ``mlx_lm.generate()``, and
    extracts the first JSON object from the raw output string.

    Parameters
    ----------
    squat_state : dict
        Structured squat snapshot from the browser (phase, angles, rep number, etc.).

    Returns
    -------
    dict
        Parsed coaching decision with keys ``feedback``, ``severity``, ``speak``.

    Raises
    ------
    ValueError
        If the model output contains no parseable JSON object.
    """
    messages = [
        {"role": "system", "content": COACH_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(squat_state)},
    ]
    prompt = _mlx_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw = generate(_mlx_model, _mlx_tokenizer, prompt=prompt, max_tokens=MLX_MAX_TOKENS, verbose=False)
    match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON in MLX output: {raw!r}")


async def call_mlx(squat_state: dict) -> dict:
    """Forward a squat state snapshot to the MLX model and return a coaching decision.

    Runs synchronous MLX inference in a thread pool executor to avoid blocking
    FastAPI's async event loop. Falls back to ``fallback_feedback`` if the model
    is not loaded or inference fails.

    Parameters
    ----------
    squat_state : dict
        Structured squat snapshot from the browser.

    Returns
    -------
    dict
        Coaching decision with keys ``feedback`` (str), ``severity`` (str),
        and ``speak`` (bool).
    """
    _log = logger.bind(phase=squat_state.get("phase"), rep_number=squat_state.get("rep_number"))
    _log.debug(f"Calling MLX ({MLX_MODEL})")
    if _mlx_model is None:
        _log.warning("MLX model not loaded, using rule-based fallback")
        return fallback_feedback(squat_state)
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _mlx_infer_sync, squat_state)
        _log.debug(
            f"MLX response: severity={result.get('severity')} speak={result.get('speak')} "
            f"feedback=\"{result.get('feedback')}\""
        )
        return result
    except Exception as exc:
        _log.error(f"MLX inference failed, using fallback: {exc}")
        return {"feedback": "Form check unavailable — keep steady pace.", "severity": "info", "speak": False}


def fallback_feedback(snapshot: dict) -> dict:
    """Deterministic rule-based coaching fallback used when MLX is unavailable.

    Checks torso lean, knee valgus, and squat depth in priority order and
    returns the first matching coaching cue.

    Parameters
    ----------
    snapshot : dict
        Structured squat snapshot from the browser.

    Returns
    -------
    dict
        Coaching decision with keys ``feedback``, ``severity``, and ``speak``.
    """
    phase = snapshot.get("phase", "standing")
    torso_angle = float(snapshot.get("torso_angle") or 0)
    knee_valgus_ratio = float(snapshot.get("knee_valgus_ratio") or 1)
    knee_angle = float(snapshot.get("knee_angle") or 180)
    recent_depth = snapshot.get("depth_history") or []
    if torso_angle >= TORSO_WARN_DEG and phase != "standing":
        return {"feedback": "Keep your chest up and reduce the forward lean.", "severity": "warn", "speak": True}
    if knee_valgus_ratio < VALGUS_RATIO and phase in {"descending", "bottom"}:
        return {"feedback": "Push your knees out to track over your toes.", "severity": "bad", "speak": True}
    if phase == "bottom" and knee_angle > DEPTH_TARGET_KNEE_DEG:
        return {"feedback": "Sit a little deeper to hit your depth target.", "severity": "warn", "speak": True}
    if recent_depth and max(recent_depth[-3:]) > DEPTH_TARGET_KNEE_DEG + 8:
        return {"feedback": "You are cutting depth high. Stay patient into the bottom.", "severity": "warn", "speak": True}
    return {"feedback": "Good rep. Keep that shape.", "severity": "good", "speak": False}


def deterministic_setup_feedback(snapshot: dict) -> dict | None:
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


def canonicalize_feedback(text: str) -> str:
    value = (text or "").strip()
    lowered = value.lower()
    for cue in CANONICAL_LIVE_CUES:
        if lowered == cue.lower():
            return cue
    mapping = (
        (("chest", "lean"), "Chest up."),
        (("chest",), "Chest up."),
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


def log_event(payload: dict) -> None:
    COACH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COACH_LOG_PATH.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def apply_runtime_controls(result: dict, state: dict) -> dict:
    now = time.monotonic()
    result = dict(result)
    result["feedback"] = canonicalize_feedback(result.get("feedback", ""))

    last_emit_at = state.get("last_emit_at", 0.0)
    last_feedback = state.get("last_feedback", "")
    recent = now - last_emit_at
    same_feedback = result["feedback"] == last_feedback

    if recent < COACH_MIN_INTERVAL_S:
        cached = dict(state.get("last_result") or result)
        cached["source"] = f"{cached.get('source', 'cached')}:cadence_gate"
        cached["speak"] = False
        return cached

    if same_feedback and recent < COACH_REPEAT_COOLDOWN_S:
        result["speak"] = False
        result["source"] = f"{result.get('source', 'mlx')}:repeat_gate"

    state["last_emit_at"] = now
    state["last_feedback"] = result["feedback"]
    state["last_result"] = dict(result)
    return result


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


# ---- WebSocket: LLM coaching via MLX ----

@app.websocket("/ws/coach")
async def coaching_ws(websocket: WebSocket):
    """Accept browser WebSocket connections and return MLX coaching decisions.

    Receives structured squat state snapshots from the browser, forwards them
    to ``call_mlx()``, and streams back coaching decisions. Falls back to
    rule-based feedback if the MLX model is unavailable.

    Swap the model without changing code::

        MODEL=mlx-community/gemma-3-4b-it-4bit python3 server.py

    Expected input::

        {
            "phase": "bottom",
            "rep_number": 4,
            "knee_angle": 108,
            "hip_angle": 72,
            "torso_angle": 41,
            "knee_valgus_ratio": 0.79,
            "depth_history": [94, 96, 101, 108],
            "phase_durations_ms": {"descent": 820, "bottom": 340}
        }

    Expected output::

        {
            "feedback": "...",
            "severity": "good | warn | bad | info",
            "speak": true
        }
    """
    await websocket.accept()
    state = {"last_emit_at": 0.0, "last_feedback": "", "last_result": None}
    logger.info(f"WebSocket client connected to /ws/coach (model={MLX_MODEL})")
    await websocket.send_json({"type": "hello", "model": MLX_MODEL})
    try:
        while True:
            data = await websocket.receive_json()
            setup_result = deterministic_setup_feedback(data)
            if setup_result is not None:
                setup_result["source"] = "rules_setup_gate"
                final = apply_runtime_controls(setup_result, state)
                log_event({"ts": time.time(), "snapshot": data, "result": final})
                await websocket.send_json(final)
                continue
            phase = data.get("phase")
            rep = data.get("rep_number")
            logger.debug(
                f"Snapshot received: phase={phase} rep={rep} "
                f"knee={data.get('knee_angle')} hip={data.get('hip_angle')} "
                f"torso={data.get('torso_angle')} valgus={data.get('knee_valgus_ratio')}"
            )
            if phase == "bottom":
                logger.bind(
                    rep_number=rep,
                    phase=phase,
                    knee_angle=data.get("knee_angle"),
                    torso_angle=data.get("torso_angle"),
                    valgus_ratio=data.get("knee_valgus_ratio"),
                ).info(f"Rep {rep} bottom reached")
            result = await call_mlx(data)
            result["source"] = f"mlx:{MLX_MODEL}"
            final = apply_runtime_controls(result, state)
            logger.bind(
                rep_number=rep,
                phase=phase,
                severity=final.get("severity"),
                speak=final.get("speak"),
            ).info(f"Coach decision: {final.get('feedback')}")
            log_event({"ts": time.time(), "snapshot": data, "result": final})
            await websocket.send_json(final)
            logger.debug(f"Response sent to client: severity={final.get('severity')} speak={final.get('speak')}")
    except WebSocketDisconnect:
        pass
    except json.JSONDecodeError as exc:
        logger.error(f"Bad WebSocket message (invalid JSON): {exc}")
    except Exception as exc:
        logger.warning(f"WebSocket /ws/coach closed with error: {exc}")
    finally:
        logger.info("WebSocket client disconnected from /ws/coach")


# ---- Serve the main app ----

@app.get("/")
async def index():
    return FileResponse(BASE / "static" / "index.html")


# ---- Static files fallback ----
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

if __name__ == "__main__":
    logger.info("Squat Coach — Fully Local")
    logger.info("Listening at http://localhost:8420")
    logger.info("All inference runs on-device via MLX")
    logger.info(f"LLM: {MLX_MODEL} (in-process)")
    if MLX_ADAPTER_PATH:
        logger.info(f"LoRA adapters: {MLX_ADAPTER_PATH}")
    logger.info(f"MLX max tokens: {MLX_MAX_TOKENS}")
    logger.info(f"Coach log path: {COACH_LOG_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8420)
