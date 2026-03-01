"""
Squat Coach — Local Server
Serves everything from disk. Zero cloud dependencies at runtime.

LLM coaching via Ollama (local). Swap models with MODEL env var:
    MODEL=gemma3:4b python3 server.py
"""
import os
import sys
import json
import uvicorn
import httpx
from pathlib import Path
from urllib import error, request
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from loguru import logger

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

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("MODEL", "gemma3:1b")

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


async def call_ollama(squat_state: dict) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": COACH_SYSTEM_PROMPT},
            {"role": "user",   "content": json.dumps(squat_state)},
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "feedback": {"type": "string"},
                "severity": {"type": "string", "enum": ["good", "warn", "bad", "info"]},
                "speak":    {"type": "boolean"},
            },
            "required": ["feedback", "severity", "speak"],
        },
        "options": {"temperature": 0},
    }
    _log = logger.bind(phase=squat_state.get("phase"), rep_number=squat_state.get("rep_number"))
    _log.debug(f"Calling Ollama ({OLLAMA_MODEL})")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            result = json.loads(content)
            _log.debug(
                f"Ollama response: severity={result.get('severity')} speak={result.get('speak')} "
                f"feedback=\"{result.get('feedback')}\""
            )
            return result
    except Exception as exc:
        _log.error(f"Ollama call failed, using fallback: {exc}")
        return {"feedback": "Form check unavailable — keep steady pace.", "severity": "info", "speak": False}

app = FastAPI(title="Squat Coach")

BASE = Path(__file__).parent
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
DEPTH_TARGET_KNEE_DEG = float(os.environ.get(
    "COACH_DEPTH_TARGET_KNEE_DEG", "100"))
TORSO_WARN_DEG = float(os.environ.get("COACH_TORSO_WARN_DEG", "45"))
VALGUS_RATIO = float(os.environ.get("COACH_VALGUS_RATIO", "0.82"))

OLLAMA_SYSTEM = """You are a squat coach model running locally for realtime feedback.
Return strict JSON only with this schema:
{
  "feedback": string,
  "severity": "good|warn|bad|info",
  "speak": boolean
}
Rules:
- Give one short sentence only.
- Prioritize torso safety, then knee tracking, then depth, then tempo, then encouragement.
- Use the structured input only.
- Output valid JSON only."""


def ollama_chat(snapshot):
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": OLLAMA_SYSTEM},
                {"role": "user", "content": json.dumps(snapshot)},
            ],
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "feedback": {"type": "string"},
                    "severity": {"type": "string"},
                    "speak": {"type": "boolean"},
                },
                "required": ["feedback", "severity", "speak"],
            },
            "options": {"temperature": 0},
        }
    ).encode()
    req = request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=60) as response:
        parsed = json.loads(response.read().decode())
    return json.loads(parsed["message"]["content"])


def fallback_feedback(snapshot):
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

    mime = "application/wasm" if filename.endswith(
        ".wasm") else "application/javascript"
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

# ---- WebSocket: LLM coaching via Ollama ----


@app.websocket("/ws/coach")
async def coaching_ws(websocket: WebSocket):
    """
    Receives structured signal snapshots from the browser, forwards them
    to the configured Ollama model (MODEL env var), and returns coaching decisions.

    Swap the model without changing code:
        MODEL=gemma3:4b python3 server.py

    Expected input:
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

    Expected output:
    {
        "feedback": "...",
        "severity": "good | warn | bad | info",
        "speak": true
    }

    Falls back to a safe info-severity message if Ollama is unreachable.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected to /ws/coach (model={OLLAMA_MODEL})")
    await websocket.send_json({"type": "hello", "model": OLLAMA_MODEL})
    try:
        while True:
            data = await websocket.receive_json()
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
            result = await call_ollama(data)
            logger.bind(
                rep_number=rep,
                phase=phase,
                severity=result.get("severity"),
                speak=result.get("speak"),
            ).info(f"Coach decision: {result.get('feedback')}")
            await websocket.send_json(result)
            logger.debug(f"Response sent to client: severity={result.get('severity')} speak={result.get('speak')}")
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
    logger.info("All inference runs on-device")
    logger.info(f"LLM: {OLLAMA_MODEL} via Ollama at {OLLAMA_BASE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=8420)
