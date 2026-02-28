"""
Squat Coach — Local Server
Serves everything from disk. Zero cloud dependencies at runtime.
"""
import json
import os
import uvicorn
from pathlib import Path
from urllib import error, request
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

app = FastAPI(title="Squat Coach")

BASE = Path(__file__).parent
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
DEPTH_TARGET_KNEE_DEG = float(os.environ.get("COACH_DEPTH_TARGET_KNEE_DEG", "100"))
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
    try:
        while True:
            snapshot = await websocket.receive_json()
            try:
                result = ollama_chat(snapshot)
                result["source"] = f"ollama:{OLLAMA_MODEL}"
            except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                result = fallback_feedback(snapshot)
                result["source"] = f"rules_fallback:{type(exc).__name__}"
            await websocket.send_json(result)
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
    print(f"  ➜  Coach model: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    print("  ➜  Disconnect from internet anytime\n")
    uvicorn.run(app, host="0.0.0.0", port=8420)
