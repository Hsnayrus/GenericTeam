"""
Squat Coach — Local Server
Serves everything from disk. Zero cloud dependencies at runtime.

Future: WebSocket endpoint for Gemma 2B inference.
"""
import uvicorn
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

app = FastAPI(title="Squat Coach")

BASE = Path(__file__).parent

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

# ---- WebSocket for future Gemma integration ----
@app.websocket("/ws/coach")
async def coaching_ws(websocket: WebSocket):
    """
    Future: receives structured signal snapshots from the browser,
    runs them through fine-tuned Gemma 2B, returns coaching decisions.
    
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
        "severity": "warn",
        "speak": true
    }
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # TODO: replace with Gemma 2B inference via MLX
            # For now, echo back a placeholder
            await websocket.send_json({
                "feedback": "Gemma integration pending — using rule-based fallback",
                "severity": "info",
                "speak": False,
            })
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
    print("  ➜  All inference runs on-device")
    print("  ➜  Disconnect from internet anytime\n")
    uvicorn.run(app, host="0.0.0.0", port=8420)
