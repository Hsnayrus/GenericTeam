# Squat Coach — Fully Local, Fully Offline

A realtime squat form coach that runs **entirely on your device**. No cloud, no API calls, no internet required after initial setup.

## Architecture

```
Webcam
   ↓
MediaPipe Pose Lite (WASM+WebGL, in-browser)
   ↓
Confidence Gating (drop frames where joints < 0.5 visibility)
   ↓
EMA Smoothing (α=0.7 on landmark coordinates)
   ↓
Angle Computation (knee, hip, torso — averaged bilateral)
   ↓
Phase Detection (standing → descent → bottom → ascent state machine)
   ↓
Form Checks (depth, knee valgus, forward lean — phase-aware)
   ↓
Decision Gating (250ms persistence before any alert fires)
   ↓
Feedback Engine (visual overlay + speech synthesis)
   ↓
[Optional] Gemma 2B via WebSocket (fine-tuned coaching model)
```

## What runs where

| Component | Runs in | Notes |
|---|---|---|
| Pose estimation | Browser (WASM+WebGL) | ~5-15ms/frame, MediaPipe Lite |
| Signal processing | Browser (JS) | EMA, angles, phase detection |
| Rule-based coaching | Browser (JS) | Hardcoded form checks |
| Gemma 2B coaching | Local Python server (MLX) | Optional, replaces hardcoded rules |
| UI + audio | Browser | Canvas overlay + Web Speech API |

## Quick Start

```bash
# 1. Clone and enter directory
cd squat-coach-local

# 2. Install JS dependency once if needed
npm install

# 3. Run setup (copies MediaPipe assets from node_modules and downloads the pose model)
chmod +x setup.sh
bash setup.sh

# 4. Start the local server
python3 server.py

# 5. Open in browser
#    http://localhost:8420

# 6. Disconnect from internet — everything still works
```

## Setup Details

`setup.sh` prepares these files once:
- MediaPipe WASM runtime (~3MB) → `static/mediapipe/wasm/`
- MediaPipe JS bundle → `static/mediapipe/vision_bundle.mjs`
- Pose Landmarker Lite model (~4MB) → `models/`
- Python deps: `fastapi`, `uvicorn`

If `@mediapipe/tasks-vision` is already installed locally, the setup script copies the MediaPipe runtime from `node_modules` instead of downloading it again.

After setup, **zero network requests** are made at runtime.

## Camera Setup

**Best angle: ~45° from the side**, about 6-8 feet away, full body in frame.

- Side view = best for knee/hip angle accuracy
- Front view = best for knee valgus detection
- 45° = good compromise for both

## The "Unplug the Internet" Demo

1. Run `setup.sh` with internet
2. Start `python3 server.py`
3. Open `http://localhost:8420`
4. Do a few squats — see real-time coaching
5. **Disconnect WiFi/Ethernet**
6. Do more squats — **identical performance**
7. Point to the HUD: NET shows OFFLINE, everything else green

## Gemma 2B Integration (Coming)

The WebSocket endpoint at `/ws/coach` is ready. When Gemma is connected:

**Input** (sent on phase transitions):
```json
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
```

**Output** (coaching decision):
```json
{
  "feedback": "Depth degrading over last 4 reps — fatigue pattern. One more good one or rack it.",
  "severity": "warn",
  "speak": true
}
```

Why Gemma > hardcoded rules:
- Tracks patterns across reps (fatigue detection)
- Correlates multiple signals (lean + depth = compensation)
- Adapts coaching style to experience level
- Makes judgment calls about when to interrupt

Why on-device:
- Privacy (filming yourself exercising)
- Latency (<100ms needed for mid-rep feedback)
- Offline (garage gyms, basements)
- Cost (no per-inference API charges)

## File Structure

```
squat-coach-local/
├── setup.sh                          # One-time dependency download
├── server.py                         # FastAPI local server + Gemma WS endpoint
├── static/
│   ├── index.html                    # Full app (pose + pipeline + UI)
│   └── mediapipe/
│       ├── vision_bundle.mjs         # MediaPipe JS (local)
│       └── wasm/                     # WASM runtime (local)
├── models/
│   └── pose_landmarker_lite.task     # Pose model (local)
└── README.md
```
