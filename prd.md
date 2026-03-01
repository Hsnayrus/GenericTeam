# PRD — Squat Coach Local

## Hackathon Requirements

**Challenge:** Build a system combining a fine-tuned on-device model + agentic behavior + visual input, with a genuine reason to run on-device. Voice is an optional bonus.

### Component Checklist

| Component | Requirement | Implementation |
| --------- | ----------- | -------------- |
| **On-device model** | Fine-tuned Gemma family model | Gemma 2B (or 3 4B) via MLX; fine-tuned with LoRA on JSONL from `generate_synthetic_data.py` |
| **Agentic behavior** | System decides and acts autonomously | `PhaseDetector` state machine triggers `FormChecker` → `FeedbackEngine` (visual + speech) → WebSocket coaching loop |
| **Visual input** | Camera/video feeds into the system | Webcam → MediaPipe Pose Lite WASM → bilateral angle computation → phase + form analysis |
| **On-device justification** | Must not be "cloud with local runtime" | (1) <50ms feedback latency physically impossible with cloud round-trip, (2) workout video cannot leave device, (3) works fully offline |
| **Voice (bonus)** | Natural voice I/O adds value | Web Speech API TTS for real-time form cues (e.g. "go deeper", "knees out") |

### Why On-Device Genuinely Matters Here

- **Latency:** Form correction must happen within the rep (~300–800ms descent). Cloud adds 100–500ms+ — too slow to be actionable.
- **Privacy:** Users won't stream workout video to a server. Local inference removes the trust barrier entirely.
- **Offline:** Gyms often have poor connectivity. The app works without any network after setup.
- **Economics:** Per-call API pricing for real-time 15fps inference is prohibitive. Local model = zero marginal cost.

### Fine-Tuning Strategy

- **Base model:** Gemma 2B (MLX) or Gemma 3 4B depending on available time
- **Method:** LoRA fine-tuning (<1 hour target on Apple Silicon M-series)
- **Data:** `generate_synthetic_data.py` → 6-label JSONL (good_rep, shallow_depth, knee_valgus, forward_lean, cropped_feet, bad_side_view)
- **Goal:** Gemma replaces/augments hardcoded `FormChecker` rules with pattern-learned coaching that generalizes to edge cases

### System Integration (How Components Reinforce Each Other)

```
[Visual] Webcam → MediaPipe WASM → pose landmarks
    ↓
[Agent] PhaseDetector (state) → FormChecker (rules) → GemmaClient (learned)
    ↓
[Voice] FeedbackEngine → Web Speech API TTS
    ↓
[Data loop] SessionRecorder → JSONL → fine-tune → better Gemma coaching
```

The visual feed drives the agent; the agent's fine-tuned model produces coaching; the coaching shapes the next session's fine-tuning data. The components are a closed loop, not independent modules.

---

## Purpose

A fully offline, privacy-first, real-time squat form coaching application.
Runs entirely on the user's device after initial setup — no cloud calls at runtime.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vanilla JS (ES6 modules), HTML5 Canvas, Web Speech API |
| Pose ML | MediaPipe Pose Landmarker Lite (WASM + WebGL, ~4MB, on-device) |
| Backend | Python 3, FastAPI, Uvicorn |
| Optional AI coaching | Gemma 2B via MLX (local inference, WebSocket) |
| Build | Vite 7, npm |
| Package mgmt | pip, npm |

---

## Folder Structure

```bash
squat-coach-local/
├── static/index.html        # Entire frontend (1370 lines, all logic here)
├── static/mediapipe/        # WASM runtime + JS bundle (gitignored, downloaded at setup)
├── models/                  # pose_landmarker_lite.task (gitignored, downloaded at setup)
├── server.py                # FastAPI: serves static files + /ws/coach WebSocket
├── setup.sh                 # One-time: download MediaPipe WASM + model
├── run.sh                   # npm install → setup.sh → python3 server.py
├── requirements.txt         # fastapi>=0.115.0, uvicorn>=0.32.0
├── package.json             # @mediapipe/tasks-vision, vite
├── generate_synthetic_data.py  # Creates JSONL training data for Gemma
└── benchmark_gemma.py       # Evaluates rules-based vs. Gemma coaching quality
```

---

## Architecture

### Signal Pipeline (browser)

```text
Camera → MediaPipe WASM (5–15ms/frame)
       → Confidence gate (drop < 0.5)
       → EMA smoother (α=0.7)
       → Angle computation (knee, hip, torso)
       → Phase detector (state machine: Standing→Descending→Bottom→Ascending)
       → Form checks (depth, valgus, lean)
       → Decision gate (250ms persistence)
       → Feedback engine (visual overlay + speech synthesis)
       → [Optional] GemmaClient via WebSocket → server.py → Gemma 2B
```

### Key Frontend Classes (`static/index.html`)

| Class | Responsibility |
|-------|---------------|
| `LandmarkSmoother` | EMA noise reduction on pose coordinates |
| `PhaseDetector` | State machine for squat phase transitions |
| `FormChecker` | Rule-based checks: depth, knee valgus, torso lean |
| `SetupGate` | Pre-workout camera framing validator |
| `DecisionGate` | Temporal gating to prevent flickering feedback |
| `FeedbackEngine` | Cooldown-managed visual + audio feedback |
| `GemmaClient` | WebSocket client for optional Gemma coaching |
| `SessionRecorder` | Frame-level data capture (JSONL for training) |
| `Renderer` | Canvas skeleton + angle label drawing |

### Backend (`server.py`)

- `GET /` → serve `static/index.html`
- `GET /models/{filename}` → pose model with correct MIME type
- `GET /mediapipe/**` → WASM runtime files (aggressive caching headers)
- `WS /ws/coach` → WebSocket for Gemma coaching (currently echoes placeholder)

---

## Key CONFIG Constants (`static/index.html`)

```javascript
CONFIG = {
  confidenceThreshold: 0.5,    // Min pose confidence
  emaAlpha: 0.7,               // EMA smoothing strength
  standingAngle: 155,          // Knee angle (degrees) when standing
  bottomAngle: 110,            // Knee angle at full squat
  depthAngle: 100,             // Max knee angle for "good depth"
  torsoLeanAngle: 45,          // Max forward lean tolerated
  valgusRatio: 0.82,           // Min knee-to-ankle width ratio
  decisionPersistMs: 250,      // Decision gate window
  feedbackCooldownMs: 2000,    // Min ms between feedback messages
  audioCooldownMs: 3000,       // Min ms between audio cues
  setupReadyMs: 1500,          // Hold-good-framing time to unlock
  minBodyFill: 0.55,           // Min body height in frame
  maxBodyFill: 0.9,            // Max body height in frame
}
```

---

## Session Recording Schema

```json
{
  "session_id": "session-<ISO>",
  "camera_angle": "45deg",
  "camera_height": "hip",
  "frames": [{
    "timestamp_ms": 12345,
    "camera_angle": "45deg",
    "camera_height": "hip",
    "gate_pass": true,
    "confidence_avg": 0.85,
    "confidence_min": 0.71,
    "knee_angle": 98,
    "knee_angle_left": 97,
    "knee_angle_right": 99,
    "hip_angle": 72,
    "torso_angle": 34,
    "knee_valgus_ratio": 0.9,
    "left_right_knee_diff": 2.0,
    "phase": "bottom",
    "rep_number": 1,
    "rep_complete": false,
    "setup_state": {}
  }],
  "rep_summaries": [{ "rep_number": 1, "bottom_frame": {} }]
}
```

---

## Form Checks

| Check | Criterion |
|-------|-----------|
| Depth | knee_angle ≤ 100° at bottom phase |
| Knee valgus | knee-to-ankle width ratio ≥ 0.82 |
| Torso lean | forward lean ≤ 45° |

---

## Training Data & Evaluation

**`generate_synthetic_data.py`** — produces JSONL with 6 labeled case types:
good rep, shallow depth, knee valgus, forward lean, cropped feet, bad side view.

**`benchmark_gemma.py`** — token-overlap scoring comparing rule engine vs. Gemma coaching responses.

---

## Current Status

- **Branch**: `feat/finetune-pose-detection`
- **Main branch**: `main`
- **Active work**: Fine-tuning pose detection / Gemma integration
- **Gemma WebSocket**: server-side stub only — real MLX inference not yet wired up

---

## Design Principles

1. **Offline-first** — all inference on device, no video uploaded anywhere
2. **Low latency** — 5–15ms/frame targeting 5–15fps viable on consumer hardware
3. **Graceful degradation** — full functionality with hardcoded rules; Gemma is additive
4. **Modular** — signal processing (JS) is decoupled from coaching logic (optional Python/Gemma)
5. **Data-driven** — SessionRecorder captures ground truth for future model fine-tuning
