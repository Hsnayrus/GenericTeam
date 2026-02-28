# CLAUDE.md — Squat Coach

## Project Overview

Squat Coach is a **fully local, offline** real-time squat form coaching application. Pose estimation runs in-browser via MediaPipe WASM; a local Python server serves static files. The optional Gemma 2B integration (via WebSocket) is a future enhancement — currently the coaching logic is rule-based JavaScript.

## Architecture

```
Webcam → MediaPipe Pose Lite (WASM/WebGL, browser)
       → EMA Smoothing → Angle Computation
       → Phase Detection (state machine)
       → Form Checks → Decision Gating
       → Feedback Engine (visual + Web Speech API)
       → [Future] WebSocket → Python Server → Gemma 2B (MLX)
```

## Key Files

| File | Purpose |
|---|---|
| `static/index.html` | Entire frontend SPA (~1000+ lines). All pose logic, UI, audio, session recording. |
| `server.py` | FastAPI server. Serves static files + models. Has WS endpoint `/ws/coach` (stub). |
| `setup.sh` | One-time setup: copies MediaPipe WASM from node_modules, downloads pose model. |
| `run.sh` | Entry point: `npm install && bash setup.sh && python3 server.py` |
| `generate_synthetic_data.py` | Generates JSONL training data for 6 form-issue categories. |
| `benchmark_gemma.py` | Benchmarks rule-based vs LLM coaching using token overlap score. |
| `requirements.txt` | Python deps: `fastapi>=0.115.0`, `uvicorn>=0.32.0` |
| `package.json` | JS deps: `@mediapipe/tasks-vision@0.10.32`, `vite@7.3.1` |

## Frontend JS Classes (in `static/index.html`)

- **`LandmarkSmoother`** — EMA filter (α=0.7) on pose landmark coordinates
- **`DecisionGate`** — 250ms persistence filter; prevents transient alerts
- **`SessionRecorder`** — Records frames + rep summaries; exports JSON for fine-tuning
- **`SetupGate`** — Validates camera framing (head/feet visible, distance, centering, side view)
- **`PhaseDetector`** — State machine: `standing → descent → bottom → ascent`
- **`FormChecker`** — Rules: depth (knee > 100°), valgus (knee/ankle ratio < 0.82), lean (torso > 45°)
- **`FeedbackEngine`** — Manages cooldowns (2s visual, 3s audio), speech synthesis

## Key Constants

| Constant | Value | Location |
|---|---|---|
| Pose confidence threshold | 0.5 | `index.html` |
| EMA alpha | 0.7 | `LandmarkSmoother` |
| Decision gate persistence | 250ms | `DecisionGate` |
| Feedback cooldown | 2000ms | `FeedbackEngine` |
| Audio cooldown | 3000ms | `FeedbackEngine` |
| Body fill range | 55%–90% of frame | `SetupGate` |
| Standing knee angle | > 155° | `PhaseDetector` |
| Bottom knee angle | ≤ 110° | `PhaseDetector` |
| Shallow depth threshold | knee > 100° at bottom | `FormChecker` |
| Valgus threshold | knee/ankle width < 0.82 | `FormChecker` |
| Lean threshold | torso angle > 45° | `FormChecker` |
| Server port | 8420 | `server.py` |

## Pose Landmarks Used

MediaPipe landmark indices:

- Shoulders: 11, 12
- Hips: 23, 24
- Knees: 25, 26
- Ankles: 27, 28

Angles are **bilaterally averaged** (left + right sides).

## WebSocket Protocol (`/ws/coach`)

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
  "feedback": "Depth degrading over last 4 reps — fatigue pattern.",
  "severity": "warn",
  "speak": true
}
```

Severity levels: `good` (green), `warn` (yellow), `bad` (red), `info` (cyan).

## Development Conventions

- **No external CDN at runtime** — all assets must be served locally by `server.py`
- **No build step required** — `index.html` is a self-contained SPA; edit and refresh
- **Python virtual environment** is at `.venv/` — activate before running scripts
- **Models and MediaPipe assets are gitignored** — run `setup.sh` to restore them
- **Do not add network requests** in the browser frontend — offline-first is a core constraint
- **Feedback severity** must be one of: `good`, `warn`, `bad`, `info`
- **Angle computation** must remain bilaterally averaged for consistency

## Running Locally

```bash
npm install
bash setup.sh      # one-time; downloads ~7MB of assets
python3 server.py  # starts at http://localhost:8420
```

## Synthetic Data Format (JSONL)

Each record:

```json
{
  "id": "good_rep-0000",
  "label": "good_rep",
  "input": {
    "phase": "bottom",
    "rep_number": 4,
    "knee_angle": 95.2,
    "hip_angle": 72.1,
    "torso_angle": 28.3,
    "knee_valgus_ratio": 0.95,
    "confidence_avg": 0.84,
    "confidence_min": 0.71,
    "setup_state": {}
  },
  "expected_feedback": "Good depth. Keep going."
}
```

Labels: `good_rep`, `shallow_depth`, `knee_valgus`, `forward_lean`, `cropped_feet`, `bad_side_view`

## Git Branches

- `main` — stable base
- `feat/finetune-pose-detection` — current working branch
- `feat/define-problem-statement`, `feat/setup-terraform-repos` — planning branches
- `dev` — integration branch

## Gemma 2B Integration (Planned)

- Backend: MLX (Apple Silicon local inference)
- Entry point: WebSocket `/ws/coach` in `server.py`
- Fine-tuning data: output of `generate_synthetic_data.py`
- Goal: replace hardcoded `FormChecker` rules with learned patterns

## Rule: always use qmd before reading files

Before reading files or exploring directories, always use qmd to search for information in local projects.

Available tools:

- `qmd search “query”` — fast keyword search (BM25)

- `qmd query “query”` — hybrid search with reranking (best quality)

- `qmd vsearch “query”` — semantic vector search

- `qmd get <file>` — retrieve a specific document

Use qmd search for quick lookups and qmd query for complex questions.

Use Read/Glob only if qmd doesn’t return enough results.

Once this is in place, Claude will always search the index first. It will only fall back to reading full files when it genuinely can’t find what it needs through the

index.

— -

The three search modes explained

qmd gives you three ways to search, each with different tradeoffs:

search — BM25 keyword search. The fastest option. Works great when you know the exact terms you’re looking for. “auth middleware”, “checkout handler”, “database

connection”. Use this 80% of the time.

vsearch — Vector similarity search. Uses embeddings to find conceptually related content even without exact keyword matches. Ask “how does the app handle errors?” and

it’ll find your error boundary component even if it’s called FallbackUI.tsx. Great for exploratory questions.

query — Hybrid search with query expansion and LLM reranking. Combines both approaches and re-ranks results for maximum accuracy. Slower, but the best quality. Use this

for complex questions where you need the most relevant results.

— -
