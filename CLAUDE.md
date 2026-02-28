# CLAUDE.md — Squat Coach

## Hackathon Context

This project is a submission for an 8-hour hackathon. Judges evaluate on four required components:

| Requirement | How this project satisfies it |
|---|---|
| **Fine-tuned on-device model** | Gemma 2B (MLX, Apple Silicon) fine-tuned on squat coaching JSONL via `generate_synthetic_data.py` |
| **Agentic behavior** | Autonomous monitor-assess-act loop: phase detector + form checker → coaching decision → speech output |
| **Visual input** | Webcam → MediaPipe Pose WASM → real-time landmark tracking |
| **Genuine on-device reason** | Sub-50ms feedback latency (impossible via cloud), workout video never leaves the device, offline operation |
| **Voice (bonus)** | Web Speech API audio cues on form errors |

**Key judge criteria:** Components must reinforce each other as a system, not sit side-by-side. Fine-tuning must make the coaching noticeably better. Demo must run live.

**Time constraint:** LoRA fine-tuning on Gemma 2B targets <1 hour. Core demo is webcam → pose → rule coaching; Gemma is the differentiator.

---

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

## Rule: Architecture Reasoning Before Every Change

**Before writing any code, you MUST trace the change through the full system.**

This project has two tightly coupled layers:

| Layer | Files |
|---|---|
| **Backend** | `server.py`, `generate_synthetic_data.py`, `benchmark_gemma.py` |
| **Frontend** | `static/index.html` (all JS classes, UI, WebSocket client) |
| **Protocol** | WebSocket `/ws/coach` — shared JSON contract between the two layers |

**Required checklist for every change:**

1. **Identify the primary file** being changed.
2. **Determine which layer** it belongs to (backend / frontend / protocol).
3. **Check cross-layer impact:**
   - Changing `server.py` WebSocket output → does `index.html` need to handle new/changed fields?
   - Changing `index.html` WebSocket payload → does `server.py` need to parse new fields?
   - Changing a severity level, field name, or message format → update both sides.
   - Adding a new Python endpoint or route → does the frontend need to call it?
   - Changing constants (thresholds, port, cooldowns) → verify the other layer doesn't hard-code the same value.
4. **State the impact explicitly** in your response before making edits: "This change affects backend only / frontend only / both layers."
5. **Make all necessary edits** in the same response — never leave a cross-layer change half-done.

**Never assume a change is isolated.** When in doubt, search the other layer's file for the symbol, field name, or constant you're modifying.

---

## Development Conventions

- **No external CDN at runtime** — all assets must be served locally by `server.py`
- **No build step required** — `index.html` is a self-contained SPA; edit and refresh
- **Python virtual environment** is at `.venv/` — activate before running scripts
- **Models and MediaPipe assets are gitignored** — run `setup.sh` to restore them
- **Do not add network requests** in the browser frontend — offline-first is a core constraint
- **Feedback severity** must be one of: `good`, `warn`, `bad`, `info`
- **Angle computation** must remain bilaterally averaged for consistency

## Logging Conventions

All Python code must use **[Loguru](https://github.com/Delgan/loguru)** for structured, production-grade logging. Never use `print()` or the standard `logging` module for diagnostics.

**Setup (module-level import only — no configuration needed per module):**

```python
from loguru import logger
```

Configure sinks once at application entry point (`server.py`):

```python
from loguru import logger
import sys

logger.remove()  # remove default stderr sink
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}")
logger.add("logs/squat_coach.log", rotation="10 MB", retention="7 days", level="DEBUG", serialize=True)
```

**Log levels — use the correct level:**

| Level | When to use |
|---|---|
| `DEBUG` | Fine-grained trace info, intermediate values, loop iterations |
| `INFO` | Key lifecycle events: server start, WebSocket connect/disconnect, rep completed |
| `WARNING` | Recoverable anomalies: low pose confidence, missing landmarks, skipped frames |
| `ERROR` | Non-fatal failures: coaching inference error, bad WebSocket message |
| `CRITICAL` | System-level failures: model load failure, port bind failure |

**Use f-strings (Loguru evaluates lazily by default):**

```python
# Correct — Loguru defers evaluation until the message is actually logged
logger.info(f"Rep {rep} completed: phase={phase}, knee_angle={knee_angle:.1f}")

# Also correct — structured key=value binding for machine-readable logs
logger.bind(rep=rep, phase=phase, knee_angle=knee_angle).info("Rep completed")
```

**Always include contextual fields** when logging coaching events:

- `phase` — current squat phase
- `rep_number` — rep count at time of log
- Relevant angle/ratio values when a form check fires

Use `logger.bind(...)` to attach structured context that persists across a call chain.

**Never log:** raw video frames, webcam buffers, or any data that could reconstruct user biometrics beyond what's needed for debugging.

## Docstring Conventions

All Python functions, classes, and public methods must have **NumPy-style docstrings**. This applies to `server.py`, `generate_synthetic_data.py`, `benchmark_gemma.py`, and any new Python modules.

**Function/method template:**

```python
def compute_angle(a: tuple, b: tuple, c: tuple) -> float:
    """Compute the interior angle at vertex b formed by points a, b, c.

    Uses the law of cosines on the three 2-D landmark coordinates.
    Returns a value in [0, 180] degrees.

    Parameters
    ----------
    a : tuple of float
        (x, y) coordinates of the first endpoint landmark.
    b : tuple of float
        (x, y) coordinates of the vertex landmark (angle is measured here).
    c : tuple of float
        (x, y) coordinates of the second endpoint landmark.

    Returns
    -------
    float
        Interior angle in degrees, clamped to [0, 180].

    Raises
    ------
    ValueError
        If any coordinate tuple does not have exactly two elements.
    """
```

**Class template:**

```python
class PhaseDetector:
    """State-machine phase detector for squat rep segmentation.

    Transitions between `standing`, `descent`, `bottom`, and `ascent`
    states based on bilaterally averaged knee angle thresholds.

    Parameters
    ----------
    standing_threshold : float, optional
        Minimum knee angle (degrees) to classify as standing. Default 155.
    bottom_threshold : float, optional
        Maximum knee angle (degrees) to classify as bottom. Default 110.

    Attributes
    ----------
    state : str
        Current phase: one of ``standing``, ``descent``, ``bottom``, ``ascent``.
    rep_count : int
        Number of completed full reps since initialisation.
    """
```

**Rules:**

- One-line summary on the first line, separated from the body by a blank line.
- `Parameters`, `Returns`, `Raises`, and `Attributes` sections use the NumPy dashed-underline style.
- Types go in the section header (e.g., `float`, `str`, `list of dict`), not inline in the summary.
- Omit sections that do not apply (e.g., no `Raises` if the function cannot raise).
- Private helpers (single leading underscore) may use a single-line docstring when the name is self-explanatory.

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
