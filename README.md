# Squat Coach — On-Device Visual Coaching Agent

A realtime squat coach designed to run on-device end to end:
- visual input from the webcam
- local pose estimation
- deterministic squat-state extraction
- a fine-tuned local Gemma model for coaching and agentic decisions

The target demo is not just "AI text over pose data." The target demo is:
- visual input
- on-device reasoning
- private realtime feedback
- agentic behavior across setup, calibration, live cues, and set summaries

## Product Direction

The current design direction is:

1. `setup_coach`
- Separate from live coaching
- Handles framing/readiness only
- Example issues: head not visible, feet not visible, off-center

2. `live_cue_side`
- Side-view only
- Uses a compact structured state instead of raw landmarks
- Priorities: `safety`, `depth`, `knees`, `encouragement`

3. `set_summary`
- Optional higher-level summary after a set
- Uses rep trends and fatigue metrics

## High-Level Architecture

```
Webcam
   ↓
MediaPipe Pose Lite (on-device)
   ↓
Landmark Filtering + Confidence Gating
   ↓
Deterministic State Extraction
   ↓
Compact Squat State
   ↓
Local Gemma (fine-tuned)
   ↓
Agentic Coaching Output
```

Where each layer fits:
- MediaPipe = eyes
- State extraction = biomechanics interpreter
- Gemma = coaching brain
- Agent loop = decides what to do next

## Why On-Device

- Privacy: workout video never leaves the machine
- Latency: mid-rep feedback is only useful if it is local and fast
- Reliability: works offline in a garage, basement, or gym
- Cost: no per-inference API calls at runtime
- Personalization: local model + session calibration can adapt to the user

## What Runs Where

| Component | Runs in | Notes |
|---|---|---|
| Pose estimation | Browser (WASM+WebGL) | MediaPipe Pose |
| State extraction | Browser (JS) | Angles, phase, rep metrics, setup gate |
| Recording/export | Browser (JS) | Exports session JSON today |
| Local coaching model | Local runtime / VM | Gemma via MLX (in-process, Apple Silicon) |
| UI + audio | Browser | Overlay + speech |

Recommended split:
- local machine: browser or local app for webcam capture and UI
- local or VM runtime: model inference, data processing, and evals
- no internet dependency after models and assets are installed

For the current project split:
- [local app integration](/Users/jwalinshah/projects/squat-coach-local/README.md): browser recorder, UI, and `/ws/coach` integration work
- offline training and eval box: a local machine or VM with Ollama, exported JSON, and recorded `.webm` sessions

## Current Status

Working now:
- local webcam pose pipeline
- setup gate
- automatic setup-to-tracking handoff
- rep counting
- angle computation
- session JSON + local video export
- Gemini teacher dataset generation
- evaluation harness for the new teacher schema

Not done yet:
- final side-view feature schema cleanup in the recorder
- fine-tuned Gemma student model
- audited labeled dataset from real sessions

Current collection requirement:
- use a floor-mounted side-view camera setup
- the app now auto-arms tracking when framing is good enough

## Quick Start

```bash
# 1. Clone and enter directory
cd squat-coach-local

# 2. Install Python and JS dependencies
python3 -m pip install -r requirements.txt
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

## MLX Demo Run

If the teammate already has an MLX model directory on disk, point `MODEL` at that path:

```bash
cd GenericTeam
python3 -m pip install -r requirements.txt

MODEL=/absolute/path/to/local-mlx-model \
MLX_ADAPTER_PATH=/absolute/path/to/local-mlx-adapter \
bash run_mlx_demo.sh
```

If you want MLX to pull the model from Hugging Face instead, use the repo id:

```bash
cd GenericTeam
python3 -m pip install -r requirements.txt

MODEL=mlx-community/gemma-3-1b-it-4bit \
bash run_mlx_demo.sh
```

Runtime knobs for the smoother demo:
- `MLX_MAX_TOKENS`
- `COACH_MIN_INTERVAL_S`
- `COACH_REPEAT_COOLDOWN_S`
- `COACH_LOG_PATH`

What is logged:
- every live snapshot
- final emitted cue
- whether setup gating or runtime suppression changed the result

The JSONL log lands at `logs/coach_events.jsonl`.

## Branch Split

Use these branches for separation of concerns:
- `codex/squat-coach-integration-v1` for app and pipeline integration
- `train/squat-coach-v1` for VM-only data cleaning, evals, and fine-tuning work

Suggested flow:
1. land browser and pipeline changes on `codex/squat-coach-integration-v1`
2. fetch that branch on the VM after it is pushed
3. branch `train/squat-coach-v1` from it for offline experiments and generated artifacts

## Locked Baseline Thresholds

The current baseline assumes these locked live-coaching thresholds:
- depth target: `knee_angle <= 100`
- torso warning: `torso_angle >= 45`
- knees in warning: `knee_valgus_ratio < 0.82`

Browser rules, the baseline summary script, and the local websocket fallback should stay aligned with these values unless we intentionally revise the policy.

## VM Setup

Use the VM as the offline training and eval box.

```bash
ssh <your-vm-user>@<your-vm-host>
git clone https://github.com/Hsnayrus/GenericTeam.git
cd GenericTeam
git fetch --all
git checkout codex/squat-coach-integration-v1
git checkout -b train/squat-coach-v1

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
pip install -r requirements.txt
pip install -U google-genai

npm install
```

MLX and the Gemma model weights are installed via `pip install -r requirements.txt`. On first run, `mlx_lm.load()` downloads the model from HuggingFace and caches it locally. No separate server process is needed.

Put exported session assets on the VM under a data folder such as:

```bash
mkdir -p data/raw_videos data/raw_sessions data/results

scp /path/to/session.webm <your-vm-user>@<your-vm-host>:~/GenericTeam/data/raw_videos/
scp /path/to/session.json <your-vm-user>@<your-vm-host>:~/GenericTeam/data/raw_sessions/
```

Set the Gemini teacher key on the VM before running teacher-data generation:

```bash
export GEMINI_API_KEY=YOUR_KEY
```

Persist it in the VM shell profile only if that machine is dedicated to this project.

Recommended teacher model choice:
- default to `gemini-2.5-pro` for stable dataset generation and rep review
- test `gemini-3-pro-preview` on a small audited slice only if you want to compare preview quality

## Recording Data

Right now the app is good enough to start a pilot collection pass.

Workflow:
1. Start the app
2. Put the camera at floor height in a side view
3. Stand in frame and let the setup gate turn green
4. Tracking starts automatically when framing is stable
5. Click `Start Rec`
6. Perform a short set
7. Click `Stop Rec`
8. Click `Export JSON`

What gets saved today:
- per-frame pose-derived metrics
- setup state
- rep summaries
- session metadata
- local recorded video when the browser supports `MediaRecorder`
- full pose stream under `datasets.live_cue_side_all`, including provisional frames, setup flags, and confidence fields for offline debugging

Recommended collection setup:
- floor-mounted camera
- full-body side view
- keep the camera fixed for the full session

## Recommended Data Plan

Use real data first.

Why:
- real camera geometry
- real body proportions
- real tracking noise
- real squat patterns

Use synthetic data later only to fill gaps.

Suggested collection plan:
1. Record clean side-view sets
2. Record intentional shallow reps
3. Record intentional forward-lean reps
4. Record intentional knees-in reps
5. Repeat across multiple users

For labeling:
- use the video to judge the rep
- use the pose JSON to extract the structured training features

Humans should label from video, not from raw pose numbers alone.

## Target Live Feature Schema

The simplified `live_cue_side` dataset should use only these 7 features:

1. `phase`
2. `rep_count`
3. `knee_angle`
4. `hip_angle`
5. `torso_angle`
6. `shin_angle`
7. `hip_below_knee`

Separate setup-only fields:
- `head_visible`
- `feet_visible`
- `centered`
- `gate_pass`

Interpretation:
- `setup_coach` dataset = framing/readiness only
- `live_cue_side` dataset = only rows where `gate_pass = true`

## Calibration Strategy

To keep the system simple and robust:
- use one fixed side-view camera setup
- calibrate per session instead of trying to solve all viewpoint variation

Simple calibration flow:
1. Standing baseline for 2-3 seconds
2. Two or three easy squats
3. Set personalized thresholds for:
- depth target
- torso warning threshold

This matters more than trying to hardcode one ideal squat angle for every body type.

## Agent Behavior

The intended local agent loop is:

1. `setup_coach`
- asks for better framing if needed

2. `calibrate`
- observes a few initial reps
- sets thresholds for the current user/session

3. `live_cue`
- gives one concise actionable cue during the set

4. `set_summary`
- summarizes rep quality and fatigue trends

This is the "agentic" part of the project:
- it does not just classify a frame
- it chooses the next useful coaching action

## Setup Details

`setup.sh` prepares these files once:
- MediaPipe WASM runtime (~3MB) → `static/mediapipe/wasm/`
- MediaPipe JS bundle → `static/mediapipe/vision_bundle.mjs`
- Pose Landmarker Lite model (~4MB) → `models/`
- Python deps: `fastapi`, `uvicorn`

If `@mediapipe/tasks-vision` is already installed locally, the setup script copies the MediaPipe runtime from `node_modules` instead of downloading it again.

After setup, **zero network requests** are made at runtime.

## Camera Setup

Current target:
- floor-mounted side view only for live coaching
- full body visible
- fixed camera position
- consistent distance

The current product direction is no longer "mixed front + side." It is:
- `setup_coach` handles framing
- `live_cue_side` handles squat coaching

## The "Unplug the Internet" Demo

1. Run `setup.sh` with internet
2. Start `python3 server.py`
3. Open `http://localhost:8420`
4. Do a few squats — see real-time coaching
5. **Disconnect WiFi/Ethernet**
6. Do more squats — **identical performance**
7. Point to the HUD: NET shows OFFLINE, everything else green

## Gemma Integration

The WebSocket endpoint at `/ws/coach` exists, but true local Gemma integration is still pending.

MLX demo path:

```bash
source .venv/bin/activate
python -m pip install mlx-lm
COACH_BACKEND=mlx \
MLX_MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit \
bash run_mlx_demo.sh
```

Optional adapter:

```bash
COACH_BACKEND=mlx \
MLX_MODEL=/absolute/path/to/your/mlx-model \
MLX_ADAPTER_PATH=/absolute/path/to/your/mlx-adapter \
bash run_mlx_demo.sh
```

Notes:
- live events are logged to `logs/coach_events.jsonl`
- setup issues are gated before model inference
- cadence and repeat cooldown are enforced in [`server.py`](/Users/jwalinshah/projects/squat-coach-local/server.py)
- the current Hugging Face LoRA adapters need an MLX-compatible adapter path or a merged MLX model; they are not consumed directly by `mlx-lm` unless converted

What the current local runtime already constrains:
- structured JSON output through Ollama schema mode
- deterministic or near-deterministic decoding through server-side options
- short responses with bounded token budget
- explicit severity enum: `good|warn|bad|info`

Prompt-template reference:
- [`GEMMA_PROMPTING.md`](/Users/jwalinshah/projects/squat-coach-local/GEMMA_PROMPTING.md)
- [`prompt_templates.py`](/Users/jwalinshah/projects/squat-coach-local/prompt_templates.py)
- [`evaluate_prompt_variants.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_prompt_variants.py)

Runtime knobs in [`server.py`](/Users/jwalinshah/projects/squat-coach-local/server.py):
- `OLLAMA_TEMPERATURE`
- `OLLAMA_TOP_P`
- `OLLAMA_TOP_K`
- `OLLAMA_MIN_P`
- `OLLAMA_REPEAT_PENALTY`
- `OLLAMA_NUM_PREDICT`
- `OLLAMA_SEED`
- `OLLAMA_STOP` using `||` as the separator for multiple stop strings

Example:

```bash
OLLAMA_MODEL=gemma3:4b \
OLLAMA_TEMPERATURE=0 \
OLLAMA_TOP_P=0.8 \
OLLAMA_REPEAT_PENALTY=1.2 \
OLLAMA_NUM_PREDICT=48 \
OLLAMA_SEED=7 \
python3 server.py
```

Important limitation:
- this stack does not implement per-token logit biasing today
- if you need true token suppression/boosting, grammar-constrained decoding, or custom logits processors, move inference to a backend that exposes token-level generation hooks instead of the current Ollama chat path

Target model role:
- input = compact squat state, not raw pixels
- output = coaching decision and phrasing
- runtime = fully local

Target input:

```json
{
  "phase": "bottom|descending|ascending|standing",
  "rep_count": 4,
  "knee_angle": 108,
  "hip_angle": 72,
  "torso_angle": 41,
  "shin_angle": 28,
  "hip_below_knee": true
}
```

Target output:
```json
{
  "say": "Keep your chest up as you drive out of the bottom.",
  "priority": "safety",
  "ui": {"highlight": "torso", "show_checklist": false},
  "cooldown_s": 3.0
}
```

Why fine-tune Gemma instead of only using rules:
- better phrasing
- better prioritization across multiple metrics
- can reason over rep history and fatigue trends
- fits the on-device agent story for the hackathon

## Teacher / Student Training Flow

Use Gemini offline as the "teacher" to generate a labeled JSONL dataset, then fine-tune Gemma locally as the "student" for realtime coaching. This step is optional and separate from the runtime app.

### Setup

```bash
python3 -m pip install -U google-genai
export GEMINI_API_KEY="YOUR_KEY_HERE"
```

### Generate JSONL training data

```bash
python3 generate_gemini_dataset.py \
  --count 200 \
  --batch-size 50 \
  --output data/gemini_teacher_dataset.jsonl \
  --raw-output data/gemini_teacher_raw.json
```

What the script does:
- Uses the squat-coach teacher prompt and schema described in `generate_gemini_dataset.py`
- Calls Gemini in batches and asks for strict JSON only
- Validates every example before it is accepted
- Writes one JSON object per line for fine-tuning

Current recommendation:
- use real recorded data first
- use Gemini later for language generation and synthetic gap-filling
- do not let Gemini be the source of truth for squat quality

Instead:
- deterministic rubric = truth
- Gemini = language teacher
- Gemma = local student

## Evaluation

The repo now includes a schema-aligned evaluator:

```bash
python3 evaluate_coach_dataset.py --backend gold
python3 evaluate_coach_dataset.py --backend rules
python3 evaluate_coach_dataset.py --backend mlx --model mlx-community/gemma-3-1b-it-4bit
```

What it measures today:
- priority accuracy against labels
- priority accuracy against deterministic policy
- token overlap
- response length constraints

Current takeaway:
- the dataset policy must be cleaned up before fine-tuning
- base local Gemma is not good enough on the current mixed sample
- rules currently beat base Gemma on policy consistency

## Offline Session Workflow

Once session JSON and `.webm` files are on the training machine, use this flow:

### 1. Clean exported session JSON

```bash
python3 clean_offline_session.py \
  --input data/raw_sessions/session.offline.json \
  --output data/results/session.cleaned.json \
  --min-confidence 0.3
```

### 1b. Recover reps offline from the full pose stream

If the live app captured pose frames but failed to emit `rep_summaries`, derive them from the exported pose stream:

```bash
python3 derive_reps_offline.py \
  --input /Users/you/Downloads/side-floor-session-2026-02-28T22-54-09-713Z.json \
  --output-dir data/results/derived_reps
```

This uses `datasets.live_cue_side_all` when available and falls back to raw `frames[].live`.

### 2. Compare local models on one session

```bash
python3 compare_session_models.py \
  --session-json data/results/session.cleaned.json \
  --video data/raw_videos/session.webm \
  --rep 3 \
  --models gemma3:1b gemma3:4b \
  --mode both \
  --output data/results/session_compare.json
```

### 3. Summarize the session baseline against locked thresholds

```bash
python3 summarize_session_baseline.py \
  --input data/results/session.cleaned.json \
  --output data/results/session_baseline.json
```

### 4. Evaluate the structured dataset

```bash
python3 evaluate_coach_dataset.py --backend rules
python3 evaluate_coach_dataset.py --backend mlx --model mlx-community/gemma-3-1b-it-4bit
```

### 5. Generate teacher data only if needed

```bash
export GEMINI_API_KEY=YOUR_KEY
python3 generate_gemini_dataset.py \
  --model gemini-2.5-pro \
  --count 200 \
  --batch-size 50 \
  --output data/gemini_teacher_dataset.jsonl \
  --raw-output data/gemini_teacher_raw.json
```

### 6. Review one rep with Gemini Pro from side-view frames

```bash
python3 review_rep_with_gemini.py \
  --model gemini-2.5-pro \
  --session-json data/results/session.cleaned.json \
  --video data/raw_videos/session.webm \
  --rep 3 \
    --output data/results/gemini_rep_review.json
```

### 7. Batch-review recent sessions with Gemini and local Gemma using pose only

Use this when you want a fast teacher/student comparison on exported sessions without uploading the full videos:

```bash
source .venv/bin/activate

python batch_pose_label_sessions.py \
  --input-glob '/Users/you/Downloads/side-floor-session-*.json' \
  --limit 5 \
  --min-confidence 0.0 \
  --gemini-models models/gemini-2.5-pro models/gemini-3.1-pro-preview \
  --gemma-models gemma3:1b gemma3:4b \
  --output-dir data/results/batch_pose_review
```

This writes cleaned session payloads, baseline summaries, pose-only teacher/student reviews, and per-model latency.

### 8. Use Gemini vision as a separate teacher for pipeline debugging

Use the video teacher only to understand what the pose pipeline missed or distorted:

```bash
source .venv/bin/activate

python batch_video_vision_review.py \
  --video-glob '/Users/you/Downloads/side-floor-session-*.webm' \
  --json-dir /Users/you/Downloads \
  --output-dir data/results/vision_review \
  --model models/gemini-2.5-pro
```

Keep this output separate from the pose-teacher labels:
- pose teacher = coaching policy labels for model training
- vision teacher = MediaPipe / gating / framing error analysis

### Synthetic Side-View Video Generation With Google

Use Veo when you want fake side-view squat clips for demos or offline experiments.

```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"

python3 generate_synthetic_squat_videos.py \
  --count 4 \
  --model veo-3.1-generate-preview \
  --output-dir data/generated_videos
```

This writes:
- generated `.mp4` clips under `data/generated_videos/`
- `data/generated_videos/synthetic_squat_manifest.json` with the exact prompt, synthetic profile, and `mediaprose`

If you want Gemini coaching without sending the videos, review the manifest prose only:

```bash
python3 review_synthetic_squat_profiles_with_gemini.py \
  --manifest data/generated_videos/synthetic_squat_manifest.json \
  --output data/results/synthetic_squat_reviews.json
```

The prose-only review script sends just the structured profile plus `mediaprose` to Gemini. It does not upload the generated videos.

### Synthetic Side-View Keyframe Images With Google

If you mainly need pose-aligned examples, generating side-view keyframes is faster and easier to control than full video.

```bash
export GEMINI_API_KEY="YOUR_KEY_HERE"

python3 generate_synthetic_squat_frames.py \
  --count 3 \
  --model gemini-2.5-flash-image \
  --output-dir data/generated_frames
```

This writes one folder per squat example with standing, descending, bottom, ascending, and lockout images plus:
- `data/generated_frames/synthetic_squat_frames_manifest.json`

You can then review the generated example using only the manifest prose and frame specs:

```bash
python3 review_synthetic_squat_profiles_with_gemini.py \
  --manifest data/generated_frames/synthetic_squat_frames_manifest.json \
  --output data/results/synthetic_squat_frame_reviews.json
```

Equivalent `make` targets:

```bash
make clean-session RAW_SESSION=data/raw_sessions/session.offline.json
make baseline-summary CLEAN_SESSION=data/results/session.cleaned.json
make eval-rules
make eval-mlx
make compare-session CLEAN_SESSION=data/results/session.cleaned.json SESSION_VIDEO=data/raw_videos/session.webm REP=3
make review-rep CLEAN_SESSION=data/results/session.cleaned.json SESSION_VIDEO=data/raw_videos/session.webm REP=3
make teacher-generate
```

## Immediate Next Steps

1. Update the recorder/export pipeline to the final side-view schema
2. Record real side-view sessions
3. Split setup vs live datasets using `gate_pass`
4. Label reps from video
5. Build deterministic truth labels
6. Fine-tune local Gemma on that structured dataset
7. Reconnect the student model to `/ws/coach`

Each JSONL row includes:

```json
{
  "id": "gemini-teacher-00001",
  "source_model": "gemini-3-flash-preview",
  "input": { "...": "structured squat state" },
  "output": { "...": "strict coaching response" }
}
```

Use `--dry-run` to print the exact system instruction and user prompt without calling the API.

## File Structure

```
squat-coach-local/
├── requirements.txt                  # Python runtime deps for app and VM setup
├── Makefile                          # Repeatable local/VM setup, baseline, and eval commands
├── setup.sh                          # One-time dependency download
├── server.py                         # FastAPI local server + Gemma WS endpoint
├── clean_offline_session.py          # Clean exported offline session JSON
├── compare_session_models.py         # Compare rules vs Ollama models on one session
├── evaluate_coach_dataset.py         # Evaluate rules or Ollama against the coaching schema
├── review_rep_with_gemini.py         # Gemini Pro teacher review for one rep from extracted frames
├── summarize_session_baseline.py     # Quantify one cleaned session against locked thresholds
├── generate_synthetic_data.py        # Local synthetic benchmark dataset
├── generate_gemini_dataset.py        # Gemini teacher -> JSONL fine-tuning dataset
├── benchmark_gemma.py                # Rules vs model benchmark harness
├── static/
│   ├── index.html                    # Full app (pose + pipeline + UI)
│   └── mediapipe/
│       ├── vision_bundle.mjs         # MediaPipe JS (local)
│       └── wasm/                     # WASM runtime (local)
├── models/
│   └── pose_landmarker_lite.task     # Pose model (local)
└── README.md
```
