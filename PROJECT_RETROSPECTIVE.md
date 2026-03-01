# Squat Coach Retrospective

**Branch:** `codex/squat-coach-auto-setup-teacher-v1`  
**Last updated:** 2026-03-01

## Goal

Build a squat coach that runs locally end to end:
- webcam input
- on-device pose estimation
- deterministic squat-state extraction
- short realtime coaching cues from a local model

This document summarizes what we tried, what worked, what did not work, and what currently looks like the right path forward.

## What We Tried

### 1. Browser-first realtime pipeline

We built the live app around:
- MediaPipe Pose in the browser
- deterministic landmark filtering and angle extraction
- setup gating before live coaching
- rep counting and session export

Relevant files:
- [`static/index.html`](/Users/jwalinshah/projects/squat-coach-local/static/index.html)
- [`server.py`](/Users/jwalinshah/projects/squat-coach-local/server.py)
- [`README.md`](/Users/jwalinshah/projects/squat-coach-local/README.md)

### 2. Setup coach separated from live coach

We explicitly split setup/framing from rep coaching. The setup layer checks whether the user is visible, centered, at a usable distance, and in side view before enabling live rep logic.

This reduced the chance of the model trying to coach off bad input, and it matches the product direction in the repo.

### 3. Teacher-assisted labeling and review

We added an offline workflow to review sessions and generate labels using Gemini-based teacher tooling.

Relevant scripts:
- [`batch_video_vision_review.py`](/Users/jwalinshah/projects/squat-coach-local/batch_video_vision_review.py)
- [`generate_coaching_labels_with_gemini.py`](/Users/jwalinshah/projects/squat-coach-local/generate_coaching_labels_with_gemini.py)
- [`judge_model_outputs_with_gemini.py`](/Users/jwalinshah/projects/squat-coach-local/judge_model_outputs_with_gemini.py)
- [`calibrate_mediapipe_with_gemini.py`](/Users/jwalinshah/projects/squat-coach-local/calibrate_mediapipe_with_gemini.py)

### 4. Offline derivation and evaluation tooling

We built scripts to derive reps from exported pose trajectories, compare model outputs, and score prompt/model variants against teacher-labeled data.

Relevant scripts:
- [`derive_reps_offline.py`](/Users/jwalinshah/projects/squat-coach-local/derive_reps_offline.py)
- [`evaluate_coach_dataset.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_coach_dataset.py)
- [`evaluate_prompt_variants.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_prompt_variants.py)
- [`evaluate_guardrailed_ollama.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_guardrailed_ollama.py)
- [`compare_session_models.py`](/Users/jwalinshah/projects/squat-coach-local/compare_session_models.py)

### 5. Structured prompting plus runtime guardrails

Instead of free-form coach text, we moved toward:
- explicit JSON schemas
- centralized prompt templates
- short cue constraints
- deterministic-ish decoding settings
- sanitization/guardrails after generation

Relevant files:
- [`prompt_templates.py`](/Users/jwalinshah/projects/squat-coach-local/prompt_templates.py)
- [`GEMMA_PROMPTING.md`](/Users/jwalinshah/projects/squat-coach-local/GEMMA_PROMPTING.md)
- [`server.py`](/Users/jwalinshah/projects/squat-coach-local/server.py)

### 6. Multiple local inference paths

We kept Ollama as the main local runtime and also added an MLX demo path.

This was an attempt to keep the project flexible between:
- direct local chat inference
- lightweight local evaluation
- eventually running a fine-tuned local student model

## What Worked

### The core pose pipeline works

The most important validation from the calibration work is that MediaPipe was still producing usable pose signals:
- knee angles moved through realistic squat ranges
- hip angles tracked
- hip-below-knee state could be derived
- confidence was often good enough to use

This matters because it means the main issue was not "pose tracking is useless." The main issue was downstream gating and state logic.

Source:
- [`CALIBRATION_FINDINGS.md`](/Users/jwalinshah/projects/squat-coach-local/CALIBRATION_FINDINGS.md)

### Offline rep derivation worked

When we bypassed the live gating problems and analyzed the raw pose trajectory offline, rep detection recovered meaningful squat counts.

That gave us two important signals:
- the exported session data is valuable
- offline analysis can rescue sessions that fail live detection

### Auto-setup and tracking handoff are the right product shape

The repo direction is now clearly:
- setup coach first
- live side-view cues second
- optional set summary after

That separation is sound and should remain. It reduces ambiguity in both the UX and the model prompt space.

### Centralized prompt templates are better than scattered prompt strings

Moving prompt policy into [`prompt_templates.py`](/Users/jwalinshah/projects/squat-coach-local/prompt_templates.py) is a good architectural improvement.

Why it worked:
- one schema definition instead of drift across scripts
- easier prompt iteration
- easier evaluation across the same policy
- easier sanitization and fallback logic

### Runtime guardrails and cadence controls are necessary

The current server now has:
- schema-constrained output
- canonical cue normalization
- repeat cooldown
- minimum interval gating
- event logging

That is the correct direction for realtime coaching. Even a decent model will become annoying if cue cadence is uncontrolled.

### Teacher + evaluator workflow is useful

The repo now has a usable loop for:
1. collect session data
2. generate or review labels
3. compare prompt/model variants
4. inspect raw vs sanitized behavior

That is much stronger than trying to fine-tune blindly.

## What Did Not Work

### The original feet visibility gate broke live squat detection

This is the clearest failure documented so far.

The live system treated ankle landmark visibility as a hard setup requirement. In real recordings, that was wrong often enough that setup stayed locked, rep mode never activated correctly, and valid squats were effectively ignored.

Observed impact from the calibration write-up:
- 27 offline reps across 5 sessions
- only 1 live rep detected across those same sessions
- many frames incorrectly marked as `feet_not_visible`

Root cause:
- ankle visibility is too fragile for a hard gate
- clothing, shoes, floor contrast, and camera angle can all break it

Source:
- [`CALIBRATION_FINDINGS.md`](/Users/jwalinshah/projects/squat-coach-local/CALIBRATION_FINDINGS.md)

### Phase detection looked broken in practice because it was blocked upstream

This is an important nuance. The state machine looked non-functional because the gating logic prevented it from running under realistic conditions.

The lesson is that calibration has to inspect the whole chain:
- visibility checks
- setup lock
- rep-mode activation
- phase transitions

### Raw local model output is not enough by itself

The current codebase assumes this and adds explicit sanitization plus deterministic fallbacks. That is the right conclusion.

What did not work well enough on its own:
- unconstrained phrasing
- trusting the first local model output
- relying on prompt wording alone without runtime checks

The existence of [`evaluate_guardrailed_ollama.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_guardrailed_ollama.py) is a sign that raw model behavior needed cleanup.

### Direct adapter portability across runtimes is not solved yet

The current README notes that Hugging Face LoRA adapters are not directly consumed by `mlx-lm` without conversion or a merged MLX model.

So the "train once, run anywhere locally" path is not complete yet.

### Fine-tuned local student model is still unfinished

The repo has training and dataset prep scripts, but the README still lists the fine-tuned Gemma student model and an audited real-session dataset as not done yet.

That means the project has strong scaffolding, but the final model stage is still incomplete.

## What We Learned

### 1. Structured state is the right interface

The project should keep feeding the coach model compact squat state, not raw landmarks and not raw video, during the live path.

### 2. Hard gates must be extremely conservative

If a setup check is noisy, it should not block the entire live pipeline. Hard blockers should be reserved for conditions that are reliably measurable.

### 3. Offline analysis is not optional

It is necessary for:
- calibration
- recovering failed live sessions
- generating cleaner training labels
- debugging whether the problem is pose, logic, or prompting

### 4. Prompting alone will not make the product reliable

Prompting helps, but the stable system comes from:
- deterministic feature extraction
- schema validation
- sanitization
- cooldown logic
- clear priority ordering

### 5. The current best path is hybrid, not pure model-only

The repo already points to the right architecture:
- deterministic setup and safety gating
- local model for short phrasing and prioritization
- offline teacher/eval loop for improvement

## Current Best-Known Direction

Based on the current repo state, the strongest path forward is:

1. Keep the browser pose pipeline and session export flow.
2. Keep setup coaching separate from live rep coaching.
3. Continue using structured-state prompts with strict schemas.
4. Treat local model output as assistive, not authoritative.
5. Use offline rep derivation and teacher review to audit every major pipeline change.
6. Finish dataset cleanup before spending heavily on fine-tuning.
7. Validate any new setup blocker against real session recordings before making it hard-gating.

## Open Risks

- Other setup checks may still be too strict, even after removing feet as a blocker.
- Real-session label quality may still be noisier than the training scripts assume.
- Local runtime portability between Ollama, MLX, and future fine-tuned adapters still needs cleanup.
- The student-model path is not yet validated well enough to replace deterministic fallback behavior.

## Supporting Docs

- [`README.md`](/Users/jwalinshah/projects/squat-coach-local/README.md)
- [`CALIBRATION_FINDINGS.md`](/Users/jwalinshah/projects/squat-coach-local/CALIBRATION_FINDINGS.md)
- [`GEMMA_PROMPTING.md`](/Users/jwalinshah/projects/squat-coach-local/GEMMA_PROMPTING.md)
