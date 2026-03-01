# Gemma Prompting Guide

This repo now keeps Gemma prompt policy in [`prompt_templates.py`](/Users/jwalinshah/projects/squat-coach-local/prompt_templates.py).

## Goals

- keep Gemma outputs short, parseable, and consistent
- make policy changes in one place
- evaluate prompt changes against the same schema everywhere

## Current Template Strategy

Use one user-turn prompt that includes:

- the role: local squat coach
- the exact JSON schema
- policy ordering for priorities
- short-output constraints
- a few high-signal examples
- one final `INPUT` block followed by `OUTPUT`

This is intentional. Gemma generally behaves better when the task, schema, and examples are all in the same prompt instead of split across many loosely related instructions.

## Best Practices For This Repo

1. Keep the output space narrow.
   Prefer enums like `priority` and `ui.highlight` over open-ended text.

2. Keep the cue short.
   `say` should stay under 14 words. Long coaching text drifts and hurts latency.

3. Separate policy from language.
   The template forces a single priority first, then one cue.

4. Use structured state only.
   Feed phase, angles, quality, recent events, and trends. Avoid prose unless there is no structured equivalent.

5. Use few-shot examples sparingly.
   Add examples only for important edge cases:
   - framing failures
   - low-quality tracking
   - torso safety faults
   - shallow depth
   - knee tracking faults
   - clean reps

6. Validate everything after generation.
   Reject outputs that fail schema, exceed length, or invent labels.

7. Fall back to rules when invalid.
   Prompting improves behavior, but runtime safety still comes from validation and fallback logic.

## Where To Edit

- Schema and examples: [`prompt_templates.py`](/Users/jwalinshah/projects/squat-coach-local/prompt_templates.py)
- Evaluation harness: [`evaluate_coach_dataset.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_coach_dataset.py)
- Session comparison: [`compare_session_models.py`](/Users/jwalinshah/projects/squat-coach-local/compare_session_models.py)
- Batch pose review: [`batch_pose_label_sessions.py`](/Users/jwalinshah/projects/squat-coach-local/batch_pose_label_sessions.py)

## Recommended Iteration Loop

1. Edit the examples or policy in [`prompt_templates.py`](/Users/jwalinshah/projects/squat-coach-local/prompt_templates.py).
2. Run [`evaluate_coach_dataset.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_coach_dataset.py) against the sample dataset.
3. Run [`evaluate_prompt_variants.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_prompt_variants.py) to compare prompt variants such as `default` vs `no_examples`.
4. Run [`compare_session_models.py`](/Users/jwalinshah/projects/squat-coach-local/compare_session_models.py) on a few real reps.
5. Run [`evaluate_guardrailed_ollama.py`](/Users/jwalinshah/projects/squat-coach-local/evaluate_guardrailed_ollama.py) to compare raw vs sanitized outputs and measure guardrail intervention rate.
6. Check priority accuracy first, then cue quality.
7. Only fine-tune after the prompt/template stops improving.
