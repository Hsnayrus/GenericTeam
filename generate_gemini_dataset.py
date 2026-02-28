#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from google import genai
except ImportError:  # pragma: no cover - import path depends on optional dependency
    genai = None


SYSTEM_INSTRUCTION = """You are a data generator for fine-tuning a local coaching model (Gemma) for real-time squat feedback.
Your job: create training examples that map structured sensor state -> a structured coaching response.

CRITICAL OUTPUT RULES:
- Output MUST be valid JSON only (no markdown, no commentary).
- Output MUST be a JSON object with one key: "examples".
- "examples" MUST be an array of N items.
- Each item MUST match the schema exactly and include realistic values.
- "output.say" must be a single sentence, <= 18 words, no emojis.
- Never mention "confidence scores" or internal variable names in "say".
- If input quality is poor, prioritize setup/framing instructions over form coaching.
- If no issues, give short encouragement.

PRIORITY ORDER (choose the highest priority issue present):
1) visibility/framing (feet/head missing, too close/far, off-center, wrong view)
2) low tracking quality (avg_kpt_conf low, high jitter)
3) safety (excessive forward lean)
4) form (depth, knees in)
5) encouragement

TASKS:
- setup_coach: user not ready; fix framing and view
- live_cue: user in a rep; one actionable cue relevant to phase
- calibrate: suggest a small bounded calibration_patch
- set_summary: 2–3 sentences + one focus tip

Do not use tempo as a coaching priority in this dataset.
Prefer depth, knees, or torso safety over timing commentary.

ALLOWED calibration_patch bounds:
- depth_target_knee_deg: 80..115
- torso_warn_deg: 25..60
- stance_width_norm: 0.20..0.60"""


USER_PROMPT_TEMPLATE = """Generate N={count} training examples in JSON.

Use these distributions:
- tasks: 35% setup_coach, 45% live_cue, 10% calibrate, 10% set_summary
- view_target: 70% side, 30% front
- avg_kpt_conf: uniform 0.35..0.95
- include at least 10 cases with missing ankles or missing head
- include at least 10 cases with off_center left/right
- include at least 10 cases where shallow is detected at bottom phase
- include at least 10 cases where valgus (knees in) persists during descent
- include at least 6 cases with excessive torso lean (>= 45 deg) with good confidence

Input schema (each example.input):
{{
  "mode": "setup|live|calibrate|summary",
  "fps": 15|30,
  "device": "mac_webcam",
  "view_target": "side|front",
  "quality": {{
    "avg_kpt_conf": float,
    "ankles_visible": bool,
    "head_visible": bool,
    "too_close": bool,
    "too_far": bool,
    "off_center": "left|right|none",
    "camera_tilt_hint": "up|down|none"
  }},
  "phase": "standing|descending|bottom|ascending",
  "angles": {{
    "knee_deg": float,
    "hip_deg": float,
    "torso_lean_deg": float
  }},
  "events_recent": [string],
  "rep": {{"count": int, "in_set": bool}},
  "calibration": {{
    "depth_target_knee_deg": int,
    "torso_warn_deg": int,
    "stance_width_norm": float
  }},
  "user_prefs": {{"tone": "chill|coach", "brevity": "short"}}
}}

Output schema (each example.output):
{{
  "say": string,
  "priority": "framing|quality|safety|depth|knees|summary|encouragement|calibration",
  "ui": {{"highlight": "feet|head|center|hips|knees|torso|none", "show_checklist": bool}},
  "cooldown_s": float,
  "calibration_patch": null OR {{"depth_target_knee_deg": int?,"torso_warn_deg": int?,"stance_width_norm": float?}}
}}

Return ONLY:
{{"examples":[ ...{count} items... ]}}"""


VALID_MODES = {"setup", "live", "calibrate", "summary"}
VALID_PHASES = {"standing", "descending", "bottom", "ascending"}
VALID_VIEWS = {"side", "front"}
VALID_OFF_CENTER = {"left", "right", "none"}
VALID_TILT = {"up", "down", "none"}
VALID_TONES = {"chill", "coach"}
VALID_PRIORITIES = {
    "framing",
    "quality",
    "safety",
    "depth",
    "knees",
    "summary",
    "encouragement",
    "calibration",
}
VALID_HIGHLIGHTS = {"feet", "head", "center", "hips", "knees", "torso", "none"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate JSONL squat-coaching training data with Gemini as the teacher model."
    )
    parser.add_argument("--output", default="data/gemini_teacher_dataset.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--seed-prefix", default="gemini-teacher")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument(
        "--raw-output",
        default="",
        help="Optional path to persist the raw Gemini responses as a JSON array for auditing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated system instruction and prompt, then exit.",
    )
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit(
            "Missing dependency: install `google-genai` first with `python3 -m pip install -U google-genai`."
        )
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def build_prompt(count):
    return USER_PROMPT_TEMPLATE.format(count=count)


def call_gemini(client, model, prompt, temperature):
    response = client.models.generate_content(
        model=model,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
        contents=prompt,
    )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response body.")
    return text


def expect_type(value, expected_type, path):
    if not isinstance(value, expected_type):
        expected = getattr(expected_type, "__name__", str(expected_type))
        raise ValueError(f"{path} must be {expected}.")


def expect_number(value, path):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be numeric.")


def validate_bounds(value, low, high, path):
    if value < low or value > high:
        raise ValueError(f"{path} must be between {low} and {high}.")


def validate_example(example, index):
    expect_type(example, dict, f"examples[{index}]")
    if set(example.keys()) != {"input", "output"}:
        raise ValueError(f"examples[{index}] must only contain input and output.")

    input_payload = example["input"]
    output_payload = example["output"]

    expect_type(input_payload, dict, f"examples[{index}].input")
    expect_type(output_payload, dict, f"examples[{index}].output")

    if input_payload.get("mode") not in VALID_MODES:
        raise ValueError(f"examples[{index}].input.mode is invalid.")
    if input_payload.get("fps") not in {15, 30}:
        raise ValueError(f"examples[{index}].input.fps must be 15 or 30.")
    if input_payload.get("device") != "mac_webcam":
        raise ValueError(f"examples[{index}].input.device must be mac_webcam.")
    if input_payload.get("view_target") not in VALID_VIEWS:
        raise ValueError(f"examples[{index}].input.view_target is invalid.")
    if input_payload.get("phase") not in VALID_PHASES:
        raise ValueError(f"examples[{index}].input.phase is invalid.")

    quality = input_payload.get("quality")
    angles = input_payload.get("angles")
    rep = input_payload.get("rep")
    calibration = input_payload.get("calibration")
    prefs = input_payload.get("user_prefs")
    events_recent = input_payload.get("events_recent")

    expect_type(quality, dict, f"examples[{index}].input.quality")
    expect_number(quality.get("avg_kpt_conf"), f"examples[{index}].input.quality.avg_kpt_conf")
    validate_bounds(quality["avg_kpt_conf"], 0.0, 1.0, f"examples[{index}].input.quality.avg_kpt_conf")
    for key in ("ankles_visible", "head_visible", "too_close", "too_far"):
        expect_type(quality.get(key), bool, f"examples[{index}].input.quality.{key}")
    if quality.get("off_center") not in VALID_OFF_CENTER:
        raise ValueError(f"examples[{index}].input.quality.off_center is invalid.")
    if quality.get("camera_tilt_hint") not in VALID_TILT:
        raise ValueError(f"examples[{index}].input.quality.camera_tilt_hint is invalid.")

    expect_type(angles, dict, f"examples[{index}].input.angles")
    for key in ("knee_deg", "hip_deg", "torso_lean_deg"):
        expect_number(angles.get(key), f"examples[{index}].input.angles.{key}")

    expect_type(events_recent, list, f"examples[{index}].input.events_recent")
    if not all(isinstance(item, str) for item in events_recent):
        raise ValueError(f"examples[{index}].input.events_recent must contain strings only.")

    expect_type(rep, dict, f"examples[{index}].input.rep")
    if not isinstance(rep.get("count"), int) or isinstance(rep.get("count"), bool):
        raise ValueError(f"examples[{index}].input.rep.count must be an integer.")
    expect_type(rep.get("in_set"), bool, f"examples[{index}].input.rep.in_set")

    expect_type(calibration, dict, f"examples[{index}].input.calibration")
    for key in ("depth_target_knee_deg", "torso_warn_deg"):
        if not isinstance(calibration.get(key), int) or isinstance(calibration.get(key), bool):
            raise ValueError(f"examples[{index}].input.calibration.{key} must be an integer.")
    expect_number(
        calibration.get("stance_width_norm"),
        f"examples[{index}].input.calibration.stance_width_norm",
    )
    validate_bounds(
        calibration["depth_target_knee_deg"],
        80,
        115,
        f"examples[{index}].input.calibration.depth_target_knee_deg",
    )
    validate_bounds(
        calibration["torso_warn_deg"],
        25,
        60,
        f"examples[{index}].input.calibration.torso_warn_deg",
    )
    validate_bounds(
        calibration["stance_width_norm"],
        0.20,
        0.60,
        f"examples[{index}].input.calibration.stance_width_norm",
    )

    expect_type(prefs, dict, f"examples[{index}].input.user_prefs")
    if prefs.get("tone") not in VALID_TONES:
        raise ValueError(f"examples[{index}].input.user_prefs.tone is invalid.")
    if prefs.get("brevity") != "short":
        raise ValueError(f"examples[{index}].input.user_prefs.brevity must be short.")

    say = output_payload.get("say")
    if not isinstance(say, str) or not say.strip():
        raise ValueError(f"examples[{index}].output.say must be a non-empty string.")
    if len(say.split()) > 18:
        raise ValueError(f"examples[{index}].output.say exceeds 18 words.")
    if output_payload.get("priority") not in VALID_PRIORITIES:
        raise ValueError(f"examples[{index}].output.priority is invalid.")

    ui = output_payload.get("ui")
    expect_type(ui, dict, f"examples[{index}].output.ui")
    if ui.get("highlight") not in VALID_HIGHLIGHTS:
        raise ValueError(f"examples[{index}].output.ui.highlight is invalid.")
    expect_type(ui.get("show_checklist"), bool, f"examples[{index}].output.ui.show_checklist")
    expect_number(output_payload.get("cooldown_s"), f"examples[{index}].output.cooldown_s")

    patch = output_payload.get("calibration_patch")
    if patch is not None:
        expect_type(patch, dict, f"examples[{index}].output.calibration_patch")
        unknown_keys = set(patch.keys()) - {
            "depth_target_knee_deg",
            "torso_warn_deg",
            "stance_width_norm",
        }
        if unknown_keys:
            raise ValueError(
                f"examples[{index}].output.calibration_patch contains unsupported keys: {sorted(unknown_keys)}."
            )
        if "depth_target_knee_deg" in patch:
            if not isinstance(patch["depth_target_knee_deg"], int):
                raise ValueError(
                    f"examples[{index}].output.calibration_patch.depth_target_knee_deg must be an integer."
                )
            validate_bounds(
                patch["depth_target_knee_deg"],
                80,
                115,
                f"examples[{index}].output.calibration_patch.depth_target_knee_deg",
            )
        if "torso_warn_deg" in patch:
            if not isinstance(patch["torso_warn_deg"], int):
                raise ValueError(
                    f"examples[{index}].output.calibration_patch.torso_warn_deg must be an integer."
                )
            validate_bounds(
                patch["torso_warn_deg"],
                25,
                60,
                f"examples[{index}].output.calibration_patch.torso_warn_deg",
            )
        if "stance_width_norm" in patch:
            expect_number(
                patch["stance_width_norm"],
                f"examples[{index}].output.calibration_patch.stance_width_norm",
            )
            validate_bounds(
                patch["stance_width_norm"],
                0.20,
                0.60,
                f"examples[{index}].output.calibration_patch.stance_width_norm",
            )


def parse_examples(raw_text, expected_count):
    payload = json.loads(raw_text)
    expect_type(payload, dict, "response")
    if set(payload.keys()) != {"examples"}:
        raise ValueError("Gemini response must only contain the examples key.")
    examples = payload["examples"]
    expect_type(examples, list, "response.examples")
    if len(examples) != expected_count:
        raise ValueError(f"Expected {expected_count} examples, got {len(examples)}.")
    for index, example in enumerate(examples):
        validate_example(example, index)
    return examples


def write_jsonl(path, examples, seed_prefix, model_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index, example in enumerate(examples, start=1):
            row = {
                "id": f"{seed_prefix}-{index:05d}",
                "source_model": model_name,
                **example,
            }
            handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    prompt = build_prompt(min(args.batch_size, args.count))

    if args.dry_run:
        print("SYSTEM INSTRUCTION\n")
        print(SYSTEM_INSTRUCTION)
        print("\nUSER PROMPT\n")
        print(prompt)
        return

    if args.count <= 0:
        raise SystemExit("--count must be greater than 0.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")

    require_sdk()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    collected = []
    raw_responses = []
    batch_number = 0

    while len(collected) < args.count:
        batch_number += 1
        remaining = args.count - len(collected)
        current_batch_size = min(args.batch_size, remaining)
        batch_prompt = build_prompt(current_batch_size)

        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                raw_text = call_gemini(client, args.model, batch_prompt, args.temperature)
                batch_examples = parse_examples(raw_text, current_batch_size)
                collected.extend(batch_examples)
                raw_responses.append(
                    {
                        "batch": batch_number,
                        "examples": current_batch_size,
                        "raw_text": raw_text,
                    }
                )
                print(
                    f"Completed batch {batch_number}: {len(collected)}/{args.count} examples written to memory.",
                    file=sys.stderr,
                )
                break
            except Exception as exc:  # pragma: no cover - depends on API/runtime behavior
                last_error = exc
                print(
                    f"Batch {batch_number} attempt {attempt}/{args.max_retries} failed: {exc}",
                    file=sys.stderr,
                )
                if attempt < args.max_retries:
                    time.sleep(args.sleep_seconds)
        else:
            raise SystemExit(f"Failed to generate batch {batch_number}: {last_error}")

    output_path = Path(args.output)
    write_jsonl(output_path, collected, args.seed_prefix, args.model)
    if args.raw_output:
        raw_output_path = Path(args.raw_output)
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(json.dumps(raw_responses, indent=2))

    print(f"Wrote {len(collected)} examples to {output_path}")
    if args.raw_output:
        print(f"Wrote raw Gemini responses to {args.raw_output}")


if __name__ == "__main__":
    main()
