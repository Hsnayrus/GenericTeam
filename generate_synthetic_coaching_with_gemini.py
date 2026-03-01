#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from random import Random

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


SYSTEM_INSTRUCTION = """You are generating synthetic squat-coaching fine-tuning data for a small local model.

Return valid JSON only with one top-level key: "examples".
Each example must match this schema exactly:
{
  "label": "good_depth" | "shallow_depth" | "forward_lean" | "setup_issue",
  "input": {
    "phase": "bottom" | "descent" | "standing",
    "rep_number": integer,
    "knee_angle": number | null,
    "hip_angle": number | null,
    "torso_angle": number | null,
    "shin_angle": number | null,
    "hip_below_knee": boolean | null,
    "confidence_min": number | null,
    "is_setup": boolean
  },
  "expected_feedback": string,
  "severity": "good" | "warn" | "bad",
  "speak": boolean,
  "reasoning": string
}

Rules:
- Keep expected_feedback short and natural. Prefer 2-8 words for live cues.
- Use only one coaching priority per example.
- Setup issues must have phase="standing", rep_number=0, is_setup=true, and knee_angle/hip_angle/torso_angle/shin_angle/hip_below_knee/confidence_min all null.
- Rep coaching rows must have is_setup=false and realistic angle values.
- Shallow depth should usually have knee_angle above 105 at bottom.
- Good depth should usually have knee_angle between 85 and 105 at bottom.
- Forward lean should usually have torso_angle above 38 during descent or bottom.
- Do not invent labels outside the allowed set.
- Keep values realistic for side-view squat coaching with MediaPipe-derived features.
- Produce a balanced mix across the allowed labels.
"""


USER_PROMPT = """Generate {count} synthetic squat-coaching examples.

Target distribution:
- 35% shallow_depth
- 25% good_depth
- 25% forward_lean
- 15% setup_issue

Requirements:
- Include varied wording, but keep feedback concise.
- Use realistic pose values.
- For setup_issue rows, vary the camera/setup reasoning, but keep the label as setup_issue.
- Mix a few examples with low confidence_min (0.3-0.55), but most should be usable (0.55-0.95).
- Use rep_number values 1-8 for rep coaching rows.

Return only:
{{"examples":[...]}}
"""


ALLOWED_LABELS = {"good_depth", "shallow_depth", "forward_lean", "setup_issue"}
ALLOWED_PHASES = {"bottom", "descent", "standing"}
ALLOWED_SEVERITIES = {"good", "warn", "bad"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic squat-coaching JSONL with Gemini.")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--output-jsonl", default="data/gemini_synthetic_coaching.jsonl")
    parser.add_argument("--train-jsonl", default="data/gemini_synthetic_coaching.train.jsonl")
    parser.add_argument("--eval-jsonl", default="data/gemini_synthetic_coaching.eval.jsonl")
    parser.add_argument("--merge-with", default="", help="Optional existing JSONL dataset to merge with.")
    parser.add_argument("--merged-output", default="data/gemini_combined_coaching.jsonl")
    parser.add_argument("--merged-train-output", default="data/gemini_combined_coaching.train.jsonl")
    parser.add_argument("--merged-eval-output", default="data/gemini_combined_coaching.eval.jsonl")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "models/gemini-2.5-pro"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing dependency: install google-genai first.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def expect_type(value, expected_type, path):
    if not isinstance(value, expected_type):
        expected = getattr(expected_type, "__name__", str(expected_type))
        raise ValueError(f"{path} must be {expected}.")


def expect_number_or_null(value, path):
    if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool)):
        raise ValueError(f"{path} must be numeric or null.")


def validate_example(example, index):
    expect_type(example, dict, f"examples[{index}]")
    expected_keys = {"label", "input", "expected_feedback", "severity", "speak", "reasoning"}
    if set(example.keys()) != expected_keys:
        raise ValueError(f"examples[{index}] must contain exactly {sorted(expected_keys)}.")

    if example["label"] not in ALLOWED_LABELS:
        raise ValueError(f"examples[{index}].label is invalid.")
    if example["severity"] not in ALLOWED_SEVERITIES:
        raise ValueError(f"examples[{index}].severity is invalid.")
    expect_type(example["speak"], bool, f"examples[{index}].speak")
    if not isinstance(example["expected_feedback"], str) or not example["expected_feedback"].strip():
        raise ValueError(f"examples[{index}].expected_feedback must be a non-empty string.")
    if not isinstance(example["reasoning"], str) or not example["reasoning"].strip():
        raise ValueError(f"examples[{index}].reasoning must be a non-empty string.")

    payload = example["input"]
    expect_type(payload, dict, f"examples[{index}].input")
    required_input = {
        "phase",
        "rep_number",
        "knee_angle",
        "hip_angle",
        "torso_angle",
        "shin_angle",
        "hip_below_knee",
        "confidence_min",
        "is_setup",
    }
    if set(payload.keys()) != required_input:
        raise ValueError(f"examples[{index}].input must contain exactly {sorted(required_input)}.")
    if payload["phase"] not in ALLOWED_PHASES:
        raise ValueError(f"examples[{index}].input.phase is invalid.")
    if not isinstance(payload["rep_number"], int) or isinstance(payload["rep_number"], bool):
        raise ValueError(f"examples[{index}].input.rep_number must be an integer.")
    expect_type(payload["is_setup"], bool, f"examples[{index}].input.is_setup")
    expect_number_or_null(payload["knee_angle"], f"examples[{index}].input.knee_angle")
    expect_number_or_null(payload["hip_angle"], f"examples[{index}].input.hip_angle")
    expect_number_or_null(payload["torso_angle"], f"examples[{index}].input.torso_angle")
    expect_number_or_null(payload["shin_angle"], f"examples[{index}].input.shin_angle")
    expect_number_or_null(payload["confidence_min"], f"examples[{index}].input.confidence_min")
    if payload["hip_below_knee"] is not None and not isinstance(payload["hip_below_knee"], bool):
        raise ValueError(f"examples[{index}].input.hip_below_knee must be boolean or null.")

    if payload["is_setup"]:
        if payload["phase"] != "standing" or payload["rep_number"] != 0:
            raise ValueError(f"examples[{index}] setup rows must use standing/rep_number=0.")
        for key in ("knee_angle", "hip_angle", "torso_angle", "shin_angle", "hip_below_knee", "confidence_min"):
            if payload[key] is not None:
                raise ValueError(f"examples[{index}] setup rows must leave {key} as null.")
    else:
        if payload["phase"] == "standing":
            raise ValueError(f"examples[{index}] rep coaching rows cannot use standing phase.")
        if payload["rep_number"] <= 0:
            raise ValueError(f"examples[{index}] rep coaching rows must use rep_number > 0.")
        for key in ("knee_angle", "hip_angle", "torso_angle", "confidence_min"):
            if payload[key] is None:
                raise ValueError(f"examples[{index}] rep coaching rows must include {key}.")


def normalize_example(example):
    payload = example.get("input") or {}
    if payload.get("is_setup") or example.get("label") == "setup_issue":
        payload["phase"] = "standing"
        payload["rep_number"] = 0
        payload["is_setup"] = True
        for key in ("knee_angle", "hip_angle", "torso_angle", "shin_angle", "hip_below_knee", "confidence_min"):
            payload[key] = None
    example["input"] = payload
    return example


def parse_examples(raw_text):
    payload = json.loads(raw_text)
    expect_type(payload, dict, "response")
    if set(payload.keys()) != {"examples"}:
        raise ValueError("Gemini response must only contain the examples key.")
    examples = payload["examples"]
    expect_type(examples, list, "response.examples")
    for index, example in enumerate(examples):
        examples[index] = normalize_example(example)
        validate_example(example, index)
    return examples


def call_gemini(client, model, prompt, temperature):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response body.")
    return text


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def load_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("label") is None and (row.get("input") or {}).get("is_setup"):
                    row["label"] = "setup_issue"
                rows.append(row)
    return rows


def split_examples(rows, seed, eval_ratio=0.25):
    grouped = defaultdict(list)
    for row in rows:
        session_key = row["id"].rsplit("-rep", 1)[0].rsplit("-setup", 1)[0]
        grouped[session_key].append(row)
    keys = sorted(grouped)
    Random(seed).shuffle(keys)
    eval_count = max(1, round(len(keys) * eval_ratio)) if len(keys) > 1 else 0
    eval_keys = set(keys[:eval_count])
    train_rows = []
    eval_rows = []
    for key in keys:
        destination = eval_rows if key in eval_keys else train_rows
        destination.extend(grouped[key])
    return train_rows, eval_rows


def attach_ids(examples, seed_prefix, start_index):
    rows = []
    for index, example in enumerate(examples, start=start_index):
        payload = {
            "id": f"{seed_prefix}-{index:05d}",
            "source": "gemini_synthetic_teacher",
            **example,
        }
        rows.append(payload)
    return rows


def main():
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be greater than 0.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")

    require_sdk()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    collected = []
    start_index = 1

    while len(collected) < args.count:
        current_batch = min(args.batch_size, args.count - len(collected))
        prompt = USER_PROMPT.format(count=current_batch)
        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                raw_text = call_gemini(client, args.model, prompt, args.temperature)
                batch_rows = attach_ids(parse_examples(raw_text), "synthetic", start_index)
                collected.extend(batch_rows)
                start_index += len(batch_rows)
                print(
                    f"Completed batch: {len(collected)}/{args.count} synthetic examples",
                    file=sys.stderr,
                )
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                print(f"Batch attempt {attempt}/{args.max_retries} failed: {exc}", file=sys.stderr)
                if attempt < args.max_retries:
                    time.sleep(args.sleep_seconds)
        else:
            raise SystemExit(f"Failed to generate synthetic coaching data: {last_error}")

    write_jsonl(args.output_jsonl, collected)
    synthetic_train, synthetic_eval = split_examples(collected, args.seed)
    write_jsonl(args.train_jsonl, synthetic_train)
    write_jsonl(args.eval_jsonl, synthetic_eval)

    labels = defaultdict(int)
    for row in collected:
        labels[row["label"]] += 1

    print(f"Wrote {len(collected)} synthetic examples to {args.output_jsonl}")
    print(f"Synthetic train split: {args.train_jsonl} ({len(synthetic_train)})")
    print(f"Synthetic eval split: {args.eval_jsonl} ({len(synthetic_eval)})")
    print(f"Label distribution: {dict(sorted(labels.items()))}")

    if args.merge_with:
        merged = load_jsonl(args.merge_with) + collected
        merged_train, merged_eval = split_examples(merged, args.seed)
        write_jsonl(args.merged_output, merged)
        write_jsonl(args.merged_train_output, merged_train)
        write_jsonl(args.merged_eval_output, merged_eval)
        print(f"Merged dataset: {args.merged_output} ({len(merged)})")
        print(f"Merged train split: {args.merged_train_output} ({len(merged_train)})")
        print(f"Merged eval split: {args.merged_eval_output} ({len(merged_eval)})")


if __name__ == "__main__":
    main()
