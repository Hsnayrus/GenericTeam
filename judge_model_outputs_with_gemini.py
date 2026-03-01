#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None


SYSTEM_PROMPT = (
    "You are grading squat coaching cues. "
    "Choose which candidate is better for the given structured squat state and gold target cue. "
    "Prefer the candidate that is more actionable, more aligned with the state, and closer to the gold cue. "
    "Return strict JSON only."
)


def require_sdk():
    if genai is None:
        raise SystemExit("Missing dependency: install google-genai.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def load_variant(path, variant):
    data = json.loads(Path(path).read_text())
    for item in data["variants"]:
        if item["variant"] == variant:
            return {row["id"]: row for row in item["results"]}
    raise SystemExit(f"Variant {variant!r} not found in {path}")


def load_rows(path):
    return {row["id"]: row for row in (json.loads(line) for line in open(path) if line.strip())}


def judge_one(client, model, row, left_name, left_text, right_name, right_text):
    payload = {
        "input": row["input"],
        "gold_cue": row["expected_feedback"],
        "candidate_a": {"name": left_name, "cue": left_text},
        "candidate_b": {"name": right_name, "cue": right_text},
        "rules": [
            "Pick the better cue for this state.",
            "If both are poor, still choose the less wrong one.",
            "Favor setup instructions only when setup/framing is actually the issue.",
            "Favor concise canonical cues when they fit the state.",
        ],
    }
    response = client.models.generate_content(
        model=model,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0,
            "response_mime_type": "application/json",
        },
        contents=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )
    text = getattr(response, "text", "") or ""
    parsed = json.loads(text)
    winner = parsed.get("winner")
    if winner not in {"a", "b", "tie"}:
        raise ValueError(f"Unexpected judge output: {parsed}")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--left-results", required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-results", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--variant", default="adapter")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    require_sdk()
    rows = load_rows(args.dataset)
    left = load_variant(args.left_results, args.variant)
    right = load_variant(args.right_results, args.variant)
    ids = sorted(set(rows) & set(left) & set(right))

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results = []
    counts = {"a": 0, "b": 0, "tie": 0}
    for idx, row_id in enumerate(ids, start=1):
        row = rows[row_id]
        left_text = left[row_id]["predicted_feedback"]
        right_text = right[row_id]["predicted_feedback"]
        judgment = judge_one(client, args.model, row, args.left_name, left_text, args.right_name, right_text)
        winner = judgment["winner"]
        counts[winner] += 1
        results.append(
            {
                "id": row_id,
                "gold_cue": row["expected_feedback"],
                "left": left_text,
                "right": right_text,
                "winner": winner,
                "reason": judgment.get("reason", ""),
            }
        )
        print(f"judge: {idx}/{len(ids)}", flush=True)

    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "left_name": args.left_name,
        "right_name": args.right_name,
        "cases": len(results),
        "wins": counts,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
