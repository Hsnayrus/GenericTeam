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
    "You are a squat coach. Given structured pose-derived squat features, "
    "return one short coaching cue only. Keep it concise and actionable."
)


def build_user_prompt(row):
    payload = {
        "task": "squat_coaching",
        "input": row["input"],
        "rules": [
            "Return one short coaching cue.",
            "Do not explain the reasoning.",
            "Do not mention raw variable names.",
            "If is_setup is true, give a setup/framing instruction.",
        ],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def normalize(text):
    return "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split()


def overlap(expected, actual):
    left = set(normalize(expected))
    right = set(normalize(actual))
    return len(left & right) / max(1, len(left))


def require_sdk():
    if genai is None:
        raise SystemExit("Missing dependency: install google-genai.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def call_gemini(client, model, row):
    response = client.models.generate_content(
        model=model,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0,
        },
        contents=build_user_prompt(row),
    )
    text = getattr(response, "text", "") or ""
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    require_sdk()
    rows = [json.loads(line) for line in open(args.dataset) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results = []
    for idx, row in enumerate(rows, start=1):
        text = call_gemini(client, args.model, row)
        results.append(
            {
                "id": row["id"],
                "label": row.get("label"),
                "expected_feedback": row["expected_feedback"],
                "predicted_feedback": text,
                "token_overlap": round(overlap(row["expected_feedback"], text), 4),
                "exact_match": row["expected_feedback"].strip().lower() == text.strip().lower(),
            }
        )
        print(f"gemini: {idx}/{len(rows)}", flush=True)

    avg = sum(r["token_overlap"] for r in results) / max(1, len(results))
    exact = sum(r["exact_match"] for r in results) / max(1, len(results))
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "cases": len(results),
        "avg_token_overlap": round(avg, 4),
        "exact_match_rate": round(exact, 4),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
