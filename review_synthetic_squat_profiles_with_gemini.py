#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


SYSTEM_INSTRUCTION = """You are an expert squat-form reviewer for a local squat coach.
You are given MediaPipe-style prose and structured squat metadata only.
You must not ask for the original video and you must not mention missing pixels or missing footage.

Return strict JSON only with this schema:
{
  "say": string,
  "priority": "framing|safety|depth|knees|encouragement",
  "reason": string,
  "confidence": number
}

Rules:
- Prioritize one coaching issue only.
- Keep "say" to one short sentence, maximum 18 words.
- Use the prose and structured metrics only.
- If the rep is acceptable, return encouragement.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Review synthetic squat examples with Gemini using prose only.")
    parser.add_argument("--manifest", required=True, help="Manifest from generate_synthetic_squat_videos.py")
    parser.add_argument("--example-id", default="", help="Optional single manifest id to review.")
    parser.add_argument("--output", default="data/results/synthetic_squat_reviews.json")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--dry-run", action="store_true", help="Print the prose-only review payload and exit.")
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing dependency: install `google-genai` first with `python3 -m pip install -r requirements.txt`.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def load_manifest(path):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise SystemExit("Manifest must be a JSON array.")
    return payload


def select_examples(rows, example_id):
    if not example_id:
        return rows
    selected = [row for row in rows if row.get("id") == example_id]
    if not selected:
        raise SystemExit(f"Example id {example_id} not found in manifest.")
    return selected


def build_prompt(example):
    compact = {
        "id": example.get("id"),
        "label": example.get("label"),
        "issue": example.get("issue"),
        "profile": example.get("profile"),
        "mediaprose": example.get("mediaprose"),
        "frames": [
            {
                "phase": frame.get("phase"),
                "spec": frame.get("spec"),
            }
            for frame in (example.get("frames") or [])
        ] or None,
    }
    return (
        "Review this synthetic side-view squat example using only the provided pose prose and structured metadata.\n\n"
        f"{json.dumps(compact, indent=2)}"
    )


def main():
    args = parse_args()
    rows = load_manifest(args.manifest)
    examples = select_examples(rows, args.example_id)

    if args.dry_run:
        preview = [{"id": example.get("id"), "prompt": build_prompt(example)} for example in examples]
        print(json.dumps(preview, indent=2))
        return

    require_sdk()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    reviews = []

    for example in examples:
        response = client.models.generate_content(
            model=args.model,
            contents=build_prompt(example),
            config={
                "system_instruction": SYSTEM_INSTRUCTION,
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )
        text = getattr(response, "text", None)
        if not text:
            raise SystemExit(f"Gemini returned an empty response for {example.get('id')}.")
        reviews.append(
            {
                "id": example.get("id"),
                "label": example.get("label"),
                "model": args.model,
                "video_path": example.get("video_path"),
                "frames": example.get("frames"),
                "mediaprose": example.get("mediaprose"),
                "review": json.loads(text),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reviews, indent=2))
    print(f"Wrote Gemini prose reviews to {output}")
    print(json.dumps(reviews, indent=2))


if __name__ == "__main__":
    main()
