#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from loguru import logger
from urllib import request


def load_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def rules_backend(payload):
    state = payload["input"]
    setup = state["setup_state"]
    if not setup["head_visible"]:
        return "Step back so your full head is visible."
    if not setup["feet_visible"]:
        return "Step back or tilt the camera down so both feet are visible."
    if not setup["distance_ok"]:
        return "Adjust your distance so your full body fits in frame."
    if not setup["centered"]:
        return "Center yourself in the frame."
    if not setup["side_view_ok"]:
        return "Rotate into a clearer side view."
    if state["knee_valgus_ratio"] < 0.82:
        return "Push your knees out as you descend."
    if state["torso_angle"] > 45:
        return "Chest up. Too much forward lean."
    if state["phase"] == "bottom" and state["knee_angle"] > 100:
        return "Go deeper and break parallel."
    return "Good depth. Keep going."


def openai_backend(payload, model, base_url):
    api_key = os.environ.get("OPENAI_API_KEY", "local")
    prompt = (
        "You are a concise squat coach. Reply with one sentence of coaching only.\n"
        f"Structured input:\n{json.dumps(payload['input'])}"
    )
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()
    req = request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with request.urlopen(req, timeout=60) as response:
        parsed = json.loads(response.read().decode())
    return parsed["choices"][0]["message"]["content"].strip()


def normalize(text):
    return "".join(char.lower() for char in text if char.isalnum() or char.isspace()).strip()


def score(expected, actual):
    expected_tokens = set(normalize(expected).split())
    actual_tokens = set(normalize(actual).split())
    return len(expected_tokens & actual_tokens) / max(1, len(expected_tokens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic_benchmark.jsonl")
    parser.add_argument("--backend", choices=["rules", "openai"], default="rules")
    parser.add_argument("--model", default="gemma")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="data/benchmark_results.json")
    args = parser.parse_args()

    rows = list(load_jsonl(args.dataset))
    if args.limit:
      rows = rows[:args.limit]

    results = []
    for row in rows:
        if args.backend == "rules":
            actual = rules_backend(row)
        else:
            actual = openai_backend(row, args.model, args.base_url)
        results.append({
            "id": row["id"],
            "label": row["label"],
            "expected_feedback": row["expected_feedback"],
            "actual_feedback": actual,
            "token_overlap_score": round(score(row["expected_feedback"], actual), 4),
        })

    avg_score = sum(item["token_overlap_score"] for item in results) / max(1, len(results))
    summary = {
        "backend": args.backend,
        "model": args.model,
        "dataset": args.dataset,
        "cases": len(results),
        "avg_token_overlap_score": round(avg_score, 4),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote benchmark summary to {output}")
    logger.info(f"Average token overlap score: {summary['avg_token_overlap_score']}")


if __name__ == "__main__":
    main()
