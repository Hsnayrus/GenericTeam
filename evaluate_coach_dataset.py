#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from urllib import request

from prompt_templates import (
    COACH_RESPONSE_SCHEMA,
    build_gemma_messages,
    expected_priority_from_state,
    rule_coach_output,
    sanitize_coach_output,
)


def load_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize(text):
    return "".join(char.lower() if char.isalnum() or char.isspace() else " " for char in text).split()


def token_overlap(a, b):
    left = set(normalize(a))
    right = set(normalize(b))
    return len(left & right) / max(1, len(left))


def has_framing_issue(row):
    quality = row["input"]["quality"]
    return (
        not quality["ankles_visible"]
        or not quality["head_visible"]
        or quality["too_close"]
        or quality["too_far"]
        or quality["off_center"] != "none"
    )


def has_quality_issue(row):
    return row["input"]["quality"]["avg_kpt_conf"] < 0.55


def has_safety_issue(row):
    return row["input"]["angles"]["torso_lean_deg"] >= row["input"]["calibration"]["torso_warn_deg"]


def has_depth_issue(row):
    return (
        row["input"]["phase"] == "bottom"
        and row["input"]["angles"]["knee_deg"] > row["input"]["calibration"]["depth_target_knee_deg"]
    )


def has_knees_issue(row):
    events = " ".join(row["input"]["events_recent"]).lower()
    return any(flag in events for flag in ("valgus", "knees_in", "knees in", "knee_valgus"))


def expected_priority(row):
    return expected_priority_from_state(row["input"])


def rules_backend(row):
    return rule_coach_output(row["input"])


def ollama_backend(row, model, base_url):
    body = json.dumps(
        {
            "model": model,
            "messages": build_gemma_messages(row["input"]),
            "stream": False,
            "format": COACH_RESPONSE_SCHEMA,
            "options": {"temperature": 0},
        }
    ).encode()
    req = request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=120) as response:
        parsed = json.loads(response.read().decode())
    content = parsed["message"]["content"]
    return sanitize_coach_output(row["input"], json.loads(content))


def evaluate_output(row, actual):
    gold = row["output"]
    inferred_priority = expected_priority(row)
    say = actual.get("say", "")
    actual_priority = actual.get("priority")
    return {
        "id": row["id"],
        "mode": row["input"]["mode"],
        "expected_priority_from_policy": inferred_priority,
        "gold_priority": gold["priority"],
        "actual_priority": actual_priority,
        "gold_priority_match": actual_priority == gold["priority"],
        "policy_priority_match": actual_priority == inferred_priority,
        "say_word_count": len(str(say).split()),
        "say_within_18_words": len(str(say).split()) <= 18,
        "gold_token_overlap": round(token_overlap(gold["say"], str(say)), 4),
        "framing_issue_present": has_framing_issue(row),
        "quality_issue_present": has_quality_issue(row),
        "safety_issue_present": has_safety_issue(row),
        "depth_issue_present": has_depth_issue(row),
        "knees_issue_present": has_knees_issue(row),
        "actual": actual,
    }


def summarize(results):
    total = len(results)
    return {
        "cases": total,
        "gold_priority_accuracy": round(sum(r["gold_priority_match"] for r in results) / max(1, total), 4),
        "policy_priority_accuracy": round(sum(r["policy_priority_match"] for r in results) / max(1, total), 4),
        "avg_gold_token_overlap": round(sum(r["gold_token_overlap"] for r in results) / max(1, total), 4),
        "say_within_18_words_rate": round(sum(r["say_within_18_words"] for r in results) / max(1, total), 4),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate squat-coach datasets against rules or a local model.")
    parser.add_argument("--dataset", default="data/gemini_teacher_dataset.sample.jsonl")
    parser.add_argument("--backend", choices=["gold", "rules", "ollama"], default="gold")
    parser.add_argument("--model", default="gemma3:1b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="data/coach_eval_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = list(load_jsonl(args.dataset))
    if args.limit:
        rows = rows[: args.limit]

    results = []
    for row in rows:
        if args.backend == "gold":
            actual = row["output"]
        elif args.backend == "rules":
            actual = rules_backend(row)
        else:
            actual = ollama_backend(row, args.model, args.base_url)
        results.append(evaluate_output(row, actual))

    summary = {
        "backend": args.backend,
        "model": args.model if args.backend == "ollama" else None,
        "dataset": args.dataset,
        **summarize(results),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote evaluation summary to {output}")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
