#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from urllib import request

from evaluate_coach_dataset import evaluate_output, load_jsonl, summarize
from prompt_templates import COACH_RESPONSE_SCHEMA, build_gemma_messages, sanitize_coach_output


def ollama_raw_and_sanitized(row, model, base_url, include_examples):
    body = json.dumps(
        {
            "model": model,
            "messages": build_gemma_messages(row["input"], include_examples=include_examples),
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
    with request.urlopen(req, timeout=180) as response:
        parsed = json.loads(response.read().decode())
    raw = json.loads(parsed["message"]["content"])
    sanitized = sanitize_coach_output(row["input"], raw)
    return raw, sanitized


def intervention_flags(raw, sanitized):
    changed = raw != sanitized
    return {
        "any": changed,
        "priority_changed": raw.get("priority") != sanitized.get("priority"),
        "say_changed": raw.get("say") != sanitized.get("say"),
        "highlight_changed": ((raw.get("ui") or {}).get("highlight")) != ((sanitized.get("ui") or {}).get("highlight")),
        "show_checklist_changed": ((raw.get("ui") or {}).get("show_checklist")) != ((sanitized.get("ui") or {}).get("show_checklist")),
        "calibration_patch_changed": raw.get("calibration_patch") != sanitized.get("calibration_patch"),
    }


def summarize_interventions(interventions):
    total = len(interventions)
    return {
        "intervention_rate": round(sum(item["any"] for item in interventions) / max(1, total), 4),
        "priority_change_rate": round(sum(item["priority_changed"] for item in interventions) / max(1, total), 4),
        "say_change_rate": round(sum(item["say_changed"] for item in interventions) / max(1, total), 4),
        "highlight_change_rate": round(sum(item["highlight_changed"] for item in interventions) / max(1, total), 4),
        "show_checklist_change_rate": round(sum(item["show_checklist_changed"] for item in interventions) / max(1, total), 4),
        "calibration_patch_change_rate": round(sum(item["calibration_patch_changed"] for item in interventions) / max(1, total), 4),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate raw vs sanitized Gemma outputs and guardrail intervention rate.")
    parser.add_argument("--dataset", default="data/gemini_teacher_dataset.sample.jsonl")
    parser.add_argument("--model", default="gemma3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-examples", action="store_true")
    parser.add_argument("--output", default="data/results/guardrailed_ollama_eval.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = list(load_jsonl(args.dataset))
    if args.limit:
        rows = rows[: args.limit]

    raw_results = []
    sanitized_results = []
    per_case = []
    interventions = []

    for row in rows:
        raw, sanitized = ollama_raw_and_sanitized(
            row,
            model=args.model,
            base_url=args.base_url,
            include_examples=not args.no_examples,
        )
        raw_eval = evaluate_output(row, raw)
        sanitized_eval = evaluate_output(row, sanitized)
        flags = intervention_flags(raw, sanitized)
        raw_results.append(raw_eval)
        sanitized_results.append(sanitized_eval)
        interventions.append(flags)
        per_case.append(
            {
                "id": row["id"],
                "gold_priority": row["output"]["priority"],
                "raw": raw,
                "sanitized": sanitized,
                "intervention": flags,
                "raw_eval": raw_eval,
                "sanitized_eval": sanitized_eval,
            }
        )

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "cases": len(rows),
        "include_examples": not args.no_examples,
        "raw_summary": summarize(raw_results),
        "sanitized_summary": summarize(sanitized_results),
        "intervention_summary": summarize_interventions(interventions),
        "results": per_case,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote guardrailed evaluation to {output}")
    compact = {
        "dataset": summary["dataset"],
        "model": summary["model"],
        "cases": summary["cases"],
        "include_examples": summary["include_examples"],
        "raw_summary": summary["raw_summary"],
        "sanitized_summary": summary["sanitized_summary"],
        "intervention_summary": summary["intervention_summary"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
