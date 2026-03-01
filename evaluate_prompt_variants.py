#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from urllib import request

from evaluate_coach_dataset import evaluate_output, load_jsonl, summarize
from prompt_templates import COACH_RESPONSE_SCHEMA, build_gemma_messages, sanitize_coach_output


PROMPT_VARIANTS = {
    "default": {"include_examples": True},
    "no_examples": {"include_examples": False},
}


def ollama_backend_variant(row, model, base_url, include_examples):
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
    with request.urlopen(req, timeout=120) as response:
        parsed = json.loads(response.read().decode())
    return sanitize_coach_output(row["input"], json.loads(parsed["message"]["content"]))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Gemma prompt variants on the existing coach dataset.")
    parser.add_argument("--dataset", default="data/gemini_teacher_dataset.sample.jsonl")
    parser.add_argument("--model", default="gemma3:1b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--variants", nargs="+", default=["default", "no_examples"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="data/results/prompt_variant_eval.json")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = list(load_jsonl(args.dataset))
    if args.limit:
        rows = rows[: args.limit]

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "cases": len(rows),
        "variants": {},
    }

    for variant in args.variants:
        if variant not in PROMPT_VARIANTS:
            raise SystemExit(f"Unknown variant: {variant}. Choose from {sorted(PROMPT_VARIANTS)}")
        config = PROMPT_VARIANTS[variant]
        results = []
        for row in rows:
            actual = ollama_backend_variant(
                row,
                model=args.model,
                base_url=args.base_url,
                include_examples=config["include_examples"],
            )
            results.append(evaluate_output(row, actual))
        summary["variants"][variant] = {
            **summarize(results),
            "results": results,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote prompt comparison to {output}")
    compact = {
        "dataset": summary["dataset"],
        "model": summary["model"],
        "cases": summary["cases"],
        "variants": {
            name: {k: v for k, v in data.items() if k != "results"}
            for name, data in summary["variants"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
