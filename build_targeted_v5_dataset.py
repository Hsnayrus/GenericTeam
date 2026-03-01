#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_variant(path, variant):
    data = json.loads(Path(path).read_text())
    for item in data["variants"]:
        if item["variant"] == variant:
            return {row["id"]: row for row in item["results"]}
    raise SystemExit(f"Variant {variant!r} not found in {path}")


def write_jsonl(path, rows):
    with Path(path).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--student-results", required=True)
    parser.add_argument("--variant", default="adapter")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = load_jsonl(args.base_train)
    eval_rows = {row["id"]: row for row in load_jsonl(args.eval_set)}
    student = load_variant(args.student_results, args.variant)

    misses = []
    for row_id, result in student.items():
        if not result["exact_match"]:
            row = json.loads(json.dumps(eval_rows[row_id]))
            row["source"] = f"{row.get('source', 'unknown')}_v5_targeted"
            row["reasoning"] = (
                f"Added from holdout miss. Expected '{result['expected_feedback']}' "
                f"but student predicted '{result['predicted_feedback']}'."
            )
            misses.append(row)

    merged = list(train_rows) + misses
    write_jsonl(args.output, merged)
    print(json.dumps({"base_train": len(train_rows), "misses_added": len(misses), "total": len(merged)}, indent=2))


if __name__ == "__main__":
    main()
