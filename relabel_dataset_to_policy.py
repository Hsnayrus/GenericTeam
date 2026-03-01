#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from prompt_templates import rule_coach_output
from report_policy_mismatches import LABEL_TO_PRIORITY, load_jsonl, normalize_input


PRIORITY_TO_LABEL = {
    "framing": "setup_issue",
    "quality": "setup_issue",
    "safety": "forward_lean",
    "depth": "shallow_depth",
    "knees": "knee_valgus",
    "encouragement": "good_depth",
    "summary": "good_depth",
    "calibration": "good_depth",
}


def relabel_row(row):
    normalized = normalize_input(row)
    policy = rule_coach_output(normalized)
    relabeled = dict(row)
    relabeled["policy_output"] = policy
    relabeled["policy_priority"] = policy["priority"]
    relabeled["policy_source"] = "deterministic_policy_v1"

    if "output" in relabeled and isinstance(relabeled["output"], dict):
        relabeled["output"] = {
            "say": policy["say"],
            "priority": policy["priority"],
            "ui": policy["ui"],
            "cooldown_s": policy["cooldown_s"],
            "calibration_patch": policy["calibration_patch"],
        }
    if "expected_feedback" in relabeled:
        relabeled["expected_feedback"] = policy["say"]
    if "severity" in relabeled:
        relabeled["severity"] = "bad" if policy["priority"] in {"framing", "quality"} else "warn" if policy["priority"] in {"safety", "depth", "knees"} else "good"
    if "speak" in relabeled:
        relabeled["speak"] = policy["priority"] not in {"encouragement", "summary"}
    if "label" in relabeled:
        relabeled["label"] = PRIORITY_TO_LABEL.get(policy["priority"], relabeled["label"])
    relabeled["original_label_priority"] = relabeled.get("output", {}).get("priority") if isinstance(relabeled.get("output"), dict) else LABEL_TO_PRIORITY.get(row.get("label"))
    return relabeled


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite dataset rows to the deterministic coaching policy.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [relabel_row(row) for row in load_jsonl(args.input)]
    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} policy-aligned rows to {args.output}")


if __name__ == "__main__":
    main()
