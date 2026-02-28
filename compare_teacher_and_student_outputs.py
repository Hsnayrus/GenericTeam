#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text())


def safe_get_review(payload):
    if not payload:
        return None
    if "review" in payload:
        return payload["review"]
    return payload


def main():
    parser = argparse.ArgumentParser(description="Assemble a side-by-side comparison of Gemini and Gemma outputs.")
    parser.add_argument("--gemini-pose", required=True)
    parser.add_argument("--gemini-visual", required=True)
    parser.add_argument("--gemma-compare", required=True)
    parser.add_argument("--eval-1b", default="")
    parser.add_argument("--eval-4b", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gemini_pose = load_json(args.gemini_pose)
    gemini_visual = load_json(args.gemini_visual)
    gemma_compare = load_json(args.gemma_compare)

    summary = {
        "rep_number": gemma_compare.get("rep_number"),
        "pose_input": gemma_compare.get("pose_input"),
        "frame_paths": gemma_compare.get("frame_paths"),
        "rules": gemma_compare.get("rules"),
        "outputs": {
            "gemini_pose_prose": safe_get_review(gemini_pose),
            "gemini_visual": safe_get_review(gemini_visual),
            "gemma_1b_pose": ((gemma_compare.get("models") or {}).get("gemma3:1b") or {}).get("pose_prompted"),
            "gemma_4b_pose": ((gemma_compare.get("models") or {}).get("gemma3:4b") or {}).get("pose_prompted"),
        },
        "eval_summary": {},
    }

    if args.eval_1b:
        eval_1b = load_json(args.eval_1b)
        summary["eval_summary"]["gemma3:1b"] = {
            "cases": eval_1b.get("cases"),
            "gold_priority_accuracy": eval_1b.get("gold_priority_accuracy"),
            "policy_priority_accuracy": eval_1b.get("policy_priority_accuracy"),
            "avg_gold_token_overlap": eval_1b.get("avg_gold_token_overlap"),
            "say_within_18_words_rate": eval_1b.get("say_within_18_words_rate"),
        }
    if args.eval_4b:
        eval_4b = load_json(args.eval_4b)
        summary["eval_summary"]["gemma3:4b"] = {
            "cases": eval_4b.get("cases"),
            "gold_priority_accuracy": eval_4b.get("gold_priority_accuracy"),
            "policy_priority_accuracy": eval_4b.get("policy_priority_accuracy"),
            "avg_gold_token_overlap": eval_4b.get("avg_gold_token_overlap"),
            "say_within_18_words_rate": eval_4b.get("say_within_18_words_rate"),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    print(f"Wrote comparison summary to {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
