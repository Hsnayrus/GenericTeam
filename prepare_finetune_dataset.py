#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a squat coach. "
    "Given structured pose-derived squat features, return exactly one approved coaching cue. "
    "Do not explain. Do not hedge. Do not add any extra words."
)


STRICT_CUE_BY_LABEL = {
    "shallow_depth": "Go deeper.",
    "forward_lean": "Chest up.",
    "good_depth": "Good depth.",
    "too_close": "Step back so I can see you.",
    "too_far": "Move closer so I can see you.",
    "bad_side_view": "Turn sideways to the camera.",
    "setup_issue": "Fix your camera setup.",
}


def load_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_user_prompt(row):
    payload = {
        "task": "squat_coaching",
        "input": row["input"],
        "rules": [
            "Return exactly one cue from the approved set.",
            "Approved cues: Go deeper. | Chest up. | Good depth. | Step back so I can see you. | Move closer so I can see you. | Turn sideways to the camera. | Fix your camera setup.",
            "Do not explain the reasoning.",
            "Do not mention raw variable names.",
            "Do not add punctuation beyond the final period.",
            "If is_setup is true, return only a setup/framing instruction.",
        ],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def answer_text(row):
    label = row.get("label")
    if label in STRICT_CUE_BY_LABEL:
        return STRICT_CUE_BY_LABEL[label]
    if isinstance(row.get("policy_output"), dict) and row["policy_output"].get("say"):
        return row["policy_output"]["say"].strip()
    return row["expected_feedback"].strip()


def build_text_example(row):
    user_prompt = build_user_prompt(row)
    answer = answer_text(row)
    return {
        "id": row["id"],
        "label": row.get("label"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer},
        ],
        "text": (
            f"<bos><start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
            f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{answer}<end_of_turn>"
        ),
    }


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare squat-coach JSONL for Gemma LoRA fine-tuning.")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--eval-input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    args = parser.parse_args()

    train_rows = [build_text_example(row) for row in load_jsonl(args.train_input)]
    eval_rows = [build_text_example(row) for row in load_jsonl(args.eval_input)]

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)

    print(f"Wrote {len(train_rows)} train rows to {args.train_output}")
    print(f"Wrote {len(eval_rows)} eval rows to {args.eval_output}")


if __name__ == "__main__":
    main()
