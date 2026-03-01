#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SESSION_SUFFIX_RE = re.compile(r"(-rep\d+|-setup)$")


def load_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nested_get(value, dotted_path):
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def auto_group_key(row):
    for path in ("session_id", "input.session_id", "input.session.session_id"):
        value = nested_get(row, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    row_id = row.get("id")
    if isinstance(row_id, str) and row_id.strip():
        row_id = row_id.strip()
        stripped = SESSION_SUFFIX_RE.sub("", row_id)
        if row_id.startswith("synthetic-"):
            return row_id
        if stripped != row_id.strip():
            return stripped
    source = row.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    source_model = row.get("source_model")
    if isinstance(source_model, str) and source_model.strip():
        return source_model.strip()
    if isinstance(row_id, str) and row_id.strip():
        return row_id.strip()
    digest = hashlib.sha1(json.dumps(row, sort_keys=True).encode()).hexdigest()[:12]
    return f"row-{digest}"


def group_key(row, group_by):
    if group_by == "auto":
        return auto_group_key(row)
    value = nested_get(row, group_by)
    if value is None:
        return auto_group_key(row)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def summarize_rows(rows):
    labels = Counter()
    modes = Counter()
    groups = set()
    for row in rows:
        if "label" in row:
            labels[row["label"]] += 1
        elif isinstance(row.get("output"), dict) and "priority" in row["output"]:
            labels[row["output"]["priority"]] += 1
        mode = nested_get(row, "input.mode")
        if mode:
            modes[mode] += 1
        groups.add(row["_group"])
    return {
        "rows": len(rows),
        "groups": len(groups),
        "labels": dict(sorted(labels.items())),
        "modes": dict(sorted(modes.items())),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            clean = dict(row)
            clean.pop("_group", None)
            handle.write(json.dumps(clean) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Split a JSONL dataset by group so related rows stay in the same split.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--group-by", default="auto", help="Dotted path like source or input.session_id, or auto.")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.input)
    if not rows:
        raise SystemExit("Input dataset is empty.")

    grouped = defaultdict(list)
    enriched = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row["_group"] = group_key(row, args.group_by)
        grouped[enriched_row["_group"]].append(enriched_row)
        enriched.append(enriched_row)

    group_ids = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(group_ids)

    target_eval_rows = max(1, round(len(enriched) * args.eval_ratio))
    eval_groups = []
    eval_rows = 0
    for gid in group_ids:
        if eval_rows >= target_eval_rows:
            break
        eval_groups.append(gid)
        eval_rows += len(grouped[gid])

    eval_group_set = set(eval_groups)
    train_rows = [row for row in enriched if row["_group"] not in eval_group_set]
    eval_rows = [row for row in enriched if row["_group"] in eval_group_set]

    if not train_rows or not eval_rows:
        raise SystemExit("Split failed: train or eval ended up empty. Adjust eval-ratio or dataset size.")

    train_output = Path(args.train_output)
    eval_output = Path(args.eval_output)
    write_jsonl(train_output, train_rows)
    write_jsonl(eval_output, eval_rows)

    summary = {
        "input": args.input,
        "group_by": args.group_by,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "train": summarize_rows(train_rows),
        "eval": summarize_rows(eval_rows),
        "eval_groups": sorted(eval_group_set),
    }

    summary_output = Path(args.summary_output) if args.summary_output else train_output.parent / f"{train_output.stem}.split_summary.json"
    summary_output.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
