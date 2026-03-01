#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

from prompt_templates import rule_coach_output


LABEL_TO_PRIORITY = {
    "setup_issue": "framing",
    "too_close": "framing",
    "too_far": "framing",
    "bad_side_view": "framing",
    "forward_lean": "safety",
    "shallow_depth": "depth",
    "good_depth": "encouragement",
    "knee_valgus": "knees",
    "knees_in": "knees",
}


def load_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_input(row):
    src = row.get("input") or {}
    if "mode" in src:
        return src
    is_setup = bool(src.get("is_setup"))
    confidence = src.get("confidence_min")
    quality = {
        "avg_kpt_conf": confidence if isinstance(confidence, (int, float)) else 0.9,
        "ankles_visible": not is_setup or row.get("label") not in {"too_close", "too_far", "bad_side_view"},
        "head_visible": not is_setup,
        "too_close": row.get("label") == "too_close",
        "too_far": row.get("label") == "too_far",
        "off_center": "left" if row.get("label") == "bad_side_view" else "none",
    }
    return {
        "mode": "setup" if is_setup else "live",
        "phase": "bottom" if src.get("phase") in {"bottom", "descent", "descending"} else src.get("phase", "standing"),
        "quality": quality,
        "angles": {
            "knee_deg": src.get("knee_angle"),
            "hip_deg": src.get("hip_angle"),
            "torso_lean_deg": src.get("torso_angle"),
        },
        "events_recent": [row.get("label")] if row.get("label") in {"knee_valgus", "knees_in"} else [],
        "calibration": {
            "depth_target_knee_deg": 100,
            "torso_warn_deg": 40,
            "stance_width_norm": 0.35,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Audit label-policy mismatches in coaching datasets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = list(load_jsonl(args.input))
    mismatches = []
    counts = Counter()

    for row in rows:
        normalized = normalize_input(row)
        policy = rule_coach_output(normalized)
        label_priority = row.get("output", {}).get("priority")
        if label_priority is None:
            label_priority = LABEL_TO_PRIORITY.get(row.get("label"), "unknown")
        match = label_priority == policy["priority"]
        counts["rows"] += 1
        counts["matches"] += int(match)
        if not match:
            counts["mismatches"] += 1
            counts[f"mismatch:{label_priority}->{policy['priority']}"] += 1
            mismatches.append(
                {
                    "id": row.get("id"),
                    "label": row.get("label"),
                    "label_priority": label_priority,
                    "policy_priority": policy["priority"],
                    "expected_feedback": row.get("expected_feedback") or (row.get("output") or {}).get("say"),
                    "policy_say": policy["say"],
                    "input": row.get("input"),
                }
            )

    summary = {
        "input": args.input,
        "rows": counts["rows"],
        "matches": counts["matches"],
        "mismatches": counts["mismatches"],
        "match_rate": round(counts["matches"] / max(1, counts["rows"]), 4),
        "mismatch_breakdown": {
            key.split("mismatch:", 1)[1]: value
            for key, value in sorted(counts.items())
            if key.startswith("mismatch:")
        },
        "examples": mismatches[:50],
    }

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
