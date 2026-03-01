#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from loguru import logger


SEED_CASES = [
    {
        "label": "good_rep",
        "phase": "bottom",
        "knee_angle": (82, 98),
        "hip_angle": (65, 82),
        "torso_angle": (18, 34),
        "knee_valgus_ratio": (0.9, 1.05),
        "setup_state": {"head_visible": True, "feet_visible": True, "distance_ok": True, "centered": True, "side_view_ok": True},
        "expected_feedback": "Good depth. Keep going.",
    },
    {
        "label": "shallow_depth",
        "phase": "bottom",
        "knee_angle": (102, 120),
        "hip_angle": (72, 90),
        "torso_angle": (20, 38),
        "knee_valgus_ratio": (0.9, 1.02),
        "setup_state": {"head_visible": True, "feet_visible": True, "distance_ok": True, "centered": True, "side_view_ok": True},
        "expected_feedback": "Go deeper and break parallel.",
    },
    {
        "label": "knee_valgus",
        "phase": "descending",
        "knee_angle": (95, 120),
        "hip_angle": (70, 86),
        "torso_angle": (22, 40),
        "knee_valgus_ratio": (0.65, 0.81),
        "setup_state": {"head_visible": True, "feet_visible": True, "distance_ok": True, "centered": True, "side_view_ok": True},
        "expected_feedback": "Push your knees out as you descend.",
    },
    {
        "label": "forward_lean",
        "phase": "ascending",
        "knee_angle": (90, 118),
        "hip_angle": (68, 84),
        "torso_angle": (46, 62),
        "knee_valgus_ratio": (0.88, 1.02),
        "setup_state": {"head_visible": True, "feet_visible": True, "distance_ok": True, "centered": True, "side_view_ok": True},
        "expected_feedback": "Chest up. Too much forward lean.",
    },
    {
        "label": "cropped_feet",
        "phase": "standing",
        "knee_angle": (160, 174),
        "hip_angle": (165, 178),
        "torso_angle": (6, 18),
        "knee_valgus_ratio": (0.9, 1.0),
        "setup_state": {"head_visible": True, "feet_visible": False, "distance_ok": False, "centered": True, "side_view_ok": True},
        "expected_feedback": "Step back or tilt the camera down so both feet are visible.",
    },
    {
        "label": "bad_side_view",
        "phase": "standing",
        "knee_angle": (158, 172),
        "hip_angle": (166, 179),
        "torso_angle": (6, 18),
        "knee_valgus_ratio": (0.9, 1.0),
        "setup_state": {"head_visible": True, "feet_visible": True, "distance_ok": True, "centered": True, "side_view_ok": False},
        "expected_feedback": "Rotate into a clearer side view.",
    },
]


def sample_range(bounds):
    low, high = bounds
    return round(random.uniform(low, high), 2)


def make_example(case, index):
    return {
        "id": f"{case['label']}-{index:04d}",
        "label": case["label"],
        "input": {
            "phase": case["phase"],
            "rep_number": random.randint(1, 8),
            "knee_angle": sample_range(case["knee_angle"]),
            "hip_angle": sample_range(case["hip_angle"]),
            "torso_angle": sample_range(case["torso_angle"]),
            "knee_valgus_ratio": sample_range(case["knee_valgus_ratio"]),
            "confidence_avg": round(random.uniform(0.72, 0.96), 3),
            "confidence_min": round(random.uniform(0.52, 0.89), 3),
            "setup_state": case["setup_state"],
        },
        "expected_feedback": case["expected_feedback"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-per-case", type=int, default=40)
    parser.add_argument("--output", default="data/synthetic_benchmark.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case in SEED_CASES:
        for index in range(args.count_per_case):
            rows.append(make_example(case, index))

    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    logger.info(f"Wrote {len(rows)} synthetic examples to {output}")


if __name__ == "__main__":
    main()
