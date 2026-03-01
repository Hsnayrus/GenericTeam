#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random


CANONICAL_BY_LABEL = {
    "shallow_depth": "Go deeper.",
    "forward_lean": "Chest up.",
    "good_depth": "Good depth.",
    "too_close": "Step back so I can see you.",
    "too_far": "Move closer so I can see you.",
    "bad_side_view": "Turn sideways to the camera.",
    "setup_issue": "Fix your camera setup.",
}


SETUP_LABELS = {"too_close", "too_far", "bad_side_view", "setup_issue"}


def load_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def normalize_row(row):
    row = json.loads(json.dumps(row))
    label = row.get("label")
    if label is None and (row.get("input") or {}).get("is_setup"):
        label = "setup_issue"
        row["label"] = label
    if label in SETUP_LABELS:
        row["input"]["phase"] = "standing"
        row["input"]["rep_number"] = 0
        row["input"]["is_setup"] = True
    else:
        row["input"]["is_setup"] = False
    row["expected_feedback"] = CANONICAL_BY_LABEL.get(label, row["expected_feedback"]).strip()
    row["reasoning"] = row.get("reasoning") or f"Canonicalized {label} coaching cue."
    apply_temporal_features(row, Random(f"{row.get('id')}:{label}"))
    return row


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def apply_temporal_features(row, rng):
    payload = row["input"]
    label = row.get("label")
    is_setup = bool(payload.get("is_setup"))

    if is_setup:
        payload["depth_history"] = []
        payload["phase_durations_ms"] = {"standing": 900}
        payload["rep_knee_min"] = None
        payload["rep_torso_max"] = None
        payload["knee_angle_delta"] = None
        payload["torso_angle_delta"] = None
        payload["window"] = []
        return row

    knee = float(payload.get("knee_angle") or 120)
    hip = float(payload.get("hip_angle") or 105)
    torso = float(payload.get("torso_angle") or 25)
    shin = float(payload.get("shin_angle") or 75)
    confidence = float(payload.get("confidence_min") or 0.8)

    if label == "shallow_depth":
        depth_history = [
            clamp(knee + 24 + rng.uniform(-4, 4), 95, 170),
            clamp(knee + 14 + rng.uniform(-3, 3), 92, 160),
            clamp(knee + 7 + rng.uniform(-2, 2), 90, 150),
            clamp(knee + rng.uniform(-1.5, 1.5), 88, 145),
        ]
        rep_torso_max = clamp(max(torso, torso + rng.uniform(1, 6)), 10, 55)
        torso_delta = rng.uniform(0.5, 4.0)
    elif label == "forward_lean":
        depth_history = [
            clamp(knee + 18 + rng.uniform(-4, 4), 90, 155),
            clamp(knee + 10 + rng.uniform(-3, 3), 88, 145),
            clamp(knee + 4 + rng.uniform(-2, 2), 85, 138),
            clamp(knee + rng.uniform(-1.5, 1.5), 84, 135),
        ]
        rep_torso_max = clamp(max(40.0, torso, torso + rng.uniform(5, 10)), 38, 60)
        torso_delta = rng.uniform(3.0, 7.0)
    else:
        depth_history = [
            clamp(knee + 20 + rng.uniform(-4, 4), 88, 155),
            clamp(knee + 10 + rng.uniform(-3, 3), 84, 140),
            clamp(knee + 4 + rng.uniform(-2, 2), 82, 130),
            clamp(knee + rng.uniform(-1.5, 1.5), 80, 125),
        ]
        rep_torso_max = clamp(max(torso, torso + rng.uniform(0, 3)), 8, 42)
        torso_delta = rng.uniform(-0.5, 2.0)

    payload["depth_history"] = [round(value, 1) for value in depth_history]
    payload["rep_knee_min"] = round(min(depth_history), 1)
    payload["rep_torso_max"] = round(rep_torso_max, 1)
    payload["knee_angle_delta"] = round(depth_history[-1] - depth_history[-2], 1)
    payload["torso_angle_delta"] = round(torso_delta, 1)
    payload["phase_durations_ms"] = {
        "descent": int(520 + rng.uniform(-90, 120)),
        "bottom": int(130 + rng.uniform(-40, 90)),
        "ascent": int(470 + rng.uniform(-80, 130)),
    }
    payload["window"] = [
        {
            "phase": "descent",
            "knee_angle": round(depth_history[-2], 1),
            "hip_angle": round(clamp(hip + 10 + rng.uniform(-4, 4), 70, 150), 1),
            "torso_angle": round(clamp(torso + max(torso_delta, 0) * 0.6, 8, 60), 1),
            "shin_angle": round(clamp(shin - 4 + rng.uniform(-3, 3), 10, 90), 1),
            "hip_below_knee": False,
            "confidence_min": round(clamp(confidence - 0.04, 0.3, 0.99), 2),
        },
        {
            "phase": payload.get("phase", "bottom"),
            "knee_angle": round(knee, 1),
            "hip_angle": round(hip, 1),
            "torso_angle": round(torso, 1),
            "shin_angle": round(shin, 1),
            "hip_below_knee": payload.get("hip_below_knee"),
            "confidence_min": round(confidence, 2),
        },
    ]
    return row


def make_augmented_row(row, index, rng):
    augmented = json.loads(json.dumps(row))
    augmented["id"] = f"{row['id']}-aug{index:02d}"
    augmented["source"] = f"{row.get('source', 'unknown')}_augmented"
    augmented["reasoning"] = f"{row.get('reasoning', '').strip()} Augmented from seed example.".strip()
    payload = augmented["input"]
    label = augmented.get("label")

    if label in SETUP_LABELS:
        apply_temporal_features(augmented, rng)
        return augmented

    if payload.get("knee_angle") is not None:
        knee_jitter = rng.uniform(-4.0, 4.0)
        if label == "shallow_depth":
            payload["knee_angle"] = round(clamp(payload["knee_angle"] + knee_jitter, 108.0, 135.0), 2)
        elif label == "good_depth":
            payload["knee_angle"] = round(clamp(payload["knee_angle"] + knee_jitter, 85.0, 103.0), 2)
        else:
            payload["knee_angle"] = round(clamp(payload["knee_angle"] + knee_jitter, 88.0, 130.0), 2)

    if payload.get("hip_angle") is not None:
        payload["hip_angle"] = round(clamp(payload["hip_angle"] + rng.uniform(-5.0, 5.0), 70.0, 130.0), 2)

    if payload.get("torso_angle") is not None:
        torso_jitter = rng.uniform(-4.0, 4.0)
        if label == "forward_lean":
            payload["torso_angle"] = round(clamp(payload["torso_angle"] + torso_jitter, 39.0, 55.0), 2)
        else:
            payload["torso_angle"] = round(clamp(payload["torso_angle"] + torso_jitter, 8.0, 38.0), 2)

    if payload.get("shin_angle") is not None:
        payload["shin_angle"] = round(clamp(payload["shin_angle"] + rng.uniform(-4.0, 4.0), 65.0, 90.0), 2)

    if payload.get("confidence_min") is not None:
        payload["confidence_min"] = round(clamp(payload["confidence_min"] + rng.uniform(-0.05, 0.05), 0.45, 0.98), 3)

    if label == "shallow_depth":
        payload["phase"] = "bottom"
        payload["hip_below_knee"] = False
    elif label == "good_depth":
        payload["phase"] = "bottom"
        payload["hip_below_knee"] = True
    elif label == "forward_lean":
        payload["phase"] = rng.choice(["descent", "bottom"])
        if payload.get("hip_below_knee") is None:
            payload["hip_below_knee"] = payload["phase"] == "bottom"

    apply_temporal_features(augmented, rng)
    return augmented


def augment_train_rows(rows, seed, target_train_min):
    rng = Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row.get("label", "unknown")].append(row)

    augmented = list(rows)
    for label, minimum in sorted(target_train_min.items()):
        label_rows = list(by_label.get(label, []))
        if not label_rows or len(label_rows) >= minimum:
            continue
        needed = minimum - len(label_rows)
        for index in range(needed):
            source = label_rows[index % len(label_rows)]
            augmented.append(make_augmented_row(source, index + 1, rng))
    return augmented


def scale_target_min(rows, target_total):
    by_label = Counter(row.get("label", "unknown") for row in rows)
    total = sum(by_label.values())
    if total <= 0 or target_total <= total:
        return dict(by_label)

    scaled = {}
    running = 0
    labels = sorted(by_label)
    for index, label in enumerate(labels):
        count = by_label[label]
        if index == len(labels) - 1:
            scaled_count = target_total - running
        else:
            scaled_count = max(count, round((count / total) * target_total))
            running += scaled_count
        scaled[label] = scaled_count
    return scaled


def choose_synthetic(real_rows, synthetic_rows, seed):
    rng = Random(seed)
    real_by_label = defaultdict(list)
    synth_by_label = defaultdict(list)
    for row in real_rows:
        real_by_label[row["label"]].append(row)
    for row in synthetic_rows:
        synth_by_label[row["label"]].append(row)

    selected = []
    target_min = {
        "shallow_depth": 40,
        "forward_lean": 30,
        "good_depth": 30,
        "setup_issue": 18,
        "too_close": 8,
        "too_far": 8,
        "bad_side_view": 8,
    }

    for label, rows in synth_by_label.items():
        rng.shuffle(rows)
        real_count = len(real_by_label[label])
        cap = max(target_min.get(label, 12) - real_count, 0)
        if real_count:
            cap = min(max(real_count * 6, cap), max(target_min.get(label, 12), real_count * 8))
        selected.extend(rows[:cap])
    return selected


def split_rows(rows, seed, eval_ratio):
    by_label = defaultdict(list)
    for row in rows:
        by_label[row.get("label", "unknown")].append(row)
    rng = Random(seed)
    train_rows = []
    eval_rows = []
    for label, label_rows in sorted(by_label.items()):
        label_rows = list(label_rows)
        rng.shuffle(label_rows)
        eval_count = 0 if len(label_rows) == 1 else max(1, round(len(label_rows) * eval_ratio))
        eval_rows.extend(label_rows[:eval_count])
        train_rows.extend(label_rows[eval_count:])
    return train_rows, eval_rows


def summarize(name, rows):
    labels = Counter(row["label"] for row in rows)
    sources = Counter(row.get("source") for row in rows)
    print(name, len(rows), dict(sorted(labels.items())), dict(sorted(sources.items())))


def main():
    parser = argparse.ArgumentParser(description="Build improved normalized real-heavy squat-coach dataset.")
    parser.add_argument("--real-input", default="data/gemini_coaching_run4.jsonl")
    parser.add_argument("--synthetic-input", default="data/gemini_synthetic_coaching.v2.jsonl")
    parser.add_argument("--output", default="data/gemini_combined_coaching.v3.jsonl")
    parser.add_argument("--train-output", default="data/gemini_combined_coaching.v3.train.jsonl")
    parser.add_argument("--eval-output", default="data/gemini_combined_coaching.v3.eval.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--target-train-size", type=int, default=0)
    args = parser.parse_args()

    real_rows = [normalize_row(row) for row in load_jsonl(args.real_input)]
    synthetic_rows = [normalize_row(row) for row in load_jsonl(args.synthetic_input)]
    selected_synthetic = choose_synthetic(real_rows, synthetic_rows, args.seed)

    combined = real_rows + selected_synthetic
    train_rows, eval_rows = split_rows(combined, args.seed, args.eval_ratio)
    if args.augment_train:
        target_train_min = {
            "shallow_depth": 48,
            "forward_lean": 40,
            "good_depth": 40,
            "setup_issue": 18,
            "too_close": 8,
            "too_far": 8,
            "bad_side_view": 8,
        }
        if args.target_train_size:
            target_train_min = scale_target_min(train_rows, args.target_train_size)
        train_rows = augment_train_rows(
            train_rows,
            args.seed,
            target_train_min,
        )

    write_jsonl(args.output, combined)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)

    summarize("real", real_rows)
    summarize("synthetic_selected", selected_synthetic)
    summarize("combined", combined)
    summarize("train", train_rows)
    summarize("eval", eval_rows)


if __name__ == "__main__":
    main()
