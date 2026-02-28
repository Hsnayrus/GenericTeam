#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def is_sane_live_frame(live):
    if not isinstance(live, dict):
        return False
    ranges = {
        "knee_angle": (45, 180),
        "hip_angle": (45, 180),
        "torso_angle": (0, 80),
        "shin_angle": (0, 70),
        "tempo_ms": (0, 10000),
    }
    for key, (low, high) in ranges.items():
        value = live.get(key)
        if not isinstance(value, (int, float)) or value < low or value > high:
            return False
    return isinstance(live.get("hip_below_knee"), bool)


def load_json(path):
    return json.loads(Path(path).read_text())


def clean_session(payload, min_confidence):
    frames = payload.get("frames", [])
    cleaned_frames = []
    for frame in frames:
        setup = frame.get("setup") or {}
        live = frame.get("live")
        confidence_min = frame.get("confidence_min", 0)
        gate_pass = bool(frame.get("gate_pass"))
        cleaned_live = live if is_sane_live_frame(live) and confidence_min >= min_confidence else None
        cleaned_frames.append(
            {
                **frame,
                "gate_pass": gate_pass and cleaned_live is not None,
                "setup": {
                    "head_visible": bool(setup.get("head_visible")),
                    "feet_visible": bool(setup.get("feet_visible")),
                    "centered": bool(setup.get("centered")),
                    "side_view": bool(setup.get("side_view")),
                },
                "live": cleaned_live,
            }
        )

    setup_dataset = [
        {
            "timestamp_ms": frame["timestamp_ms"],
            "head_visible": frame["setup"]["head_visible"],
            "feet_visible": frame["setup"]["feet_visible"],
            "centered": frame["setup"]["centered"],
            "side_view": frame["setup"]["side_view"],
            "gate_pass": frame["gate_pass"],
            "confidence_avg": frame.get("confidence_avg", 0),
        }
        for frame in cleaned_frames
        if not frame["gate_pass"]
    ]
    live_dataset = [
        {
            "timestamp_ms": frame["timestamp_ms"],
            **frame["live"],
        }
        for frame in cleaned_frames
        if frame["live"] is not None
    ]

    cleaned = {
        **payload,
        "schema_version": f"{payload.get('schema_version', 'unknown')}-clean",
        "cleaning": {
            "min_confidence": min_confidence,
            "sanity_filter": "angles_and_boolean_ranges",
        },
        "frames": cleaned_frames,
        "datasets": {
            "setup_coach": setup_dataset,
            "live_cue_side": live_dataset,
        },
    }
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Clean an offline squat session export into a training-ready schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    args = parser.parse_args()

    cleaned = clean_session(load_json(args.input), args.min_confidence)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cleaned, indent=2))

    frames = cleaned["frames"]
    live = cleaned["datasets"]["live_cue_side"]
    setup = cleaned["datasets"]["setup_coach"]
    print(f"Wrote cleaned session to {output}")
    print(
        json.dumps(
            {
                "frames": len(frames),
                "live_frames": len(live),
                "setup_frames": len(setup),
                "gate_pass_frames": sum(1 for frame in frames if frame["gate_pass"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
