#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean, median


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_json(path):
    return json.loads(Path(path).read_text())


def summarize(payload, depth_target_knee_deg, torso_warn_deg):
    frames = payload.get("frames", [])
    reps = payload.get("rep_summaries", [])
    gate_pass_frames = sum(1 for frame in frames if frame.get("gate_pass"))
    live_frames = sum(1 for frame in frames if frame.get("live") is not None)

    knee_angles = [rep["bottom_frame"]["knee_angle"] for rep in reps if rep.get("bottom_frame")]
    torso_angles = [rep["bottom_frame"]["torso_angle"] for rep in reps if rep.get("bottom_frame")]
    depth_flags = [angle for angle in knee_angles if angle > depth_target_knee_deg]
    torso_flags = [angle for angle in torso_angles if angle >= torso_warn_deg]

    baseline_readiness = "usable"
    reasons = []
    if len(reps) < 10:
        baseline_readiness = "weak"
        reasons.append("fewer than 10 reps")
    if frames and gate_pass_frames / len(frames) < 0.8:
        baseline_readiness = "weak"
        reasons.append("gate pass rate below 80%")
    if len(reps) < 20:
        reasons.append("single-session sample is too small for a locked baseline")

    return {
        "session_id": payload.get("session_id"),
        "schema_version": payload.get("schema_version"),
        "thresholds": {
            "depth_target_knee_deg": depth_target_knee_deg,
            "torso_warn_deg": torso_warn_deg,
        },
        "frames": {
            "total": len(frames),
            "gate_pass": gate_pass_frames,
            "gate_pass_rate": round(gate_pass_frames / len(frames), 4) if frames else None,
            "live": live_frames,
            "live_rate": round(live_frames / len(frames), 4) if frames else None,
        },
        "reps": {
            "total": len(reps),
            "depth_issue_count": len(depth_flags),
            "torso_issue_count": len(torso_flags),
            "knee_angle": {
                "min": round(min(knee_angles), 2) if knee_angles else None,
                "median": round(median(knee_angles), 2) if knee_angles else None,
                "p90": round(percentile(knee_angles, 0.9), 2) if knee_angles else None,
                "max": round(max(knee_angles), 2) if knee_angles else None,
            },
            "torso_angle": {
                "mean": round(mean(torso_angles), 2) if torso_angles else None,
                "max": round(max(torso_angles), 2) if torso_angles else None,
            },
        },
        "baseline_readiness": baseline_readiness,
        "baseline_notes": reasons,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize one cleaned squat session against locked baseline thresholds.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--depth-target-knee-deg", type=float, default=100.0)
    parser.add_argument("--torso-warn-deg", type=float, default=45.0)
    args = parser.parse_args()

    payload = load_json(args.input)
    summary = summarize(payload, args.depth_target_knee_deg, args.torso_warn_deg)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))
        print(f"Wrote baseline summary to {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
