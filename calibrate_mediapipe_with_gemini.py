#!/usr/bin/env python3
"""
Calibrate MediaPipe with Gemini Vision Teacher

Sends FULL raw pose trajectory + sampled video frames to Gemini and lets it:
1. Identify reps from the video frames
2. Parse all the MediaPipe pose data
3. Identify where MediaPipe angles/phases don't match what the video shows
4. Produce calibration recommendations

This is the ground-truth calibration approach.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None


CALIBRATION_SYSTEM = """You are calibrating a MediaPipe pose estimation pipeline for a squat coaching app.

You are given:
1. Video frames from a squat session
2. The FULL raw MediaPipe pose data stream (all frames with angles, phases, confidence)

Your job is to:
1. Watch the video frames and identify what ACTUALLY happened
2. Compare against what MediaPipe detected
3. Identify specific calibration issues

Return strict JSON:
{
  "video_observations": {
    "actual_rep_count": number,
    "camera_angle": "side" | "quarter" | "front" | "other",
    "visibility": {
      "head_always_visible": boolean,
      "feet_always_visible": boolean,
      "knees_clear": boolean,
      "hips_clear": boolean
    },
    "clothing_issues": string | null,
    "occlusion_events": [string]
  },
  "mediapipe_analysis": {
    "total_frames_analyzed": number,
    "phase_distribution": {"standing": number, "descent": number, "bottom": number, "ascent": number},
    "knee_angle_range": {"min": number, "max": number},
    "confidence_range": {"min": number, "max": number},
    "rep_count_from_angles": number
  },
  "calibration_issues": [
    {
      "issue_type": "depth_underestimate" | "depth_overestimate" | "phase_mismatch" | "angle_bias" | "visibility_mismatch" | "rep_miss" | "false_rep",
      "description": string,
      "evidence_from_video": string,
      "evidence_from_pose": string,
      "suggested_fix": string
    }
  ],
  "setup_gate_assessment": {
    "feet_visibility_correct": boolean,
    "actual_feet_status": string,
    "recommendation": string
  },
  "overall_mediapipe_accuracy": "good" | "moderate" | "poor",
  "priority_fixes": [string]
}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate MediaPipe using Gemini as ground truth")
    parser.add_argument("--video-glob", default="/Users/jwalinshah/Downloads/*.webm")
    parser.add_argument("--json-dir", default="/Users/jwalinshah/Downloads")
    parser.add_argument("--output-dir", default="data/results/calibration")
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing: pip install google-genai")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY")


def find_videos(pattern, limit):
    base = Path("/") if pattern.startswith("/") else Path(".")
    relative = pattern.lstrip("/")
    files = sorted(base.glob(relative), key=lambda p: p.stat().st_mtime, reverse=True)
    seen = set()
    unique = []
    for f in files:
        base_name = re.sub(r'\(\d+\)', '', f.stem)
        if base_name not in seen:
            seen.add(base_name)
            unique.append(f)
    return unique[:limit]


def find_matching_json(video_path, json_dir):
    video_stem = video_path.stem
    base_stem = re.sub(r'\(\d+\)$', '', video_stem).strip()
    json_dir = Path(json_dir)

    for json_file in json_dir.glob("*.json"):
        json_stem = json_file.stem
        video_ts = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', base_stem)
        json_ts = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', json_stem)
        if video_ts and json_ts and video_ts.group() == json_ts.group():
            return json_file
    return None


def extract_frames(video_path, output_dir, num_frames):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_size = video_path.stat().st_size
    estimated_duration = max(10, min(file_size / 500000, 120))

    timestamps = [estimated_duration * i / (num_frames - 1) for i in range(num_frames)]
    frame_paths = []

    for idx, ts in enumerate(timestamps):
        output_path = output_dir / f"frame-{idx:02d}-{int(ts*1000)}ms.jpg"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "2", "-update", "1", str(output_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
            )
        except subprocess.TimeoutExpired:
            continue
        if output_path.exists() and output_path.stat().st_size > 1000:
            frame_paths.append(output_path)

    return frame_paths


def load_json(path):
    return json.loads(Path(path).read_text())


def extract_pose_trajectory(payload):
    """Extract the full pose trajectory data for Gemini to analyze."""
    frames = payload.get("frames", [])

    # Build compact trajectory representation
    trajectory = []
    for f in frames:
        live = f.get("live")
        if not live:
            continue
        trajectory.append({
            "ts": f.get("timestamp_ms"),
            "phase": live.get("phase"),
            "knee": round(live.get("knee_angle", 0), 1) if live.get("knee_angle") else None,
            "hip": round(live.get("hip_angle", 0), 1) if live.get("hip_angle") else None,
            "torso": round(live.get("torso_angle", 0), 1) if live.get("torso_angle") else None,
            "shin": round(live.get("shin_angle", 0), 1) if live.get("shin_angle") else None,
            "hbk": live.get("hip_below_knee"),
            "conf": round(f.get("confidence_avg", 0), 2),
            "gate": f.get("gate_pass"),
            "prov": live.get("provisional"),
            "setup": {
                "head": (f.get("setup") or {}).get("head_visible"),
                "feet": (f.get("setup") or {}).get("feet_visible"),
                "center": (f.get("setup") or {}).get("centered"),
                "side": (f.get("setup") or {}).get("side_view"),
            }
        })

    # Sample trajectory to keep within token limits (every Nth frame)
    if len(trajectory) > 200:
        step = len(trajectory) // 200
        trajectory = trajectory[::step]

    return trajectory


def build_statistics(trajectory):
    """Build summary statistics from trajectory."""
    knees = [f["knee"] for f in trajectory if f["knee"]]
    confs = [f["conf"] for f in trajectory if f["conf"]]
    phases = {}
    for f in trajectory:
        p = f.get("phase")
        if p:
            phases[p] = phases.get(p, 0) + 1

    setup_issues = {
        "head_not_visible": sum(1 for f in trajectory if not (f.get("setup") or {}).get("head")),
        "feet_not_visible": sum(1 for f in trajectory if not (f.get("setup") or {}).get("feet")),
        "not_centered": sum(1 for f in trajectory if not (f.get("setup") or {}).get("center")),
        "not_side_view": sum(1 for f in trajectory if not (f.get("setup") or {}).get("side")),
    }

    return {
        "total_frames": len(trajectory),
        "knee_range": {"min": round(min(knees), 1), "max": round(max(knees), 1)} if knees else None,
        "conf_range": {"min": round(min(confs), 2), "max": round(max(confs), 2)} if confs else None,
        "phase_counts": phases,
        "gate_passed": sum(1 for f in trajectory if f.get("gate")),
        "setup_issues": setup_issues,
    }


def calibrate_session(client, model, video_path, json_path, output_dir, frames_per_video):
    """Run full calibration on a single session."""
    video_name = video_path.stem
    session_dir = output_dir / video_name
    frames_dir = session_dir / "frames"

    print(f"\n{'='*60}")
    print(f"Calibrating: {video_path.name}")

    # Extract video frames
    frame_paths = extract_frames(video_path, frames_dir, frames_per_video)
    if not frame_paths:
        return {"error": "no_frames", "video": str(video_path)}
    print(f"  Extracted {len(frame_paths)} video frames")

    # Load pose data
    if not json_path or not json_path.exists():
        return {"error": "no_json", "video": str(video_path)}

    payload = load_json(json_path)
    trajectory = extract_pose_trajectory(payload)
    stats = build_statistics(trajectory)
    print(f"  Loaded {stats['total_frames']} pose frames, {stats['gate_passed']} gated")
    print(f"  Knee range: {stats['knee_range']}")
    print(f"  Phase distribution: {stats['phase_counts']}")

    # Upload frames to Gemini
    uploaded = [client.files.upload(file=str(p)) for p in frame_paths]
    print(f"  Uploaded {len(uploaded)} frames to Gemini")

    # Build the calibration prompt
    prompt = f"""Calibrate MediaPipe for this squat session.

## Video Frames
{len(frame_paths)} evenly-sampled frames from the session video are attached.

## MediaPipe Pose Trajectory
The following is the FULL pose data captured by MediaPipe (sampled to {len(trajectory)} frames):

```json
{json.dumps(trajectory[:50], indent=1)}
... ({len(trajectory) - 50} more frames) ...
{json.dumps(trajectory[-20:], indent=1)}
```

## MediaPipe Summary Statistics
{json.dumps(stats, indent=2)}

## Session Metadata
- Camera angle: {payload.get('camera_angle')}
- Camera height: {payload.get('camera_height')}
- Session ID: {payload.get('session_id')}

Please analyze:
1. What the video ACTUALLY shows (rep count, form, visibility)
2. What MediaPipe detected (from the trajectory data)
3. Where they disagree and why
4. Specific calibration recommendations
"""

    # Call Gemini
    print("  Calling Gemini for calibration analysis...")
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=[*uploaded, prompt],
        config={
            "system_instruction": CALIBRATION_SYSTEM,
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    print(f"  Gemini response: {elapsed_ms}ms")

    text = getattr(response, "text", None)
    if not text:
        return {"error": "empty_response", "video": str(video_path)}

    try:
        calibration = json.loads(text)
    except json.JSONDecodeError:
        calibration = {"raw_text": text}

    result = {
        "video": str(video_path),
        "json": str(json_path),
        "video_frames": len(frame_paths),
        "pose_frames": stats["total_frames"],
        "pose_stats": stats,
        "trajectory_sample": trajectory[:10],  # First 10 for reference
        "calibration": calibration,
        "latency_ms": elapsed_ms,
    }

    # Save result
    session_dir.mkdir(parents=True, exist_ok=True)
    result_path = session_dir / "calibration.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {result_path}")

    # Print key findings
    if isinstance(calibration, dict):
        issues = calibration.get("calibration_issues", [])
        if issues:
            print(f"  Found {len(issues)} calibration issues:")
            for issue in issues[:3]:
                print(f"    - {issue.get('issue_type')}: {issue.get('description', '')[:60]}")
        accuracy = calibration.get("overall_mediapipe_accuracy", "unknown")
        print(f"  MediaPipe accuracy: {accuracy}")

    return result


def build_aggregate(results):
    """Build aggregate calibration report."""
    all_issues = []
    accuracy_counts = {"good": 0, "moderate": 0, "poor": 0, "unknown": 0}
    setup_findings = {"feet_correct": 0, "feet_incorrect": 0}

    for r in results:
        if "error" in r:
            continue
        cal = r.get("calibration", {})
        if isinstance(cal, dict):
            for issue in cal.get("calibration_issues", []):
                all_issues.append({
                    "session": r.get("video"),
                    **issue
                })
            accuracy = cal.get("overall_mediapipe_accuracy", "unknown")
            accuracy_counts[accuracy] = accuracy_counts.get(accuracy, 0) + 1

            gate = cal.get("setup_gate_assessment", {})
            if gate.get("feet_visibility_correct"):
                setup_findings["feet_correct"] += 1
            else:
                setup_findings["feet_incorrect"] += 1

    # Count issue types
    issue_type_counts = {}
    for issue in all_issues:
        t = issue.get("issue_type", "unknown")
        issue_type_counts[t] = issue_type_counts.get(t, 0) + 1

    return {
        "total_sessions": len(results),
        "successful": len([r for r in results if "error" not in r]),
        "accuracy_distribution": accuracy_counts,
        "issue_type_counts": sorted(issue_type_counts.items(), key=lambda x: -x[1]),
        "setup_gate_findings": setup_findings,
        "all_issues": all_issues,
        "priority_fixes": list(set(
            fix
            for r in results if isinstance(r.get("calibration"), dict)
            for fix in r["calibration"].get("priority_fixes", [])
        )),
    }


def main():
    args = parse_args()
    require_sdk()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    video_paths = find_videos(args.video_glob, args.limit)
    if not video_paths:
        raise SystemExit(f"No videos found: {args.video_glob}")

    print(f"Found {len(video_paths)} videos to calibrate")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for video_path in video_paths:
        json_path = find_matching_json(video_path, args.json_dir)
        result = calibrate_session(
            client, args.model, video_path, json_path,
            output_dir, args.frames_per_video
        )
        results.append(result)

    # Build aggregate
    aggregate = build_aggregate(results)

    # Save reports
    batch_path = output_dir / "batch_calibration.json"
    batch_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "sessions": results,
        "aggregate": aggregate,
    }, indent=2))

    print(f"\n{'='*60}")
    print(f"Batch calibration: {batch_path}")
    print(f"\nAggregate Results:")
    print(f"  Sessions: {aggregate['total_sessions']}")
    print(f"  Successful: {aggregate['successful']}")
    print(f"  Accuracy: {aggregate['accuracy_distribution']}")
    print(f"\nCommon Issues:")
    for issue_type, count in aggregate["issue_type_counts"][:5]:
        print(f"  {issue_type}: {count}")
    print(f"\nSetup Gate Findings:")
    print(f"  Feet visibility correct: {aggregate['setup_gate_findings']['feet_correct']}")
    print(f"  Feet visibility incorrect: {aggregate['setup_gate_findings']['feet_incorrect']}")
    print(f"\nPriority Fixes:")
    for fix in aggregate["priority_fixes"][:5]:
        print(f"  - {fix}")


if __name__ == "__main__":
    main()
