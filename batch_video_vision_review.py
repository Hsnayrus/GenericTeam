#!/usr/bin/env python3
"""
Batch Video Vision Review - Vision Teacher for Squat Coach

Runs open-ended Gemini vision analysis on all session videos to identify
what the pose pipeline missed or distorted. Produces a disagreement report
comparing vision teacher output vs pose teacher output.

Output: framing issues, occlusion, ankle loss, side-view drift, depth misread,
torso-angle bias, phase misses
"""
import argparse
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None


VISION_TEACHER_SYSTEM = """You are an expert squat coach reviewing video footage.
Your job is to identify issues that a pose estimation pipeline might miss or distort.

Analyze the squat video frames and return strict JSON with this schema:
{
  "rep_count_observed": number | null,
  "squat_phases_seen": ["standing", "descent", "bottom", "ascent"],
  "framing_issues": {
    "head_cropped": boolean,
    "feet_cropped": boolean,
    "not_centered": boolean,
    "not_side_view": boolean,
    "too_far": boolean,
    "too_close": boolean,
    "description": string
  },
  "pose_quality_issues": {
    "occlusion": boolean,
    "ankle_loss": boolean,
    "knee_obscured": boolean,
    "torso_unclear": boolean,
    "description": string
  },
  "form_observations": {
    "depth_reached": "parallel" | "below_parallel" | "shallow" | "unknown",
    "knee_tracking": "good" | "valgus" | "varus" | "unknown",
    "torso_angle_estimate": number | null,
    "forward_lean_issue": boolean,
    "heel_rise": boolean,
    "description": string
  },
  "phase_detection_notes": string,
  "what_pose_pipeline_might_miss": [string],
  "confidence": number
}

Rules:
- Focus on what a MediaPipe-style pose pipeline running at 30fps might get wrong.
- Identify moments where landmarks would be lost or unreliable.
- Note camera angle issues that degrade pose accuracy.
- Estimate form quality from visual observation.
- Be specific about timestamps or frame positions when relevant.
"""

POSE_COMPARISON_SYSTEM = """You are comparing pose pipeline data against video observations.
Given the pose JSON summary and vision observations, identify disagreements.

Return strict JSON:
{
  "rep_count_match": boolean,
  "rep_count_pose": number | null,
  "rep_count_vision": number | null,
  "phase_disagreements": [{"timestamp_ms": number, "pose_says": string, "vision_says": string}],
  "depth_disagreement": boolean,
  "depth_pose_says": string,
  "depth_vision_says": string,
  "framing_blocked_reps": boolean,
  "pose_confidence_vs_vision": string,
  "key_disagreements": [string],
  "recommendation": string
}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gemini vision analysis on all session videos")
    parser.add_argument("--video-glob", default="/Users/jwalinshah/Downloads/*.webm")
    parser.add_argument("--json-dir", default="/Users/jwalinshah/Downloads")
    parser.add_argument("--output-dir", default="data/results/vision_review")
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing: install google-genai first")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY")


def find_videos(pattern):
    """Find video files matching glob pattern."""
    base = Path("/") if pattern.startswith("/") else Path(".")
    relative = pattern.lstrip("/")
    files = sorted(base.glob(relative), key=lambda p: p.stat().st_mtime, reverse=True)
    # Dedupe by base name (ignore (1), (2) suffixes)
    seen = set()
    unique = []
    for f in files:
        base_name = re.sub(r'\(\d+\)', '', f.stem)
        if base_name not in seen:
            seen.add(base_name)
            unique.append(f)
    return unique


def find_matching_json(video_path, json_dir):
    """Find JSON session file that matches a video file."""
    video_stem = video_path.stem
    # Handle (1), (2) suffixes
    base_stem = re.sub(r'\(\d+\)$', '', video_stem).strip()
    json_dir = Path(json_dir)
    candidates = [
        json_dir / f"{video_stem}.json",
        json_dir / f"{base_stem}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fuzzy match by session ID pattern - look for overlapping parts
    for json_file in json_dir.glob("*.json"):
        json_stem = json_file.stem
        # Check if video session ID matches json session ID
        if base_stem == json_stem or video_stem == json_stem:
            return json_file
        # Check for partial match on timestamp portion
        video_ts_match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', base_stem)
        json_ts_match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', json_stem)
        if video_ts_match and json_ts_match and video_ts_match.group() == json_ts_match.group():
            return json_file
    return None


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def extract_frames(video_path, output_dir, num_frames):
    """Extract evenly-spaced frames from video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For webm, probe duration or use file size estimate
    duration = get_video_duration(video_path)

    # For streaming webm without duration, estimate from file size
    if not duration or duration < 1:
        file_size = video_path.stat().st_size
        # Rough estimate: ~500KB/sec for 720p webm
        estimated_duration = file_size / 500000
        duration = max(10, min(estimated_duration, 300))  # Clamp 10-300 sec

    # Sample frames evenly across duration
    timestamps = [duration * i / (num_frames - 1) for i in range(num_frames)]
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
    """Load JSON file."""
    return json.loads(Path(path).read_text())


def derive_rep_summaries_from_all_frames(payload):
    """Derive rep summaries from live_cue_side_all or frames, even without gate_pass."""
    # Try datasets.live_cue_side_all first (new format)
    all_frames = (payload.get("datasets") or {}).get("live_cue_side_all", [])
    if not all_frames:
        # Fall back to frames with live data
        all_frames = [
            {
                "timestamp_ms": f.get("timestamp_ms"),
                "phase": (f.get("live") or {}).get("phase"),
                "rep_count": (f.get("live") or {}).get("rep_count"),
                "knee_angle": (f.get("live") or {}).get("knee_angle"),
                "hip_angle": (f.get("live") or {}).get("hip_angle"),
                "torso_angle": (f.get("live") or {}).get("torso_angle"),
                "shin_angle": (f.get("live") or {}).get("shin_angle"),
                "hip_below_knee": (f.get("live") or {}).get("hip_below_knee"),
                "provisional": (f.get("live") or {}).get("provisional", True),
                "gate_pass": f.get("gate_pass", False),
                "rep_mode_active": f.get("rep_mode_active", False),
                "confidence_avg": f.get("confidence_avg"),
                "confidence_min": f.get("confidence_min"),
            }
            for f in payload.get("frames", [])
            if f.get("live")
        ]

    if not all_frames:
        return []

    # Group by phase transitions to detect potential reps
    # A rep is: standing -> descent -> bottom -> ascent -> standing
    reps = []
    current_rep_frames = []
    last_phase = None
    rep_number = 0

    for frame in all_frames:
        phase = frame.get("phase")
        knee_angle = frame.get("knee_angle")

        if phase == "standing" and last_phase == "ascent":
            # Completed a rep
            if current_rep_frames:
                rep_number += 1
                reps.append(summarize_rep_frames(rep_number, current_rep_frames))
            current_rep_frames = [frame]
        elif phase in ("descent", "bottom", "ascent"):
            current_rep_frames.append(frame)
        elif phase == "standing":
            current_rep_frames = [frame]

        last_phase = phase

    # Check for incomplete final rep
    if current_rep_frames:
        bottom_frames = [f for f in current_rep_frames if f.get("phase") == "bottom"]
        if bottom_frames:
            rep_number += 1
            reps.append(summarize_rep_frames(rep_number, current_rep_frames))

    return reps


def summarize_rep_frames(rep_number, frames):
    """Create a rep summary from raw frames."""
    bottom_frames = [f for f in frames if f.get("phase") == "bottom"]
    if bottom_frames:
        bottom = min(bottom_frames, key=lambda f: f.get("knee_angle", 999))
    else:
        bottom = min(frames, key=lambda f: f.get("knee_angle", 999))

    knees = [f["knee_angle"] for f in frames if isinstance(f.get("knee_angle"), (int, float))]
    confidences = [f["confidence_avg"] for f in frames if isinstance(f.get("confidence_avg"), (int, float))]

    start_ts = frames[0].get("timestamp_ms", 0)
    end_ts = frames[-1].get("timestamp_ms", 0)

    return {
        "rep_number": rep_number,
        "source": "derived_offline",
        "frame_count": len(frames),
        "provisional_count": sum(1 for f in frames if f.get("provisional")),
        "gate_pass_count": sum(1 for f in frames if f.get("gate_pass")),
        "bottom_frame": {
            "timestamp_ms": bottom.get("timestamp_ms"),
            "knee_angle": bottom.get("knee_angle"),
            "hip_angle": bottom.get("hip_angle"),
            "torso_angle": bottom.get("torso_angle"),
            "shin_angle": bottom.get("shin_angle"),
            "hip_below_knee": bottom.get("hip_below_knee"),
        },
        "tempo": {
            "total_ms": end_ts - start_ts,
        },
        "range": {
            "knee_angle_min": round(min(knees), 2) if knees else None,
            "knee_angle_max": round(max(knees), 2) if knees else None,
        },
        "confidence": {
            "avg": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "min": round(min(confidences), 3) if confidences else None,
        },
    }


def build_pose_summary(payload):
    """Build a summary of pose pipeline data for comparison."""
    reps = payload.get("rep_summaries") or derive_rep_summaries_from_all_frames(payload)

    # Get frame statistics
    frames = payload.get("frames", [])
    all_data = (payload.get("datasets") or {}).get("live_cue_side_all", [])

    gate_passed = sum(1 for f in frames if f.get("gate_pass"))
    rep_mode_frames = sum(1 for f in frames if f.get("rep_mode_active"))

    phase_counts = defaultdict(int)
    for f in all_data or frames:
        phase = f.get("phase") or (f.get("live") or {}).get("phase")
        if phase:
            phase_counts[phase] += 1

    # Build prose summaries for each rep
    rep_prose = []
    for rep in reps:
        bottom = rep.get("bottom_frame") or {}
        tempo = rep.get("tempo") or {}
        prose = (
            f"Rep {rep.get('rep_number')}: "
            f"knee {bottom.get('knee_angle')}°, "
            f"hip {bottom.get('hip_angle')}°, "
            f"torso {bottom.get('torso_angle')}°, "
            f"duration {tempo.get('total_ms')}ms, "
            f"hip_below_knee={bottom.get('hip_below_knee')}"
        )
        rep_prose.append(prose)

    return {
        "session_id": payload.get("session_id"),
        "camera_angle": payload.get("camera_angle"),
        "camera_height": payload.get("camera_height"),
        "total_frames": len(frames),
        "gate_passed_frames": gate_passed,
        "rep_mode_frames": rep_mode_frames,
        "rep_count": len(reps),
        "reps": reps,
        "rep_prose_summary": rep_prose,
        "phase_distribution": dict(phase_counts),
        "setup_issues": analyze_setup_issues(frames),
    }


def analyze_setup_issues(frames):
    """Analyze setup issues from frame data."""
    issues = {
        "head_not_visible": 0,
        "feet_not_visible": 0,
        "not_centered": 0,
        "not_side_view": 0,
    }
    for f in frames:
        setup = f.get("setup") or {}
        if not setup.get("head_visible"):
            issues["head_not_visible"] += 1
        if not setup.get("feet_visible"):
            issues["feet_not_visible"] += 1
        if not setup.get("centered"):
            issues["not_centered"] += 1
        if not setup.get("side_view"):
            issues["not_side_view"] += 1

    total = len(frames) or 1
    return {
        "total_frames": len(frames),
        "head_not_visible_pct": round(100 * issues["head_not_visible"] / total, 1),
        "feet_not_visible_pct": round(100 * issues["feet_not_visible"] / total, 1),
        "not_centered_pct": round(100 * issues["not_centered"] / total, 1),
        "not_side_view_pct": round(100 * issues["not_side_view"] / total, 1),
    }


def call_gemini_vision(client, model, frame_paths, system_prompt, user_prompt):
    """Call Gemini with uploaded frames."""
    uploaded = [client.files.upload(file=str(p)) for p in frame_paths]
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=[*uploaded, user_prompt],
        config={
            "system_instruction": system_prompt,
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    text = getattr(response, "text", None)
    if not text:
        return {"error": "empty_response"}, elapsed_ms
    try:
        return json.loads(text), elapsed_ms
    except json.JSONDecodeError:
        return {"raw_text": text}, elapsed_ms


def call_gemini_comparison(client, model, pose_summary, vision_result):
    """Call Gemini to compare pose vs vision findings."""
    prompt = (
        "Compare these pose pipeline observations against the vision teacher findings.\n\n"
        f"Pose Pipeline Data:\n{json.dumps(pose_summary, indent=2)}\n\n"
        f"Vision Teacher Observations:\n{json.dumps(vision_result, indent=2)}"
    )
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": POSE_COMPARISON_SYSTEM,
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    text = getattr(response, "text", None)
    if not text:
        return {"error": "empty_response"}, elapsed_ms
    try:
        return json.loads(text), elapsed_ms
    except json.JSONDecodeError:
        return {"raw_text": text}, elapsed_ms


def review_video(client, model, video_path, json_path, output_dir, frames_per_video):
    """Run full vision review on a single video."""
    video_name = video_path.stem
    session_output_dir = output_dir / video_name
    frames_dir = session_output_dir / "frames"

    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")

    # Extract frames
    frame_paths = extract_frames(video_path, frames_dir, frames_per_video)
    if not frame_paths:
        return {"error": "no_frames_extracted", "video": str(video_path)}

    print(f"  Extracted {len(frame_paths)} frames")

    # Load pose data if available
    pose_summary = None
    if json_path and json_path.exists():
        payload = load_json(json_path)
        pose_summary = build_pose_summary(payload)
        print(f"  Pose data: {pose_summary['rep_count']} reps, {pose_summary['gate_passed_frames']}/{pose_summary['total_frames']} gate passed")
    else:
        print("  No matching JSON session found")

    # Run vision analysis
    vision_prompt = (
        "Analyze these frames from a squat session video. "
        "Identify any squats performed, form issues, and problems that would affect pose estimation accuracy. "
        "Note camera positioning, visibility of key joints, and any occlusions."
    )
    vision_result, vision_ms = call_gemini_vision(
        client, model, frame_paths, VISION_TEACHER_SYSTEM, vision_prompt
    )
    print(f"  Vision analysis: {vision_ms}ms")

    # Run comparison if we have pose data
    comparison = None
    if pose_summary:
        comparison, comp_ms = call_gemini_comparison(client, model, pose_summary, vision_result)
        print(f"  Comparison: {comp_ms}ms")

    result = {
        "video": str(video_path),
        "json": str(json_path) if json_path else None,
        "frame_count": len(frame_paths),
        "frame_paths": [str(p) for p in frame_paths],
        "pose_summary": pose_summary,
        "vision_review": vision_result,
        "vision_latency_ms": vision_ms,
        "comparison": comparison,
    }

    # Save individual result
    session_output_dir.mkdir(parents=True, exist_ok=True)
    result_path = session_output_dir / "review.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {result_path}")

    return result


def build_aggregate_report(results):
    """Build aggregate disagreement report across all sessions."""
    total_sessions = len(results)
    sessions_with_pose = sum(1 for r in results if r.get("pose_summary"))

    all_disagreements = []
    missed_by_pose = defaultdict(int)
    framing_issues = defaultdict(int)

    for r in results:
        vision = r.get("vision_review") or {}
        comp = r.get("comparison") or {}
        pose = r.get("pose_summary") or {}

        # Collect what pose pipeline might miss
        for issue in vision.get("what_pose_pipeline_might_miss", []):
            missed_by_pose[issue] += 1

        # Collect framing issues
        framing = vision.get("framing_issues") or {}
        for key in ["head_cropped", "feet_cropped", "not_centered", "not_side_view"]:
            if framing.get(key):
                framing_issues[key] += 1

        # Collect key disagreements
        for d in comp.get("key_disagreements", []):
            all_disagreements.append({
                "session": r.get("video"),
                "disagreement": d,
            })

    return {
        "total_sessions": total_sessions,
        "sessions_with_pose_data": sessions_with_pose,
        "common_missed_by_pose": sorted(
            missed_by_pose.items(), key=lambda x: x[1], reverse=True
        ),
        "framing_issues_frequency": dict(framing_issues),
        "key_disagreements": all_disagreements,
        "recommendations": generate_recommendations(missed_by_pose, framing_issues),
    }


def generate_recommendations(missed_by_pose, framing_issues):
    """Generate recommendations based on findings."""
    recs = []

    if framing_issues.get("feet_cropped", 0) > 0:
        recs.append("Ankle tracking is frequently lost - consider stricter feet visibility requirements")

    if framing_issues.get("not_side_view", 0) > 0:
        recs.append("Side view drift detected - add real-time angle correction prompts")

    if any("occlusion" in k.lower() for k in missed_by_pose):
        recs.append("Limb occlusion affects depth estimation - consider multi-frame smoothing")

    if any("depth" in k.lower() for k in missed_by_pose):
        recs.append("Depth estimation unreliable - cross-validate knee angle with hip-below-knee check")

    return recs


def main():
    args = parse_args()
    require_sdk()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    video_paths = find_videos(args.video_glob)
    if not video_paths:
        raise SystemExit(f"No videos found matching {args.video_glob}")

    print(f"Found {len(video_paths)} unique videos to analyze")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for video_path in video_paths:
        json_path = find_matching_json(video_path, args.json_dir)

        if args.skip_existing:
            existing = output_dir / video_path.stem / "review.json"
            if existing.exists():
                print(f"Skipping {video_path.name} (already processed)")
                results.append(load_json(existing))
                continue

        result = review_video(
            client, args.model, video_path, json_path,
            output_dir, args.frames_per_video
        )
        results.append(result)

    # Build aggregate report
    aggregate = build_aggregate_report(results)
    aggregate_path = output_dir / "aggregate_report.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))

    # Build full batch report
    batch_report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "video_count": len(results),
        "sessions": results,
        "aggregate": aggregate,
    }
    batch_path = output_dir / "batch_vision_review.json"
    batch_path.write_text(json.dumps(batch_report, indent=2))

    print(f"\n{'='*60}")
    print(f"Batch report: {batch_path}")
    print(f"Aggregate report: {aggregate_path}")
    print(f"\nSummary:")
    print(f"  Sessions analyzed: {aggregate['total_sessions']}")
    print(f"  With pose data: {aggregate['sessions_with_pose_data']}")
    print(f"\nCommon issues missed by pose pipeline:")
    for issue, count in aggregate["common_missed_by_pose"][:5]:
        print(f"  - {issue}: {count}")
    print(f"\nRecommendations:")
    for rec in aggregate["recommendations"]:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()
