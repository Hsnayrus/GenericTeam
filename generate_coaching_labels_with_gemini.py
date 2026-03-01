#!/usr/bin/env python3
"""
Generate Coaching Labels with Gemini Vision Teacher

Sends video frames + pose features to Gemini and gets back coaching feedback.
Outputs JSONL for fine-tuning Gemma to replicate the coaching decisions.

This replaces hand-coded rules with learned coaching from a vision teacher.
"""
import argparse
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from random import Random

try:
    from google import genai
except ImportError:
    genai = None


COACHING_TEACHER_SYSTEM = """You are an expert squat coach generating training data for an AI coaching system.

You will receive:
1. Video frames from a squat session
2. Pose-derived features (angles, phases, confidence) for each rep

Your job is to generate the coaching feedback a good coach would give for each rep.

For each rep, decide:
- What feedback (if any) the user should hear
- The severity: "good" (positive reinforcement), "warn" (minor correction), "bad" (important fix needed), or "silent" (no feedback needed)
- Whether this feedback should be spoken aloud

Return strict JSON:
{
  "session_assessment": {
    "overall_form": "good" | "needs_work" | "poor",
    "primary_issue": string | null,
    "camera_quality": "good" | "acceptable" | "poor"
  },
  "rep_coaching": [
    {
      "rep_number": number,
      "phase_for_feedback": "bottom" | "ascent" | "descent" | "standing",
      "label": "good_rep" | "shallow_depth" | "knee_valgus" | "forward_lean" | "good_depth" | "silent" | "other",
      "feedback": string,
      "severity": "good" | "warn" | "bad" | "silent",
      "speak": boolean,
      "reasoning": string
    }
  ],
  "setup_coaching": {
    "needed": boolean,
    "feedback": string | null,
    "label": "framing_ok" | "cropped_feet" | "cropped_head" | "bad_side_view" | "too_far" | "too_close" | null
  }
}

Coaching principles:
- Be concise. Real-time cues must be 2-8 words.
- Don't over-coach. If a rep is fine, say so briefly or stay silent.
- Prioritize one issue at a time. Don't stack multiple corrections.
- Use actionable cues: "Push knees out" not "Your knees are caving"
- Good reps deserve occasional positive reinforcement, not every time.
- Depth is the most important form check for squats.
- If you can't assess something from the video, don't guess.

Example feedback:
- "Good depth" (good)
- "Go deeper" (warn)
- "Knees out" (bad)
- "Chest up" (warn)
- "Nice rep" (good)
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Generate coaching labels using Gemini")
    parser.add_argument("--video-glob", default="/Users/jwalinshah/Downloads/*.webm")
    parser.add_argument("--json-dir", default="/Users/jwalinshah/Downloads")
    parser.add_argument("--output-dir", default="data/coaching_labels")
    parser.add_argument("--output-jsonl", default="data/gemini_coaching.jsonl")
    parser.add_argument("--train-jsonl", default="data/gemini_coaching.train.jsonl")
    parser.add_argument("--eval-jsonl", default="data/gemini_coaching.eval.jsonl")
    parser.add_argument("--frames-per-video", type=int, default=16)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "models/gemini-2.5-pro"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--min-json-bytes", type=int, default=5000)
    parser.add_argument("--min-reps", type=int, default=1)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing: pip install google-genai")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY")


def find_videos(pattern, limit=None):
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

    if limit:
        unique = unique[:limit]
    return unique


def find_matching_json(video_path, json_dir):
    """Find JSON session file that matches a video file."""
    video_stem = video_path.stem
    base_stem = re.sub(r'\(\d+\)$', '', video_stem).strip()
    json_dir = Path(json_dir)

    # Direct matches
    for candidate in [json_dir / f"{video_stem}.json", json_dir / f"{base_stem}.json"]:
        if candidate.exists():
            return candidate

    # Timestamp match
    for json_file in json_dir.glob("*.json"):
        video_ts = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', base_stem)
        json_ts = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', json_file.stem)
        if video_ts and json_ts and video_ts.group() == json_ts.group():
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

    duration = get_video_duration(video_path)
    if not duration or duration < 1:
        file_size = video_path.stat().st_size
        duration = max(10, min(file_size / 500000, 300))

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
    return json.loads(Path(path).read_text())


def session_is_usable(json_path, min_json_bytes):
    if not json_path or not json_path.exists():
        return False, "missing_json"
    if json_path.stat().st_size < min_json_bytes:
        return False, "json_too_small"
    return True, None


def frame_rows_from_payload(payload):
    all_frames = (payload.get("datasets") or {}).get("live_cue_side_all", [])
    if all_frames:
        return all_frames
    return [
        {
            "timestamp_ms": f.get("timestamp_ms"),
            "phase": (f.get("live") or {}).get("phase"),
            "rep_count": (f.get("live") or {}).get("rep_count"),
            "knee_angle": (f.get("live") or {}).get("knee_angle"),
            "hip_angle": (f.get("live") or {}).get("hip_angle"),
            "torso_angle": (f.get("live") or {}).get("torso_angle"),
            "shin_angle": (f.get("live") or {}).get("shin_angle"),
            "hip_below_knee": (f.get("live") or {}).get("hip_below_knee"),
            "confidence_avg": f.get("confidence_avg"),
            "confidence_min": f.get("confidence_min"),
        }
        for f in payload.get("frames", [])
        if f.get("live")
    ]


def derive_rep_summaries(payload):
    """Derive rep summaries from pose data."""
    all_frames = frame_rows_from_payload(payload)
    if not all_frames:
        return []

    # Group by rep transitions
    reps = []
    current_rep_frames = []
    last_phase = None
    rep_number = 0

    for frame in all_frames:
        phase = frame.get("phase")

        if phase == "standing" and last_phase == "ascent":
            if current_rep_frames:
                rep_number += 1
                reps.append(summarize_rep(rep_number, current_rep_frames))
            current_rep_frames = [frame]
        elif phase in ("descent", "descending", "bottom", "ascent", "ascending"):
            current_rep_frames.append(frame)
        elif phase == "standing":
            current_rep_frames = [frame]

        last_phase = phase

    # Handle incomplete final rep
    if current_rep_frames:
        bottom_frames = [f for f in current_rep_frames if f.get("phase") == "bottom"]
        if bottom_frames:
            rep_number += 1
            reps.append(summarize_rep(rep_number, current_rep_frames))

    if reps:
        return reps
    return derive_rep_summaries_from_knee_signal(all_frames)


def derive_rep_summaries_from_knee_signal(all_frames, min_depth=125, standing_threshold=155, min_gap_frames=45):
    usable = [frame for frame in all_frames if isinstance(frame.get("knee_angle"), (int, float))]
    if len(usable) < 10:
        return []

    minima = []
    last_idx = -min_gap_frames
    for idx in range(2, len(usable) - 2):
        knee = usable[idx]["knee_angle"]
        if knee >= min_depth:
            continue
        neighborhood = [usable[idx + offset]["knee_angle"] for offset in (-2, -1, 0, 1, 2)]
        if knee != min(neighborhood):
            continue
        if idx - last_idx < min_gap_frames:
            if minima and knee < usable[minima[-1]]["knee_angle"]:
                minima[-1] = idx
                last_idx = idx
            continue
        minima.append(idx)
        last_idx = idx

    reps = []
    rep_number = 0
    for idx in minima:
        left = idx
        while left > 0 and usable[left]["knee_angle"] < standing_threshold:
            left -= 1
        right = idx
        while right < len(usable) - 1 and usable[right]["knee_angle"] < standing_threshold:
            right += 1
        segment = usable[left:right + 1]
        if len(segment) < 8:
            continue
        if min(frame["knee_angle"] for frame in segment) >= min_depth:
            continue
        rep_number += 1
        reps.append(summarize_rep(rep_number, segment))
    return reps


def summarize_rep(rep_number, frames):
    """Create compact rep summary for Gemini."""
    bottom_frames = [f for f in frames if f.get("phase") == "bottom"]
    if bottom_frames:
        bottom = min(bottom_frames, key=lambda f: f.get("knee_angle") or 999)
    else:
        bottom = min(frames, key=lambda f: f.get("knee_angle") or 999)

    knees = [f["knee_angle"] for f in frames if isinstance(f.get("knee_angle"), (int, float))]

    return {
        "rep_number": rep_number,
        "frame_count": len(frames),
        "bottom": {
            "knee_angle": round(bottom.get("knee_angle") or 0, 1),
            "hip_angle": round(bottom.get("hip_angle") or 0, 1),
            "torso_angle": round(bottom.get("torso_angle") or 0, 1),
            "shin_angle": round(bottom.get("shin_angle") or 0, 1),
            "hip_below_knee": bottom.get("hip_below_knee"),
            "confidence_min": round(bottom.get("confidence_min") or 0, 2),
        },
        "knee_range": {
            "min": round(min(knees), 1) if knees else None,
            "max": round(max(knees), 1) if knees else None,
        },
    }


def normalize_rep(rep):
    bottom = rep.get("bottom") or rep.get("bottom_frame") or {}
    knees = rep.get("knee_range") or rep.get("range") or {}
    return {
        "rep_number": rep.get("rep_number"),
        "frame_count": rep.get("frame_count"),
        "bottom": {
            "knee_angle": bottom.get("knee_angle"),
            "hip_angle": bottom.get("hip_angle"),
            "torso_angle": bottom.get("torso_angle"),
            "shin_angle": bottom.get("shin_angle"),
            "hip_below_knee": bottom.get("hip_below_knee"),
            "confidence_min": bottom.get("confidence_min"),
        },
        "knee_range": {
            "min": knees.get("min") if "min" in knees else knees.get("knee_angle_min"),
            "max": knees.get("max") if "max" in knees else knees.get("knee_angle_max"),
        },
    }


def build_pose_context(payload):
    """Build pose context string for Gemini prompt."""
    reps = payload.get("rep_summaries") or derive_rep_summaries(payload)
    reps = [normalize_rep(rep) for rep in reps if rep.get("rep_number") is not None]

    if not reps:
        return None, []

    lines = [f"Session has {len(reps)} detected reps:\n"]
    for rep in reps:
        bottom = rep.get("bottom", {})
        lines.append(
            f"Rep {rep['rep_number']}: "
            f"knee={bottom.get('knee_angle')}° "
            f"hip={bottom.get('hip_angle')}° "
            f"torso={bottom.get('torso_angle')}° "
            f"hip_below_knee={bottom.get('hip_below_knee')} "
            f"conf={bottom.get('confidence_min')}"
        )

    return "\n".join(lines), reps


def write_jsonl(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def split_examples(examples, eval_ratio, seed):
    if not examples:
        return [], []
    grouped = defaultdict(list)
    for example in examples:
        session_key = example["id"].rsplit("-rep", 1)[0].rsplit("-setup", 1)[0]
        grouped[session_key].append(example)
    session_ids = sorted(grouped)
    Random(seed).shuffle(session_ids)
    eval_count = max(1, round(len(session_ids) * eval_ratio)) if len(session_ids) > 1 else 0
    eval_sessions = set(session_ids[:eval_count])
    train_rows = []
    eval_rows = []
    for session_id in session_ids:
        destination = eval_rows if session_id in eval_sessions else train_rows
        destination.extend(grouped[session_id])
    return train_rows, eval_rows


def call_gemini_coaching(client, model, frame_paths, pose_context):
    """Call Gemini to generate coaching labels."""
    uploaded = [client.files.upload(file=str(p)) for p in frame_paths]

    prompt = f"""Analyze this squat session and generate coaching feedback for each rep.

## Pose Pipeline Data
{pose_context}

## Instructions
Watch the video frames and generate appropriate coaching cues.
For each rep, provide the feedback a real coach would give in real-time.
Be concise - these cues will be spoken aloud during the workout.
"""

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=[*uploaded, prompt],
        config={
            "system_instruction": COACHING_TEACHER_SYSTEM,
            "temperature": 0.3,  # Slight variation in phrasing
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


def convert_to_training_format(session_id, pose_reps, coaching_result):
    """Convert Gemini output to fine-tuning JSONL format."""
    examples = []

    rep_coaching = coaching_result.get("rep_coaching", [])

    for coaching in rep_coaching:
        rep_num = coaching.get("rep_number")

        # Find matching pose data
        pose_rep = next((r for r in pose_reps if r.get("rep_number") == rep_num), None)
        if not pose_rep:
            continue

        bottom = pose_rep.get("bottom", {})

        # Skip silent coaching (no feedback needed)
        if coaching.get("severity") == "silent":
            continue

        example = {
            "id": f"{session_id}-rep{rep_num:02d}",
            "label": coaching.get("label", "other"),
            "source": "gemini_teacher",
            "input": {
                "phase": coaching.get("phase_for_feedback", "bottom"),
                "rep_number": rep_num,
                "knee_angle": bottom.get("knee_angle"),
                "hip_angle": bottom.get("hip_angle"),
                "torso_angle": bottom.get("torso_angle"),
                "shin_angle": bottom.get("shin_angle"),
                "hip_below_knee": bottom.get("hip_below_knee"),
                "confidence_min": bottom.get("confidence_min"),
            },
            "expected_feedback": coaching.get("feedback"),
            "severity": coaching.get("severity"),
            "speak": coaching.get("speak", True),
            "reasoning": coaching.get("reasoning"),
        }
        examples.append(example)

    # Add setup coaching if needed
    setup = coaching_result.get("setup_coaching", {})
    if setup.get("needed") and setup.get("feedback"):
        examples.append({
            "id": f"{session_id}-setup",
            "label": setup.get("label") or "setup_issue",
            "source": "gemini_teacher",
            "input": {
                "phase": "standing",
                "rep_number": 0,
                "is_setup": True,
            },
            "expected_feedback": setup.get("feedback"),
            "severity": "warn",
            "speak": True,
        })

    return examples


def process_video(client, model, video_path, json_path, output_dir, frames_per_video, min_json_bytes, min_reps):
    """Process a single video and generate coaching labels."""
    video_name = video_path.stem
    session_dir = output_dir / video_name
    frames_dir = session_dir / "frames"

    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")

    usable, reason = session_is_usable(json_path, min_json_bytes)
    if not usable:
        print(f"  Skipping session: {reason}")
        return {"error": reason, "video": str(video_path), "json": str(json_path) if json_path else None}, []

    payload = load_json(json_path)
    pose_context, pose_reps = build_pose_context(payload)

    if not pose_context or len(pose_reps) < min_reps:
        print("  No usable reps detected in pose data - skipping")
        return {"error": "no_reps", "video": str(video_path), "json": str(json_path)}, []

    print(f"  Found {len(pose_reps)} reps in pose data")

    # Extract frames only after the JSON clears the basic filters.
    frame_paths = extract_frames(video_path, frames_dir, frames_per_video)
    if not frame_paths:
        return {"error": "no_frames", "video": str(video_path), "json": str(json_path)}, []
    print(f"  Extracted {len(frame_paths)} frames")

    # Call Gemini
    print("  Calling Gemini for coaching labels...")
    coaching_result, latency_ms = call_gemini_coaching(
        client, model, frame_paths, pose_context
    )
    print(f"  Gemini response: {latency_ms}ms")

    if "error" in coaching_result:
        print(f"  Error: {coaching_result}")
        return coaching_result, []

    # Convert to training format
    session_id = payload.get("session_id", video_name)
    examples = convert_to_training_format(session_id, pose_reps, coaching_result)
    print(f"  Generated {len(examples)} training examples")

    # Print coaching preview
    for ex in examples[:3]:
        print(f"    Rep {ex['input'].get('rep_number')}: [{ex['severity']}] {ex['expected_feedback']}")

    # Save session result
    result = {
        "video": str(video_path),
        "json": str(json_path),
        "session_id": session_id,
        "frame_count": len(frame_paths),
        "rep_count": len(pose_reps),
        "coaching": coaching_result,
        "examples": examples,
        "latency_ms": latency_ms,
    }

    session_dir.mkdir(parents=True, exist_ok=True)
    result_path = session_dir / "coaching.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved: {result_path}")

    return result, examples


def main():
    args = parse_args()
    require_sdk()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    video_paths = find_videos(args.video_glob, args.limit)
    if not video_paths:
        raise SystemExit(f"No videos found: {args.video_glob}")

    print(f"Found {len(video_paths)} videos to process")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_examples = []

    for video_path in video_paths:
        json_path = find_matching_json(video_path, args.json_dir)

        if args.skip_existing:
            existing = output_dir / video_path.stem / "coaching.json"
            if existing.exists():
                print(f"Skipping {video_path.name} (already processed)")
                cached = load_json(existing)
                all_results.append(cached)
                all_examples.extend(cached.get("examples", []))
                continue

        result, examples = process_video(
            client, args.model, video_path, json_path,
            output_dir, args.frames_per_video, args.min_json_bytes, args.min_reps
        )
        all_results.append(result)
        all_examples.extend(examples)

    train_examples, eval_examples = split_examples(all_examples, args.eval_ratio, args.seed)

    # Write JSONL outputs
    write_jsonl(args.output_jsonl, all_examples)
    write_jsonl(args.train_jsonl, train_examples)
    write_jsonl(args.eval_jsonl, eval_examples)

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Videos processed: {len(all_results)}")
    print(f"  Training examples: {len(all_examples)}")
    print(f"  JSONL output: {args.output_jsonl}")
    print(f"  Train split: {args.train_jsonl} ({len(train_examples)} examples)")
    print(f"  Eval split: {args.eval_jsonl} ({len(eval_examples)} examples)")

    # Label distribution
    labels = defaultdict(int)
    for ex in all_examples:
        labels[ex.get("label", "unknown")] += 1
    print(f"\nLabel distribution:")
    for label, count in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    # Save batch report
    batch_path = output_dir / "batch_coaching.json"
    batch_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "video_count": len(all_results),
        "example_count": len(all_examples),
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "label_distribution": dict(labels),
        "sessions": all_results,
    }, indent=2))
    print(f"  Batch report: {batch_path}")


if __name__ == "__main__":
    main()
