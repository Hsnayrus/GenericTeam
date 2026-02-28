#!/usr/bin/env python3
"""
Derive Rep Summaries Offline

Processes squat session exports and derives rep summaries from the raw pose stream
(live_cue_side_all), even when live rep counting failed due to setup gating issues.

This enables analysis of sessions that recorded valid squat movements but never
entered "usable rep mode" because of feet visibility, centering, or other blockers.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive rep summaries offline from raw pose stream"
    )
    parser.add_argument("--input", required=True, help="Session JSON or glob pattern")
    parser.add_argument("--output-dir", default="data/results/derived_reps")
    parser.add_argument("--knee-threshold-standing", type=float, default=155,
                        help="Knee angle above which is 'standing'")
    parser.add_argument("--knee-threshold-bottom", type=float, default=120,
                        help="Knee angle below which is 'bottom'")
    parser.add_argument("--min-descent-frames", type=int, default=3,
                        help="Minimum frames in descent to count as rep")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum confidence to include frame")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text())


def get_all_pose_frames(payload):
    """Extract all pose frames from session, preferring live_cue_side_all."""
    # Try datasets.live_cue_side_all first (new export format)
    datasets = payload.get("datasets") or {}
    all_frames = datasets.get("live_cue_side_all", [])

    if all_frames:
        return all_frames

    # Fall back to extracting from frames array
    frames = payload.get("frames", [])
    extracted = []
    for f in frames:
        live = f.get("live")
        if not live:
            continue
        extracted.append({
            "timestamp_ms": f.get("timestamp_ms"),
            "phase": live.get("phase"),
            "rep_count": live.get("rep_count"),
            "knee_angle": live.get("knee_angle"),
            "hip_angle": live.get("hip_angle"),
            "torso_angle": live.get("torso_angle"),
            "shin_angle": live.get("shin_angle"),
            "hip_below_knee": live.get("hip_below_knee"),
            "tempo_ms": live.get("tempo_ms"),
            "provisional": live.get("provisional", True),
            "gate_pass": f.get("gate_pass", False),
            "rep_mode_active": f.get("rep_mode_active", False),
            "confidence_avg": f.get("confidence_avg"),
            "confidence_min": f.get("confidence_min"),
            "setup": f.get("setup", {}),
        })
    return extracted


def classify_phase(knee_angle, args):
    """Classify phase based on knee angle."""
    if knee_angle is None:
        return "unknown"
    if knee_angle > args.knee_threshold_standing:
        return "standing"
    elif knee_angle < args.knee_threshold_bottom:
        return "bottom"
    else:
        return "mid"  # Could be descent or ascent


def detect_reps_from_angles(frames, args):
    """
    Detect reps using knee angle trajectory analysis.

    A rep is detected when:
    1. Knee angle drops from standing (>155) to bottom (<120)
    2. Knee angle rises back towards standing

    This ignores the live phase detection which may have been gated.
    """
    if not frames:
        return []

    # Filter to frames with valid knee angles and sufficient confidence
    valid_frames = [
        f for f in frames
        if isinstance(f.get("knee_angle"), (int, float))
        and f.get("confidence_avg", 0) >= args.min_confidence
    ]

    if len(valid_frames) < 5:
        return []

    # State machine for rep detection
    reps = []
    current_rep_frames = []
    state = "seeking_standing"  # seeking_standing -> in_descent -> in_bottom -> in_ascent
    rep_number = 0

    for i, frame in enumerate(valid_frames):
        knee = frame["knee_angle"]
        phase = classify_phase(knee, args)

        if state == "seeking_standing":
            if phase == "standing":
                state = "at_standing"
                current_rep_frames = [frame]

        elif state == "at_standing":
            if phase == "mid":
                state = "in_descent"
                current_rep_frames.append(frame)
            elif phase == "standing":
                current_rep_frames = [frame]  # Reset

        elif state == "in_descent":
            current_rep_frames.append(frame)
            if phase == "bottom":
                state = "in_bottom"
            elif phase == "standing":
                # Aborted rep
                state = "at_standing"
                current_rep_frames = [frame]

        elif state == "in_bottom":
            current_rep_frames.append(frame)
            if phase == "mid":
                state = "in_ascent"
            elif phase == "standing":
                # Quick ascent, count it
                rep_number += 1
                reps.append(summarize_rep(rep_number, current_rep_frames, args))
                state = "at_standing"
                current_rep_frames = [frame]

        elif state == "in_ascent":
            current_rep_frames.append(frame)
            if phase == "standing":
                # Completed rep
                rep_number += 1
                reps.append(summarize_rep(rep_number, current_rep_frames, args))
                state = "at_standing"
                current_rep_frames = [frame]
            elif phase == "bottom":
                # Went back down, unusual but track it
                state = "in_bottom"

    # Check for incomplete final rep
    if state in ("in_bottom", "in_ascent") and len(current_rep_frames) >= args.min_descent_frames:
        bottom_frames = [f for f in current_rep_frames if classify_phase(f["knee_angle"], args) == "bottom"]
        if bottom_frames:
            rep_number += 1
            reps.append(summarize_rep(rep_number, current_rep_frames, args, incomplete=True))

    return reps


def summarize_rep(rep_number, frames, args, incomplete=False):
    """Create detailed rep summary from frames."""
    if not frames:
        return None

    # Find bottom frame (minimum knee angle)
    bottom_frame = min(frames, key=lambda f: f.get("knee_angle", 999))

    # Collect metrics
    knees = [f["knee_angle"] for f in frames if isinstance(f.get("knee_angle"), (int, float))]
    hips = [f["hip_angle"] for f in frames if isinstance(f.get("hip_angle"), (int, float))]
    torsos = [f["torso_angle"] for f in frames if isinstance(f.get("torso_angle"), (int, float))]
    shins = [f["shin_angle"] for f in frames if isinstance(f.get("shin_angle"), (int, float))]
    confidences = [f["confidence_avg"] for f in frames if isinstance(f.get("confidence_avg"), (int, float))]

    # Timing
    start_ts = frames[0].get("timestamp_ms", 0)
    end_ts = frames[-1].get("timestamp_ms", 0)

    # Count frame types
    provisional_count = sum(1 for f in frames if f.get("provisional"))
    gate_pass_count = sum(1 for f in frames if f.get("gate_pass"))

    # Setup issues during rep
    setup_issues = defaultdict(int)
    for f in frames:
        setup = f.get("setup") or {}
        if not setup.get("head_visible"):
            setup_issues["head_not_visible"] += 1
        if not setup.get("feet_visible"):
            setup_issues["feet_not_visible"] += 1
        if not setup.get("centered"):
            setup_issues["not_centered"] += 1
        if not setup.get("side_view"):
            setup_issues["not_side_view"] += 1

    # Depth assessment
    min_knee = min(knees) if knees else None
    hip_below_knee_frames = sum(1 for f in frames if f.get("hip_below_knee"))

    if min_knee is not None:
        if min_knee < 90:
            depth_quality = "deep"
        elif min_knee < 100:
            depth_quality = "parallel"
        elif min_knee < 110:
            depth_quality = "slightly_shallow"
        else:
            depth_quality = "shallow"
    else:
        depth_quality = "unknown"

    # Form assessment
    max_torso = max(torsos) if torsos else None
    if max_torso is not None:
        if max_torso > 50:
            lean_quality = "excessive_forward_lean"
        elif max_torso > 40:
            lean_quality = "moderate_forward_lean"
        else:
            lean_quality = "good"
    else:
        lean_quality = "unknown"

    return {
        "rep_number": rep_number,
        "source": "derived_offline",
        "incomplete": incomplete,
        "frame_count": len(frames),
        "provisional_count": provisional_count,
        "gate_pass_count": gate_pass_count,
        "bottom_frame": {
            "timestamp_ms": bottom_frame.get("timestamp_ms"),
            "knee_angle": round(bottom_frame.get("knee_angle"), 2) if bottom_frame.get("knee_angle") else None,
            "hip_angle": round(bottom_frame.get("hip_angle"), 2) if bottom_frame.get("hip_angle") else None,
            "torso_angle": round(bottom_frame.get("torso_angle"), 2) if bottom_frame.get("torso_angle") else None,
            "shin_angle": round(bottom_frame.get("shin_angle"), 2) if bottom_frame.get("shin_angle") else None,
            "hip_below_knee": bottom_frame.get("hip_below_knee"),
        },
        "tempo": {
            "total_ms": end_ts - start_ts,
        },
        "ranges": {
            "knee_angle": {"min": round(min(knees), 2), "max": round(max(knees), 2)} if knees else None,
            "hip_angle": {"min": round(min(hips), 2), "max": round(max(hips), 2)} if hips else None,
            "torso_angle": {"min": round(min(torsos), 2), "max": round(max(torsos), 2)} if torsos else None,
            "shin_angle": {"min": round(min(shins), 2), "max": round(max(shins), 2)} if shins else None,
        },
        "confidence": {
            "avg": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "min": round(min(confidences), 3) if confidences else None,
            "max": round(max(confidences), 3) if confidences else None,
        },
        "quality": {
            "depth": depth_quality,
            "lean": lean_quality,
            "hip_below_knee_frames": hip_below_knee_frames,
        },
        "setup_issues_during_rep": dict(setup_issues),
    }


def build_mediaprose(session, rep):
    """Generate MediaPipe-style prose description for coaching models."""
    bottom = rep.get("bottom_frame") or {}
    quality = rep.get("quality") or {}
    tempo = rep.get("tempo") or {}

    depth_desc = {
        "deep": "good depth with hip well below knee",
        "parallel": "parallel depth achieved",
        "slightly_shallow": "slightly shallow, hip near knee level",
        "shallow": "shallow squat, did not reach parallel",
        "unknown": "depth unclear from available data",
    }.get(quality.get("depth", "unknown"), "depth assessment unavailable")

    lean_desc = {
        "good": "good torso position",
        "moderate_forward_lean": "some forward lean present",
        "excessive_forward_lean": "excessive forward lean observed",
        "unknown": "torso angle unclear",
    }.get(quality.get("lean", "unknown"), "torso assessment unavailable")

    return (
        f"Offline-derived side-view squat rep {rep.get('rep_number')}. "
        f"Bottom position: knee angle {bottom.get('knee_angle')} degrees, "
        f"hip angle {bottom.get('hip_angle')} degrees, "
        f"torso angle {bottom.get('torso_angle')} degrees, "
        f"shin angle {bottom.get('shin_angle')} degrees. "
        f"Hip below knee: {bottom.get('hip_below_knee')}. "
        f"Rep duration {tempo.get('total_ms')} ms. "
        f"Assessment: {depth_desc}, {lean_desc}. "
        f"Source: derived from {rep.get('frame_count')} pose frames "
        f"({rep.get('provisional_count')} provisional, {rep.get('gate_pass_count')} gated). "
    )


def process_session(input_path, args):
    """Process a single session file."""
    payload = load_json(input_path)
    session_id = payload.get("session_id", input_path.stem)

    # Get all pose frames
    all_frames = get_all_pose_frames(payload)

    # Check what we have
    original_reps = payload.get("rep_summaries", [])
    live_rep_counts = set(f.get("rep_count") for f in all_frames if f.get("rep_count"))

    # Derive reps from raw angle trajectory
    derived_reps = detect_reps_from_angles(all_frames, args)

    # Add mediaprose to each rep
    for rep in derived_reps:
        rep["mediaprose"] = build_mediaprose(payload, rep)

    # Analyze why live rep counting may have failed
    total_frames = len(all_frames)
    gated_frames = sum(1 for f in all_frames if f.get("gate_pass"))
    rep_mode_frames = sum(1 for f in all_frames if f.get("rep_mode_active"))
    provisional_frames = sum(1 for f in all_frames if f.get("provisional"))

    # Setup issue analysis
    setup_blocking = defaultdict(int)
    for f in all_frames:
        setup = f.get("setup") or {}
        if not f.get("gate_pass"):
            if not setup.get("feet_visible"):
                setup_blocking["feet_not_visible"] += 1
            if not setup.get("head_visible"):
                setup_blocking["head_not_visible"] += 1
            if not setup.get("centered"):
                setup_blocking["not_centered"] += 1
            if not setup.get("side_view"):
                setup_blocking["not_side_view"] += 1

    result = {
        "session_id": session_id,
        "input_path": str(input_path),
        "camera_angle": payload.get("camera_angle"),
        "camera_height": payload.get("camera_height"),
        "frame_stats": {
            "total_pose_frames": total_frames,
            "gated_frames": gated_frames,
            "gated_pct": round(100 * gated_frames / total_frames, 1) if total_frames else 0,
            "rep_mode_frames": rep_mode_frames,
            "provisional_frames": provisional_frames,
        },
        "live_rep_detection": {
            "original_rep_count": len(original_reps),
            "live_rep_numbers_seen": sorted(live_rep_counts),
        },
        "offline_rep_detection": {
            "derived_rep_count": len(derived_reps),
            "reps": derived_reps,
        },
        "setup_blocking_analysis": {
            "total_non_gated": total_frames - gated_frames,
            "reasons": dict(setup_blocking),
            "primary_blocker": max(setup_blocking.items(), key=lambda x: x[1])[0] if setup_blocking else None,
        },
        "comparison": {
            "live_found_reps": len(original_reps) > 0,
            "offline_found_reps": len(derived_reps) > 0,
            "rep_count_match": len(original_reps) == len(derived_reps),
            "offline_found_more": len(derived_reps) > len(original_reps),
        },
    }

    return result


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle glob pattern vs single file
    if "*" in args.input:
        base = Path("/") if args.input.startswith("/") else Path(".")
        pattern = args.input.lstrip("/")
        input_paths = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        input_paths = [input_path]

    if not input_paths:
        raise SystemExit(f"No files matched: {args.input}")

    print(f"Processing {len(input_paths)} session files...")

    results = []
    for path in input_paths:
        print(f"\n{'='*60}")
        print(f"Session: {path.name}")
        result = process_session(path, args)
        results.append(result)

        # Print summary
        fs = result["frame_stats"]
        live = result["live_rep_detection"]
        offline = result["offline_rep_detection"]

        print(f"  Frames: {fs['total_pose_frames']} total, {fs['gated_frames']} gated ({fs['gated_pct']}%)")
        print(f"  Live reps: {live['original_rep_count']}")
        print(f"  Offline reps: {offline['derived_rep_count']}")

        if result["setup_blocking_analysis"]["primary_blocker"]:
            print(f"  Primary blocker: {result['setup_blocking_analysis']['primary_blocker']}")

        # Save individual result
        out_path = output_dir / f"{path.stem}_derived.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Saved: {out_path}")

    # Build aggregate report
    aggregate = {
        "total_sessions": len(results),
        "sessions_with_live_reps": sum(1 for r in results if r["live_rep_detection"]["original_rep_count"] > 0),
        "sessions_with_offline_reps": sum(1 for r in results if r["offline_rep_detection"]["derived_rep_count"] > 0),
        "sessions_where_offline_found_more": sum(1 for r in results if r["comparison"]["offline_found_more"]),
        "total_live_reps": sum(r["live_rep_detection"]["original_rep_count"] for r in results),
        "total_offline_reps": sum(r["offline_rep_detection"]["derived_rep_count"] for r in results),
        "common_blockers": defaultdict(int),
    }

    for r in results:
        blocker = r["setup_blocking_analysis"]["primary_blocker"]
        if blocker:
            aggregate["common_blockers"][blocker] += 1

    aggregate["common_blockers"] = dict(aggregate["common_blockers"])

    aggregate_path = output_dir / "aggregate_derived_reps.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))

    # Full batch report
    batch = {
        "args": {
            "knee_threshold_standing": args.knee_threshold_standing,
            "knee_threshold_bottom": args.knee_threshold_bottom,
            "min_descent_frames": args.min_descent_frames,
            "min_confidence": args.min_confidence,
        },
        "sessions": results,
        "aggregate": aggregate,
    }
    batch_path = output_dir / "batch_derived_reps.json"
    batch_path.write_text(json.dumps(batch, indent=2))

    print(f"\n{'='*60}")
    print(f"Aggregate report: {aggregate_path}")
    print(f"Batch report: {batch_path}")
    print(f"\nSummary:")
    print(f"  Sessions: {aggregate['total_sessions']}")
    print(f"  With live reps: {aggregate['sessions_with_live_reps']}")
    print(f"  With offline reps: {aggregate['sessions_with_offline_reps']}")
    print(f"  Offline found more: {aggregate['sessions_where_offline_found_more']}")
    print(f"  Total reps (live): {aggregate['total_live_reps']}")
    print(f"  Total reps (offline): {aggregate['total_offline_reps']}")
    if aggregate["common_blockers"]:
        print(f"\nCommon blockers:")
        for blocker, count in sorted(aggregate["common_blockers"].items(), key=lambda x: -x[1]):
            print(f"  {blocker}: {count} sessions")


if __name__ == "__main__":
    main()
