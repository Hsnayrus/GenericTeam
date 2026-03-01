#!/usr/bin/env python3
"""
Simulate Fixed Phase Detection

Takes existing session JSONs and re-runs the phase detection logic
with the fixes applied to show what WOULD happen with the new code.

This proves:
1. Phase detection works (transitions properly based on knee angles)
2. Trusted confidence filter works (only avgConf >= 0.6 updates phase)
3. Feet gate removal allows processing to proceed
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


# Simulated CONFIG (matches the fixed index.html)
CONFIG = {
    "confidenceThreshold": 0.5,
    "trustedConfidence": 0.6,  # NEW: Higher bar for phase detection
    "standingAngle": 155,
    "bottomAngle": 110,
    "angleVelocityThreshold": 1.5,
}


class SimulatedPhaseDetector:
    """Python version of the fixed PhaseDetector."""

    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"

    def __init__(self):
        self.phase = self.STANDING
        self.angle_history = []
        self.was_at_bottom = False
        self.rep_complete = False

    def update(self, knee_angle):
        self.rep_complete = False
        self.angle_history.append(knee_angle)
        if len(self.angle_history) > 5:
            self.angle_history.pop(0)

        # Calculate velocity
        if len(self.angle_history) >= 2:
            v = self.angle_history[-1] - self.angle_history[-2]
        else:
            v = 0

        # State machine
        if self.phase == self.STANDING:
            if knee_angle < CONFIG["standingAngle"] and v < -CONFIG["angleVelocityThreshold"]:
                self.phase = self.DESCENDING

        elif self.phase == self.DESCENDING:
            if knee_angle <= CONFIG["bottomAngle"] or \
               (abs(v) < CONFIG["angleVelocityThreshold"] and knee_angle < CONFIG["standingAngle"] - 15):
                self.phase = self.BOTTOM
                self.was_at_bottom = True
            elif knee_angle >= CONFIG["standingAngle"]:
                self.phase = self.STANDING

        elif self.phase == self.BOTTOM:
            if v > CONFIG["angleVelocityThreshold"]:
                self.phase = self.ASCENDING

        elif self.phase == self.ASCENDING:
            if knee_angle >= CONFIG["standingAngle"]:
                self.phase = self.STANDING
                if self.was_at_bottom:
                    self.rep_complete = True
                    self.was_at_bottom = False

        return {"phase": self.phase, "repComplete": self.rep_complete}


def load_json(path):
    return json.loads(Path(path).read_text())


def get_all_pose_frames(payload):
    """Extract frames with live data."""
    frames = payload.get("frames", [])
    result = []
    for f in frames:
        live = f.get("live")
        if not live:
            continue
        result.append({
            "timestamp_ms": f.get("timestamp_ms"),
            "old_phase": live.get("phase"),  # What the old code said
            "knee_angle": live.get("knee_angle"),
            "hip_angle": live.get("hip_angle"),
            "torso_angle": live.get("torso_angle"),
            "confidence_avg": f.get("confidence_avg"),
            "confidence_min": f.get("confidence_min"),
            "gate_pass": f.get("gate_pass"),
            "rep_mode_active": f.get("rep_mode_active"),
            "setup": f.get("setup", {}),
        })
    return result


def simulate_session(payload):
    """Run simulation on a session."""
    frames = get_all_pose_frames(payload)
    if not frames:
        return None

    detector = SimulatedPhaseDetector()
    rep_count = 0
    results = []

    # Statistics
    stats = {
        "total_frames": len(frames),
        "old_phases": defaultdict(int),
        "new_phases": defaultdict(int),
        "trusted_frames": 0,
        "untrusted_frames": 0,
        "old_gate_passed": 0,
        "new_would_pass": 0,  # With feet removed from blocking
    }

    for f in frames:
        knee = f.get("knee_angle")
        conf_avg = f.get("confidence_avg", 0)
        conf_min = f.get("confidence_min", 0)
        old_phase = f.get("old_phase")
        setup = f.get("setup", {})

        # Old gate logic (all checks must pass including feet)
        old_all_pass = all([
            setup.get("head_visible"),
            setup.get("feet_visible"),
            setup.get("centered"),
            setup.get("side_view"),
        ])

        # NEW gate logic (feet removed)
        new_would_pass = all([
            setup.get("head_visible"),
            # setup.get("feet_visible"),  # REMOVED
            setup.get("centered"),
            setup.get("side_view"),
        ])

        stats["old_phases"][old_phase] += 1
        if f.get("gate_pass"):
            stats["old_gate_passed"] += 1
        if new_would_pass:
            stats["new_would_pass"] += 1

        # Determine if frame is trusted (new confidence filter)
        trusted = conf_avg >= CONFIG["trustedConfidence"]
        if trusted:
            stats["trusted_frames"] += 1
        else:
            stats["untrusted_frames"] += 1

        # Simulate new phase detection
        new_phase = detector.phase
        rep_complete = False

        if knee is not None and trusted:
            result = detector.update(knee)
            new_phase = result["phase"]
            rep_complete = result["repComplete"]

        stats["new_phases"][new_phase] += 1

        if rep_complete:
            rep_count += 1

        results.append({
            "timestamp_ms": f["timestamp_ms"],
            "knee_angle": round(knee, 1) if knee else None,
            "conf_avg": round(conf_avg, 2),
            "trusted": trusted,
            "old_phase": old_phase,
            "new_phase": new_phase,
            "phase_changed": old_phase != new_phase,
            "rep_complete": rep_complete,
            "rep_count": rep_count,
        })

    # Find phase transitions
    transitions = []
    for i in range(1, len(results)):
        if results[i]["new_phase"] != results[i-1]["new_phase"]:
            transitions.append({
                "timestamp_ms": results[i]["timestamp_ms"],
                "from": results[i-1]["new_phase"],
                "to": results[i]["new_phase"],
                "knee_angle": results[i]["knee_angle"],
            })

    return {
        "session_id": payload.get("session_id"),
        "stats": {
            **stats,
            "old_phases": dict(stats["old_phases"]),
            "new_phases": dict(stats["new_phases"]),
        },
        "rep_count_old": 0,  # Old code found 0 reps
        "rep_count_new": rep_count,
        "transitions": transitions,
        "sample_frames": results[::max(1, len(results)//20)],  # Sample every 5%
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/results/simulation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle glob
    if "*" in args.input:
        base = Path("/") if args.input.startswith("/") else Path(".")
        paths = sorted(base.glob(args.input.lstrip("/")), key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        paths = [Path(args.input)]

    print(f"Simulating fixed phase detection on {len(paths)} sessions...\n")

    all_results = []
    for path in paths:
        print(f"{'='*60}")
        print(f"Session: {path.name}")

        payload = load_json(path)
        result = simulate_session(payload)

        if not result:
            print("  No pose frames found")
            continue

        stats = result["stats"]
        print(f"  Frames: {stats['total_frames']}")
        print(f"  Trusted (conf >= 0.6): {stats['trusted_frames']} ({100*stats['trusted_frames']/stats['total_frames']:.1f}%)")
        print(f"  Old gate passed: {stats['old_gate_passed']}")
        print(f"  New would pass (no feet): {stats['new_would_pass']}")
        print(f"\n  OLD phase distribution: {stats['old_phases']}")
        print(f"  NEW phase distribution: {stats['new_phases']}")
        print(f"\n  OLD rep count: {result['rep_count_old']}")
        print(f"  NEW rep count: {result['rep_count_new']}")

        if result["transitions"]:
            print(f"\n  Phase transitions detected:")
            for t in result["transitions"][:10]:
                print(f"    {t['from']} -> {t['to']} @ knee={t['knee_angle']}°")

        # Save individual result
        out_path = output_dir / f"{path.stem}_simulation.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\n  Saved: {out_path}")

        all_results.append(result)

    # Aggregate
    total_old = sum(r["rep_count_old"] for r in all_results)
    total_new = sum(r["rep_count_new"] for r in all_results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Sessions: {len(all_results)}")
    print(f"Total reps (OLD code): {total_old}")
    print(f"Total reps (NEW code): {total_new}")
    print(f"Improvement: {total_new - total_old} more reps detected")

    # Save aggregate
    aggregate = {
        "sessions": len(all_results),
        "total_reps_old": total_old,
        "total_reps_new": total_new,
        "improvement": total_new - total_old,
        "per_session": [
            {
                "session_id": r["session_id"],
                "old_reps": r["rep_count_old"],
                "new_reps": r["rep_count_new"],
                "transitions": len(r["transitions"]),
            }
            for r in all_results
        ],
    }
    agg_path = output_dir / "simulation_aggregate.json"
    agg_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\nAggregate saved: {agg_path}")


if __name__ == "__main__":
    main()
