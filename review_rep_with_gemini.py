#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


SYSTEM_INSTRUCTION = """You are an expert squat-form reviewer acting as a teacher model for a local squat coach.
You will review side-view squat evidence and produce strict JSON only.

Rules:
- Prioritize one coaching issue only.
- Allowed priorities: framing, safety, depth, knees, encouragement.
- Ignore tempo for now unless it clearly creates a safety problem.
- Use the visual evidence first, then the structured pose summary as supporting context.
- Keep "say" to one short sentence, maximum 18 words.
- Keep "reason" concise and biomechanical, not verbose.
- If the rep looks acceptable, return encouragement.

Output schema:
{
  "say": string,
  "priority": "framing|safety|depth|knees|encouragement",
  "reason": string,
  "confidence": number
}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Review one squat rep with Gemini using extracted frames or a source video.")
    parser.add_argument("--session-json", required=True, help="Cleaned session JSON with rep summaries.")
    parser.add_argument("--rep", type=int, required=True, help="Rep number to review.")
    parser.add_argument("--video", default="", help="Optional source video. If set, frames are extracted automatically.")
    parser.add_argument("--frames", nargs="*", default=[], help="Optional pre-extracted frame paths.")
    parser.add_argument("--output", default="data/results/gemini_rep_review.json")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing dependency: install `google-genai` first with `python3 -m pip install -U google-genai`.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def load_json(path):
    return json.loads(Path(path).read_text())


def choose_rep(payload, rep_number):
    for rep in payload.get("rep_summaries", []):
        if rep.get("rep_number") == rep_number:
            return rep
    raise SystemExit(f"Rep {rep_number} not found in {payload.get('session_id', 'session')}.")


def find_rep_timestamps(frames, rep_number):
    matching = [frame["timestamp_ms"] for frame in frames if (frame.get("live") or {}).get("rep_count") == rep_number]
    if not matching:
        raise SystemExit(f"No frame timestamps found for rep {rep_number}.")
    return [matching[0], matching[len(matching) // 2], matching[-1]]


def extract_frames(video_path, timestamps_ms, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for index, timestamp_ms in enumerate(timestamps_ms, start=1):
        output_path = output_dir / f"rep-{index:02d}-{timestamp_ms}ms.jpg"
        seconds = f"{timestamp_ms / 1000:.3f}"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", seconds, "-i", str(video_path), "-frames:v", "1", str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        out_paths.append(output_path)
    return out_paths


def upload_files(client, paths):
    uploaded = []
    for path in paths:
        uploaded.append(client.files.upload(file=path))
    return uploaded


def build_prompt(session_payload, rep):
    payload = {
        "session_id": session_payload.get("session_id"),
        "camera_angle": session_payload.get("camera_angle"),
        "camera_height": session_payload.get("camera_height"),
        "rep_number": rep.get("rep_number"),
        "bottom_frame": rep.get("bottom_frame"),
        "tempo": rep.get("tempo"),
        "locked_thresholds": {
            "depth_target_knee_deg": 100,
            "torso_warn_deg": 45,
            "ignore_tempo": True,
        },
    }
    return (
        "Review this side-view squat rep. "
        "The images are early, mid, and late rep frames from the same rep.\n\n"
        f"Structured context:\n{json.dumps(payload, indent=2)}"
    )


def main():
    args = parse_args()
    require_sdk()
    payload = load_json(args.session_json)
    rep = choose_rep(payload, args.rep)

    frame_paths = [Path(frame) for frame in args.frames]
    if not frame_paths:
        if not args.video:
            raise SystemExit("Provide either --frames or --video.")
        timestamps = find_rep_timestamps(payload.get("frames", []), args.rep)
        frame_dir = Path("data/extracted_frames") / f"rep-{args.rep:02d}"
        frame_paths = extract_frames(Path(args.video), timestamps, frame_dir)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    uploaded_files = upload_files(client, frame_paths)
    contents = [*uploaded_files, build_prompt(payload, rep)]
    response = client.models.generate_content(
        model=args.model,
        contents=contents,
        config={
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )
    text = getattr(response, "text", None)
    if not text:
        raise SystemExit("Gemini returned an empty response.")
    parsed = json.loads(text)
    result = {
        "model": args.model,
        "session_json": args.session_json,
        "rep_number": args.rep,
        "frame_paths": [str(path) for path in frame_paths],
        "review": parsed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote Gemini rep review to {output}")
    print(json.dumps(result, indent=2))

    if not args.keep_frames and args.video and not args.frames:
        for frame_path in frame_paths:
            try:
                frame_path.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
