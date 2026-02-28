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


REAL_VISUAL_SYSTEM = """You are an expert squat-form reviewer acting as a teacher model for a local squat coach.
You will review side-view squat evidence and produce strict JSON only.

Rules:
- Prioritize one coaching issue only.
- Allowed priorities: framing, safety, depth, knees, encouragement.
- Ignore tempo unless it clearly creates a safety problem.
- Use the visual evidence first, then the structured pose summary as supporting context.
- Keep "say" to one short sentence, maximum 18 words.
- Keep "reason" concise and biomechanical.
- If the rep looks acceptable, return encouragement.

Output schema:
{
  "say": string,
  "priority": "framing|safety|depth|knees|encouragement",
  "reason": string,
  "confidence": number
}"""

REAL_POSE_SYSTEM = """You are an expert squat-form reviewer for a local squat coach.
You are given only structured squat metrics and a MediaPipe-style prose summary.
Return strict JSON only with this schema:
{
  "say": string,
  "priority": "framing|safety|depth|knees|encouragement",
  "reason": string,
  "confidence": number
}
Rules:
- Prioritize one coaching issue only.
- Keep "say" to one short sentence, maximum 18 words.
- Use the structured metrics and prose only.
- Translate internal angles into plain coaching language.
- If the rep is acceptable, return encouragement.
"""

SYNTHETIC_POSE_SYSTEM = """You are an expert squat-form reviewer for a local squat coach.
You are given MediaPipe-style prose and structured squat metadata only.
You must not ask for the original images or video and you must not mention missing pixels or missing footage.

Return strict JSON only with this schema:
{
  "say": string,
  "priority": "framing|safety|depth|knees|encouragement",
  "reason": string,
  "confidence": number
}
Rules:
- Prioritize one coaching issue only.
- Keep "say" to one short sentence, maximum 18 words.
- Use the prose and structured metrics only.
- Translate internal angles into plain coaching language.
- If the rep is acceptable, return encouragement.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Bulk-review a real session and synthetic pose data with Gemini.")
    parser.add_argument("--session-json", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--synthetic-manifest", default="")
    parser.add_argument("--output", default="data/results/gemini_bulk_review.json")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"))
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing dependency: install `google-genai` first with `python3 -m pip install -r requirements.txt`.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def load_json(path):
    return json.loads(Path(path).read_text())


def extract_frames(video_path, timestamps_ms, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for index, timestamp_ms in enumerate(timestamps_ms, start=1):
        output_path = output_dir / f"frame-{index:02d}-{timestamp_ms}ms.jpg"
        seconds = f"{timestamp_ms / 1000:.3f}"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", seconds, "-i", str(video_path), "-frames:v", "1", str(output_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        out_paths.append(output_path)
    return out_paths


def find_rep_frames(payload, rep_number):
    frames = [frame for frame in payload.get("frames", []) if (frame.get("live") or {}).get("rep_count") == rep_number]
    if not frames:
        return []
    return [frames[0], frames[len(frames) // 2], frames[-1]]


def build_real_pose_summary(payload, rep, sampled_frames):
    first_live = sampled_frames[0].get("live") or {}
    mid_live = sampled_frames[1].get("live") or {}
    last_live = sampled_frames[2].get("live") or {}
    bottom = rep.get("bottom_frame") or {}
    tempo = rep.get("tempo") or {}
    mediaprose = (
        "Real side-view squat rep from a cleaned session export. "
        f"Rep {rep['rep_number']} lasts about {tempo.get('total_ms')} ms. "
        f"Bottom frame metrics: knee angle {bottom.get('knee_angle')} deg, hip angle {bottom.get('hip_angle')} deg, "
        f"torso angle {bottom.get('torso_angle')} deg, shin angle {bottom.get('shin_angle')} deg. "
        f"Across sampled live frames, the rep begins at knee angle {first_live.get('knee_angle')} deg in phase {first_live.get('phase')}, "
        f"reaches mid-rep around knee angle {mid_live.get('knee_angle')} deg in phase {mid_live.get('phase')}, "
        f"and finishes near knee angle {last_live.get('knee_angle')} deg in phase {last_live.get('phase')}. "
        "Use these metrics as internal evidence but explain the coaching cue in plain language."
    )
    return {
        "session_id": payload.get("session_id"),
        "camera_angle": payload.get("camera_angle"),
        "camera_height": payload.get("camera_height"),
        "rep_number": rep.get("rep_number"),
        "bottom_frame": bottom,
        "tempo": tempo,
        "frame_samples": {
            "first": first_live,
            "middle": mid_live,
            "last": last_live,
        },
        "mediaprose": mediaprose,
    }


def upload_files(client, paths):
    return [client.files.upload(file=path) for path in paths]


def generate_json(client, model, system_instruction, contents):
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config={
            "system_instruction": system_instruction,
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return json.loads(text)


def review_real_session(client, model, payload, video_path, extracted_root):
    reviews = []
    for rep in payload.get("rep_summaries", []):
        sampled_frames = find_rep_frames(payload, rep["rep_number"])
        if len(sampled_frames) != 3:
            continue
        timestamps = [frame["timestamp_ms"] for frame in sampled_frames]
        frame_dir = extracted_root / f"rep-{rep['rep_number']:02d}"
        frame_paths = extract_frames(video_path, timestamps, frame_dir)
        pose_summary = build_real_pose_summary(payload, rep, sampled_frames)
        uploaded_files = upload_files(client, frame_paths)
        visual_prompt = (
            "Review this side-view squat rep. "
            "The images are early, mid, and late rep frames from the same real rep.\n\n"
            f"Structured context:\n{json.dumps(pose_summary, indent=2)}"
        )
        visual_review = generate_json(client, model, REAL_VISUAL_SYSTEM, [*uploaded_files, visual_prompt])
        pose_review = generate_json(
            client,
            model,
            REAL_POSE_SYSTEM,
            "Review this real squat rep using only the structured pose summary and prose.\n\n"
            + json.dumps(pose_summary, indent=2),
        )
        reviews.append(
            {
                "rep_number": rep["rep_number"],
                "frame_paths": [str(path) for path in frame_paths],
                "pose_input": pose_summary,
                "visual_review": visual_review,
                "pose_review": pose_review,
            }
        )
    return reviews


def build_synthetic_prompt(example):
    compact = {
        "id": example.get("id"),
        "label": example.get("label"),
        "issue": example.get("issue"),
        "profile": example.get("profile"),
        "mediaprose": example.get("mediaprose"),
        "frames": [
            {
                "phase": frame.get("phase"),
                "spec": frame.get("spec"),
            }
            for frame in (example.get("frames") or [])
        ]
        or None,
    }
    return (
        "Review this synthetic side-view squat example using only the provided pose prose and structured metadata.\n\n"
        + json.dumps(compact, indent=2)
    )


def review_synthetic_manifest(client, model, manifest_rows):
    synthetic_reviews = []
    for example in manifest_rows:
        review = generate_json(client, model, SYNTHETIC_POSE_SYSTEM, build_synthetic_prompt(example))
        synthetic_reviews.append(
            {
                "id": example.get("id"),
                "label": example.get("label"),
                "issue": example.get("issue"),
                "mediaprose": example.get("mediaprose"),
                "frames": example.get("frames"),
                "review": review,
            }
        )
    return synthetic_reviews


def main():
    args = parse_args()
    require_sdk()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    session_payload = load_json(args.session_json)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model": args.model,
        "real_session": {
            "session_json": args.session_json,
            "video": args.video,
            "session_id": session_payload.get("session_id"),
            "rep_count": len(session_payload.get("rep_summaries", [])),
            "reviews": review_real_session(
                client,
                args.model,
                session_payload,
                Path(args.video),
                output_path.parent / "bulk_extracted_frames",
            ),
        },
        "synthetic_pose_data": [],
    }

    if args.synthetic_manifest:
        result["synthetic_pose_data"] = review_synthetic_manifest(client, args.model, load_json(args.synthetic_manifest))

    output_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote bulk Gemini review to {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
