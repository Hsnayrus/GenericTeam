#!/usr/bin/env python3
import argparse
import base64
import json
import subprocess
from pathlib import Path
from urllib import error, request

from prompt_templates import COACH_RESPONSE_SCHEMA, build_gemma_messages, sanitize_coach_output


SYSTEM_VISUAL = """You are a squat coach looking at side-view squat images.
Return strict JSON only with this schema:
{
  "say": string,
  "priority": "safety|depth|knees|encouragement",
  "reason": string
}
Infer the most important issue visible in the images."""


def extract_frames(video_path, timestamps_ms, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for index, timestamp_ms in enumerate(timestamps_ms, start=1):
        output_path = output_dir / f"frame-{index:02d}-{timestamp_ms}ms.jpg"
        seconds = f"{timestamp_ms / 1000:.3f}"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                seconds,
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        out_paths.append(output_path)
    return out_paths


def ollama_chat(model, messages, images=None, base_url="http://127.0.0.1:11434", timeout=600):
    message = {"role": "user", "content": messages[-1]["content"]}
    if images:
        message["images"] = images
    body_payload = {
        "model": model,
        "messages": [message],
        "stream": False,
        "options": {"temperature": 0},
    }
    if not images:
        body_payload["format"] = COACH_RESPONSE_SCHEMA
    body = json.dumps(body_payload).encode()
    req = request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            parsed = json.loads(response.read().decode())
    except error.HTTPError as exc:
        detail = exc.read().decode()
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    return parsed["message"]["content"]


def parse_json_response(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"raw_text": raw_text}


def rules_decision(rep):
    if rep["torso_angle"] >= 40:
        return {"say": "Keep your chest up through the rep.", "priority": "safety", "reason": "torso angle is high"}
    if not rep["hip_below_knee"] or rep["knee_angle"] > 110:
        return {"say": "Sit a little deeper on the next rep.", "priority": "depth", "reason": "bottom depth is shallow"}
    return {"say": "Good rep. Keep that shape.", "priority": "encouragement", "reason": "no major issue detected"}


def choose_rep(payload, rep_number):
    for rep in payload.get("rep_summaries", []):
        if rep["rep_number"] == rep_number:
            return rep
    raise ValueError(f"rep {rep_number} not found")


def find_rep_timestamps(frames, rep_number):
    matching = [frame["timestamp_ms"] for frame in frames if (frame.get("live") or {}).get("rep_count") == rep_number]
    if not matching:
        return []
    return [
        matching[0],
        matching[len(matching) // 2],
        matching[-1],
    ]


def main():
    parser = argparse.ArgumentParser(description="Compare rules, pose-prompted Gemma, and frame-prompted Gemma on one squat session.")
    parser.add_argument("--session-json", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=["gemma3:1b"])
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", default="data/session_model_compare.json")
    parser.add_argument("--mode", choices=["both", "pose", "visual"], default="both")
    args = parser.parse_args()

    payload = json.loads(Path(args.session_json).read_text())
    rep = choose_rep(payload, args.rep)
    timestamps = find_rep_timestamps(payload["frames"], args.rep)
    if not timestamps:
        raise SystemExit(f"No frame timestamps found for rep {args.rep}.")
    frame_dir = Path("data/extracted_frames") / f"rep-{args.rep:02d}"
    frame_paths = extract_frames(Path(args.video), timestamps, frame_dir)
    encoded_images = [base64.b64encode(path.read_bytes()).decode() for path in frame_paths]

    pose_input = {
        "rep_number": args.rep,
        "bottom_frame": rep["bottom_frame"],
        "tempo": rep["tempo"],
    }
    rules = rules_decision({**rep["bottom_frame"], "tempo": rep["tempo"]})
    results = {
        "rep_number": args.rep,
        "pose_input": pose_input,
        "frame_paths": [str(path) for path in frame_paths],
        "rules": rules,
        "models": {},
    }

    for model in args.models:
        pose_parsed = None
        visual_parsed = None
        if args.mode in {"both", "pose"}:
            pose_raw = ollama_chat(
                model,
                build_gemma_messages(pose_input),
                base_url=args.base_url,
            )
            pose_parsed = sanitize_coach_output(pose_input, parse_json_response(pose_raw))
        if args.mode in {"both", "visual"}:
            visual_prompt = (
                "These three frames show the same squat rep from side view: early rep, mid rep, late rep. "
                "Evaluate the most important coaching issue visible."
            )
            try:
                visual_raw = ollama_chat(
                    model,
                    [
                        {"role": "system", "content": SYSTEM_VISUAL},
                        {"role": "user", "content": visual_prompt},
                    ],
                    images=encoded_images,
                    base_url=args.base_url,
                )
                visual_parsed = parse_json_response(visual_raw)
            except Exception as exc:
                visual_parsed = {"error": str(exc)}
        results["models"][model] = {
            "pose_prompted": pose_parsed,
            "visual_prompted": visual_parsed,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote comparison to {output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
