#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path


SYSTEM_POSE = """You are a squat coach. You receive a compact side-view squat state.
Return strict JSON only with this schema:
{
  "say": string,
  "priority": "safety|depth|knees|encouragement",
  "reason": string
}
Keep say to one short sentence. Focus on the highest-priority issue only."""


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


def mlx_chat(mlx_model, mlx_tokenizer, messages):
    """Run MLX inference for a list of chat messages and return the raw output string.

    Parameters
    ----------
    mlx_model : object
        Loaded MLX model from ``mlx_lm.load()``.
    mlx_tokenizer : object
        Matching tokenizer from ``mlx_lm.load()``.
    messages : list of dict
        Chat messages in ``[{"role": ..., "content": ...}]`` format.

    Returns
    -------
    str
        Raw text output from the model.
    """
    from mlx_lm import generate
    prompt = mlx_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(mlx_model, mlx_tokenizer, prompt=prompt, max_tokens=150, verbose=False)


def parse_json_response(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
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
    parser = argparse.ArgumentParser(description="Compare rules and pose-prompted MLX model on one squat session.")
    parser.add_argument("--session-json", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--models", nargs="+", default=["mlx-community/gemma-3-1b-it-4bit"])
    parser.add_argument("--output", default="data/session_model_compare.json")
    parser.add_argument("--mode", choices=["pose", "visual"], default="pose")
    args = parser.parse_args()

    if args.mode == "visual":
        print("WARNING: --mode visual requires a multimodal model and is not supported with MLX text-only models. Use --mode pose.")
        raise SystemExit(1)

    from mlx_lm import load

    payload = json.loads(Path(args.session_json).read_text())
    rep = choose_rep(payload, args.rep)
    timestamps = find_rep_timestamps(payload["frames"], args.rep)
    if not timestamps:
        raise SystemExit(f"No frame timestamps found for rep {args.rep}.")
    frame_dir = Path("data/extracted_frames") / f"rep-{args.rep:02d}"
    frame_paths = extract_frames(Path(args.video), timestamps, frame_dir)

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

    for model_id in args.models:
        mlx_model, mlx_tokenizer = load(model_id)
        pose_raw = mlx_chat(
            mlx_model,
            mlx_tokenizer,
            [
                {"role": "system", "content": SYSTEM_POSE},
                {"role": "user", "content": json.dumps(pose_input)},
            ],
        )
        results["models"][model_id] = {
            "pose_prompted": parse_json_response(pose_raw),
            "visual_prompted": None,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"Wrote comparison to {output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
