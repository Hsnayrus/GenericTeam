#!/usr/bin/env python3
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from urllib import error, request

from clean_offline_session import clean_session, load_json
from prompt_templates import COACH_RESPONSE_SCHEMA, build_gemma_messages, sanitize_coach_output
from summarize_session_baseline import summarize

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


GEMINI_SYSTEM = """You are an expert squat-form reviewer for a local squat coach.
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


def parse_args():
    parser = argparse.ArgumentParser(description="Batch-review exported squat sessions with Gemini and local Gemma using pose data only.")
    parser.add_argument("--input-glob", default="/Users/jwalinshah/Downloads/side-floor-session-*.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-dir", default="data/results/batch_pose_review")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--gemini-models", nargs="+", default=["models/gemini-2.5-pro", "models/gemini-3.1-pro-preview"])
    parser.add_argument("--gemma-models", nargs="+", default=["gemma3:1b", "gemma3:4b"])
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing dependency: install requirements into the local venv first.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def recent_inputs(pattern, limit):
    files = sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern.lstrip("/")))
    files = [path for path in files if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


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


def derive_rep_summaries(payload):
    grouped = defaultdict(list)
    for frame in payload.get("frames", []):
        live = frame.get("live")
        rep_count = (live or {}).get("rep_count")
        if live and isinstance(rep_count, int) and rep_count > 0:
            grouped[rep_count].append(frame)

    derived = []
    for rep_number in sorted(grouped):
        frames = grouped[rep_number]
        live_frames = [frame["live"] for frame in frames if isinstance(frame.get("live"), dict)]
        if not live_frames:
            continue
        valid_live_frames = [
            live for live in live_frames
            if isinstance(live.get("knee_angle"), (int, float))
            and isinstance(live.get("hip_angle"), (int, float))
            and isinstance(live.get("torso_angle"), (int, float))
            and isinstance(live.get("shin_angle"), (int, float))
            and isinstance(live.get("hip_below_knee"), bool)
        ]
        if not valid_live_frames:
            continue
        bottom_source = min(
            frames,
            key=lambda frame: (
                0 if (frame["live"].get("phase") == "bottom") else 1,
                frame["live"].get("knee_angle", 9999),
            ),
        )
        bottom_live = bottom_source["live"]
        start_ms = frames[0]["timestamp_ms"]
        end_ms = frames[-1]["timestamp_ms"]
        knees = [live["knee_angle"] for live in valid_live_frames]
        tempos = [live.get("tempo_ms", 0) for live in valid_live_frames if isinstance(live.get("tempo_ms"), (int, float))]
        derived.append(
            {
                "rep_number": rep_number,
                "source": "derived_from_live_frames",
                "frame_count": len(frames),
                "bottom_frame": {
                    "timestamp_ms": bottom_source["timestamp_ms"],
                    "knee_angle": bottom_live.get("knee_angle"),
                    "hip_angle": bottom_live.get("hip_angle"),
                    "torso_angle": bottom_live.get("torso_angle"),
                    "shin_angle": bottom_live.get("shin_angle"),
                    "hip_below_knee": bottom_live.get("hip_below_knee"),
                },
                "tempo": {
                    "total_ms": end_ms - start_ms,
                    "p10_tempo_ms": round(percentile(tempos, 0.1), 2) if tempos else None,
                    "p50_tempo_ms": round(percentile(tempos, 0.5), 2) if tempos else None,
                    "p90_tempo_ms": round(percentile(tempos, 0.9), 2) if tempos else None,
                },
                "range": {
                    "knee_angle_min": round(min(knees), 2),
                    "knee_angle_max": round(max(knees), 2),
                },
            }
        )
    return derived


def pose_summary(session_payload, rep):
    bottom = rep.get("bottom_frame") or {}
    tempo = rep.get("tempo") or {}
    prose = (
        f"Side-view squat rep {rep.get('rep_number')}. "
        f"Bottom knee angle {bottom.get('knee_angle')} degrees, hip angle {bottom.get('hip_angle')} degrees, "
        f"torso angle {bottom.get('torso_angle')} degrees, shin angle {bottom.get('shin_angle')} degrees. "
        f"Hip below knee: {bottom.get('hip_below_knee')}. "
        f"Rep duration about {tempo.get('total_ms')} ms. "
        "Use the metrics as evidence and return one plain-language coaching cue."
    )
    return {
        "session_id": session_payload.get("session_id"),
        "camera_angle": session_payload.get("camera_angle"),
        "camera_height": session_payload.get("camera_height"),
        "schema_version": session_payload.get("schema_version"),
        "rep_number": rep.get("rep_number"),
        "bottom_frame": bottom,
        "tempo": tempo,
        "mediaprose": prose,
    }


def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


def call_gemini(client, model, summary_payload):
    prompt = "Review this real squat rep using only the structured pose summary and prose.\n\n" + json.dumps(summary_payload, indent=2)
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "system_instruction": GEMINI_SYSTEM,
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError(f"{model} returned an empty response.")
    return parse_json_response(text), elapsed_ms


def ollama_chat(model, summary_payload, base_url):
    body = json.dumps(
        {
            "model": model,
            "messages": build_gemma_messages(summary_payload),
            "stream": False,
            "format": COACH_RESPONSE_SCHEMA,
            "options": {"temperature": 0},
        }
    ).encode()
    req = request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=600) as response:
            parsed = json.loads(response.read().decode())
    except error.HTTPError as exc:
        detail = exc.read().decode()
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return sanitize_coach_output(summary_payload, parse_json_response(parsed["message"]["content"])), elapsed_ms


def review_session(client, session_payload, gemini_models, gemma_models, ollama_base_url):
    reps = session_payload.get("rep_summaries") or derive_rep_summaries(session_payload)
    reviews = []
    for rep in reps:
        summary_payload = pose_summary(session_payload, rep)
        row = {
            "rep_number": rep.get("rep_number"),
            "rep_source": rep.get("source", "exported"),
            "pose_input": summary_payload,
            "gemini": {},
            "gemma": {},
        }
        for model in gemini_models:
            parsed, latency_ms = call_gemini(client, model, summary_payload)
            row["gemini"][model] = {"review": parsed, "latency_ms": latency_ms}
        for model in gemma_models:
            parsed, latency_ms = ollama_chat(model, summary_payload, ollama_base_url)
            row["gemma"][model] = {"review": parsed, "latency_ms": latency_ms}
        reviews.append(row)
    return reviews


def build_cross_reference(rows):
    cross = []
    for row in rows:
        entry = {
            "rep_number": row["rep_number"],
            "rep_source": row["rep_source"],
            "gemini_priorities": {model: result["review"].get("priority") for model, result in row["gemini"].items()},
            "gemma_priorities": {model: result["review"].get("priority") for model, result in row["gemma"].items()},
            "latency_ms": {
                "gemini": {model: result["latency_ms"] for model, result in row["gemini"].items()},
                "gemma": {model: result["latency_ms"] for model, result in row["gemma"].items()},
            },
        }
        entry["all_models_agree"] = len(
            {
                priority
                for priority in [*entry["gemini_priorities"].values(), *entry["gemma_priorities"].values()]
                if priority is not None
            }
        ) == 1
        cross.append(entry)
    return cross


def main():
    args = parse_args()
    require_sdk()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    input_paths = recent_inputs(args.input_glob, args.limit)
    if not input_paths:
        raise SystemExit(f"No input files matched {args.input_glob}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_glob": args.input_glob,
        "inputs": [str(path) for path in input_paths],
        "gemini_models": args.gemini_models,
        "gemma_models": args.gemma_models,
        "sessions": [],
    }

    for input_path in input_paths:
        raw_payload = load_json(input_path)
        derived_raw_reps = derive_rep_summaries(raw_payload)
        cleaned_payload = clean_session(raw_payload, args.min_confidence)
        if not cleaned_payload.get("rep_summaries"):
            cleaned_payload["rep_summaries"] = derived_raw_reps
        baseline = summarize(cleaned_payload, 100.0, 45.0)
        reviews = review_session(client, cleaned_payload, args.gemini_models, args.gemma_models, args.ollama_base_url)
        session_dir = output_dir / input_path.stem
        session_dir.mkdir(parents=True, exist_ok=True)
        cleaned_path = session_dir / "cleaned.json"
        cleaned_path.write_text(json.dumps(cleaned_payload, indent=2))
        summary_path = session_dir / "summary.json"
        summary_path.write_text(json.dumps(baseline, indent=2))
        review_path = session_dir / "pose_model_reviews.json"
        cross_reference = build_cross_reference(reviews)
        review_path.write_text(
            json.dumps(
                {
                    "session_id": cleaned_payload.get("session_id"),
                    "reviews": reviews,
                    "cross_reference": cross_reference,
                },
                indent=2,
            )
        )
        batch_result["sessions"].append(
            {
                "input": str(input_path),
                "session_id": cleaned_payload.get("session_id"),
                "cleaned_path": str(cleaned_path),
                "summary_path": str(summary_path),
                "review_path": str(review_path),
                "rep_count": len(cleaned_payload.get("rep_summaries", [])),
                "baseline_readiness": baseline.get("baseline_readiness"),
                "cross_reference": cross_reference,
            }
        )

    output_path = output_dir / "batch_report.json"
    output_path.write_text(json.dumps(batch_result, indent=2))
    print(f"Wrote batch report to {output_path}")
    print(json.dumps(batch_result, indent=2))


if __name__ == "__main__":
    main()
