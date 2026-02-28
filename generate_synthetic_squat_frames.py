#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


DEFAULT_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
NEGATIVE_PROMPT = (
    "front view, three quarter view, mirrored camera, cropped feet, cropped head, extra limbs, "
    "multiple people, gym text overlays, subtitles, collage, anime, cartoon, blurry anatomy, "
    "camera tilt, fisheye distortion, motion blur"
)

FRAME_LIBRARY = [
    {
        "label": "good-depth",
        "issue": "encouragement",
        "phases": [
            {"phase": "standing", "knee_deg": 176, "hip_deg": 171, "torso_lean_deg": 8, "coach_hint": "upright start position"},
            {"phase": "descending", "knee_deg": 128, "hip_deg": 108, "torso_lean_deg": 22, "coach_hint": "controlled descent"},
            {"phase": "bottom", "knee_deg": 92, "hip_deg": 74, "torso_lean_deg": 26, "coach_hint": "full depth with stable torso"},
            {"phase": "ascending", "knee_deg": 116, "hip_deg": 95, "torso_lean_deg": 24, "coach_hint": "strong ascent keeping chest up"},
            {"phase": "lockout", "knee_deg": 174, "hip_deg": 169, "torso_lean_deg": 9, "coach_hint": "balanced finish"},
        ],
    },
    {
        "label": "shallow-depth",
        "issue": "depth",
        "phases": [
            {"phase": "standing", "knee_deg": 175, "hip_deg": 170, "torso_lean_deg": 9, "coach_hint": "upright setup"},
            {"phase": "descending", "knee_deg": 137, "hip_deg": 116, "torso_lean_deg": 19, "coach_hint": "descent starts normally"},
            {"phase": "bottom", "knee_deg": 114, "hip_deg": 88, "torso_lean_deg": 24, "coach_hint": "rep stays high and does not clearly break parallel"},
            {"phase": "ascending", "knee_deg": 132, "hip_deg": 104, "torso_lean_deg": 22, "coach_hint": "rises early from a shallow bottom"},
            {"phase": "lockout", "knee_deg": 173, "hip_deg": 168, "torso_lean_deg": 10, "coach_hint": "balanced finish"},
        ],
    },
    {
        "label": "forward-lean",
        "issue": "safety",
        "phases": [
            {"phase": "standing", "knee_deg": 176, "hip_deg": 171, "torso_lean_deg": 10, "coach_hint": "upright setup"},
            {"phase": "descending", "knee_deg": 131, "hip_deg": 104, "torso_lean_deg": 34, "coach_hint": "torso starts tipping forward"},
            {"phase": "bottom", "knee_deg": 98, "hip_deg": 76, "torso_lean_deg": 51, "coach_hint": "hips rise with excessive chest drop and forward lean"},
            {"phase": "ascending", "knee_deg": 120, "hip_deg": 91, "torso_lean_deg": 43, "coach_hint": "chest remains low during ascent"},
            {"phase": "lockout", "knee_deg": 174, "hip_deg": 168, "torso_lean_deg": 14, "coach_hint": "finish recovers upright posture"},
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic side-view squat keyframe images with Google.")
    parser.add_argument("--count", type=int, default=3, help="Number of squat examples to generate.")
    parser.add_argument("--output-dir", default="data/generated_frames")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image-format", default="png", choices=["png", "jpeg"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_sdk():
    if genai is None:
        raise SystemExit("Missing dependency: install `google-genai` first with `python3 -m pip install -r requirements.txt`.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_prompt(example_label, phase_spec):
    return (
        "Create a realistic fitness coaching reference image. "
        "One adult athlete performing a bodyweight squat in a clean gym studio. "
        "Strict side-view profile only, camera locked at hip height, full body visible from head to feet, "
        "neutral lighting, realistic anatomy, no text. "
        f"Frame should represent the {phase_spec['phase']} phase of the same squat. "
        f"Approximate joint geometry: knee angle {phase_spec['knee_deg']} degrees, "
        f"hip angle {phase_spec['hip_deg']} degrees, torso lean {phase_spec['torso_lean_deg']} degrees. "
        f"Motion/form goal: {phase_spec['coach_hint']}. "
        f"Reference set: {example_label}."
    )


def build_mediaprose(example):
    chunks = []
    for phase in example["phases"]:
        chunks.append(
            f"{phase['phase']} frame: knee {phase['knee_deg']} deg, hip {phase['hip_deg']} deg, "
            f"torso lean {phase['torso_lean_deg']} deg, note {phase['coach_hint']}."
        )
    return (
        "Synthetic side-view squat keyframe set. "
        "One adult athlete remains in strict side profile with full-body visibility across all frames. "
        + " ".join(chunks)
    )


def choose_examples(count):
    rows = []
    for index in range(count):
        rows.append(FRAME_LIBRARY[index % len(FRAME_LIBRARY)])
    return rows


def save_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def response_parts(response):
    if getattr(response, "parts", None):
        return response.parts
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    return getattr(content, "parts", None) or []


def save_image_part(part, output_path):
    image = part.as_image()
    image.save(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "synthetic_squat_frames_manifest.json"

    selected = choose_examples(args.count)
    manifest_rows = []

    client = None
    if not args.dry_run:
        require_sdk()
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for example_index, example in enumerate(selected, start=1):
        example_id = f"{slugify(example['label'])}-{example_index:02d}"
        example_dir = output_dir / example_id
        example_dir.mkdir(parents=True, exist_ok=True)
        mediaprose = build_mediaprose(example)
        frame_rows = []

        for frame_index, phase_spec in enumerate(example["phases"], start=1):
            prompt = build_prompt(example["label"], phase_spec)
            image_path = example_dir / f"{frame_index:02d}-{phase_spec['phase']}.{args.image_format}"
            frame_row = {
                "phase": phase_spec["phase"],
                "image_path": str(image_path),
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "spec": phase_spec,
                "status": "pending",
            }
            if args.dry_run:
                frame_row["status"] = "dry_run"
                frame_rows.append(frame_row)
                continue

            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=[prompt],
                )
                parts = response_parts(response)
                image_part = next((part for part in parts if getattr(part, "inline_data", None) is not None), None)
                if image_part is None:
                    raise RuntimeError("No image part returned by Gemini image model.")
                save_image_part(image_part, image_path)
                frame_row["status"] = "generated"
            except Exception as exc:  # pragma: no cover
                frame_row["status"] = "error"
                frame_row["error"] = str(exc)
            frame_rows.append(frame_row)

        manifest_rows.append(
            {
                "id": example_id,
                "label": example["label"],
                "issue": example["issue"],
                "model": args.model,
                "mediaprose": mediaprose,
                "frames": frame_rows,
            }
        )
        save_manifest(manifest_path, manifest_rows)

    save_manifest(manifest_path, manifest_rows)
    print(f"Wrote frame manifest to {manifest_path}")
    print(json.dumps(manifest_rows, indent=2))


if __name__ == "__main__":
    main()
