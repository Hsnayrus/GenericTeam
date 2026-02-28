#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover
    genai = None
    types = None


DEFAULT_MODEL = os.environ.get("VEO_MODEL", "veo-3.1-generate-preview")
NEGATIVE_PROMPT = (
    "front view, three quarter view, mirrored camera, cropped feet, cropped head, extra limbs, "
    "multiple people, gym text overlays, subtitles, montage, fast camera moves, fisheye distortion, "
    "cartoon, animation, low detail, broken anatomy"
)

PROFILE_LIBRARY = [
    {
        "label": "side-good-depth",
        "issue": "encouragement",
        "bottom_knee_deg": 92,
        "bottom_hip_deg": 74,
        "torso_lean_deg": 26,
        "tempo_style": "controlled",
        "coach_hint": "full depth with a stable torso",
    },
    {
        "label": "side-shallow-depth",
        "issue": "depth",
        "bottom_knee_deg": 114,
        "bottom_hip_deg": 88,
        "torso_lean_deg": 24,
        "tempo_style": "controlled",
        "coach_hint": "rep stays high and never clearly breaks parallel",
    },
    {
        "label": "side-forward-lean",
        "issue": "safety",
        "bottom_knee_deg": 98,
        "bottom_hip_deg": 76,
        "torso_lean_deg": 51,
        "tempo_style": "grindy",
        "coach_hint": "hips rise with excessive chest drop and visible forward lean",
    },
    {
        "label": "side-soft-lockout",
        "issue": "encouragement",
        "bottom_knee_deg": 96,
        "bottom_hip_deg": 79,
        "torso_lean_deg": 31,
        "tempo_style": "smooth",
        "coach_hint": "solid side-view squat with a mild soft lockout at the top",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic side-view squat videos with Google Veo.")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output-dir", default="data/generated_videos")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"])
    parser.add_argument("--resolution", default="720p", choices=["720p", "1080p", "4k"])
    parser.add_argument("--duration-seconds", type=int, default=8, choices=[4, 6, 8])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_sdk():
    if genai is None or types is None:
        raise SystemExit("Missing dependency: install `google-genai` first with `python3 -m pip install -r requirements.txt`.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY before running this script.")


def slugify(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_media_prose(profile, variant_index):
    return (
        f"Synthetic side-view squat profile {variant_index + 1}. "
        f"One adult athlete is shown in a strict side profile with the full body visible from head to feet. "
        f"The lifter performs a single bodyweight squat in a static tripod camera shot. "
        f"The descent is {profile['tempo_style']}. "
        f"At the bottom, the approximate knee angle is {profile['bottom_knee_deg']} degrees, "
        f"the hip angle is {profile['bottom_hip_deg']} degrees, and the torso lean is {profile['torso_lean_deg']} degrees. "
        f"The main coaching observation is: {profile['coach_hint']}."
    )


def build_prompt(profile, variant_index):
    return (
        "Single-scene fitness coaching clip. "
        "An adult athlete performs exactly one bodyweight squat in a clean gym studio. "
        "Camera is locked off at hip height in a strict side-view profile, full body visible at all times, "
        "neutral lighting, realistic anatomy, realistic motion, no cuts. "
        f"Descent tempo is {profile['tempo_style']}. "
        f"Bottom position should read roughly knee angle {profile['bottom_knee_deg']} degrees, "
        f"hip angle {profile['bottom_hip_deg']} degrees, torso lean {profile['torso_lean_deg']} degrees. "
        f"Motion goal: {profile['coach_hint']}. "
        f"Clip variant {variant_index + 1}."
    )


def choose_profiles(count, seed):
    rng = random.Random(seed)
    selections = []
    for index in range(count):
        profile = PROFILE_LIBRARY[index % len(PROFILE_LIBRARY)].copy()
        profile["variant_seed"] = rng.randint(1, 10_000_000)
        selections.append(profile)
    return selections


def poll_operation(client, operation, poll_seconds):
    while not operation.done:
        print("Waiting for video generation to complete...")
        time.sleep(poll_seconds)
        operation = client.operations.get(operation)
    return operation


def save_manifest(manifest_path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "synthetic_squat_manifest.json"

    profiles = choose_profiles(args.count, args.seed)
    rows = []

    client = None
    if not args.dry_run:
        require_sdk()
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for index, profile in enumerate(profiles):
        item_id = f"{slugify(profile['label'])}-{index + 1:02d}"
        prompt = build_prompt(profile, index)
        mediaprose = build_media_prose(profile, index)
        video_path = output_dir / f"{item_id}.mp4"
        row = {
            "id": item_id,
            "label": profile["label"],
            "model": args.model,
            "video_path": str(video_path),
            "issue": profile["issue"],
            "profile": profile,
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "mediaprose": mediaprose,
            "status": "pending",
        }

        if args.dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue

        try:
            operation = client.models.generate_videos(
                model=args.model,
                prompt=prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio=args.aspect_ratio,
                    resolution=args.resolution,
                    duration_seconds=args.duration_seconds,
                    negative_prompt=NEGATIVE_PROMPT,
                ),
            )
            operation = poll_operation(client, operation, args.poll_seconds)
            generated_video = operation.response.generated_videos[0]
            client.files.download(file=generated_video.video)
            generated_video.video.save(str(video_path))
            row["status"] = "generated"
        except Exception as exc:  # pragma: no cover
            row["status"] = "error"
            row["error"] = str(exc)

        rows.append(row)
        save_manifest(manifest_path, rows)

    save_manifest(manifest_path, rows)
    print(f"Wrote manifest to {manifest_path}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
