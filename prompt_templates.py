#!/usr/bin/env python3
import json


FRAMING_HIGHLIGHTS = {
    "ankles_visible": "feet",
    "head_visible": "head",
    "off_center": "center",
    "too_close": "center",
    "too_far": "center",
}


COACH_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "say": {"type": "string", "maxLength": 120},
        "priority": {
            "type": "string",
            "enum": [
                "framing",
                "quality",
                "safety",
                "depth",
                "knees",
                "summary",
                "encouragement",
                "calibration",
            ],
        },
        "ui": {
            "type": "object",
            "properties": {
                "highlight": {
                    "type": "string",
                    "enum": ["feet", "head", "center", "hips", "knees", "torso", "none"],
                },
                "show_checklist": {"type": "boolean"},
            },
            "required": ["highlight", "show_checklist"],
            "additionalProperties": False,
        },
        "cooldown_s": {"type": "number"},
        "calibration_patch": {"type": ["object", "null"]},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["say", "priority", "ui", "cooldown_s", "calibration_patch"],
    "additionalProperties": False,
}


LIVE_FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "reason_code": {
            "type": "string",
            "enum": ["good_depth", "depth_shallow", "torso_forward", "knees_in", "setup_issue"],
        },
        "feedback": {"type": "string", "maxLength": 40},
        "severity": {"type": "string", "enum": ["good", "warn", "bad", "info"]},
        "speak": {"type": "boolean"},
    },
    "required": ["reason_code", "feedback", "severity", "speak"],
    "additionalProperties": False,
}


def coach_instruction_block():
    return """You are a local squat coach.
Return strict JSON only.

Schema:
{
  "say": string,
  "priority": "framing|quality|safety|depth|knees|summary|encouragement|calibration",
  "ui": {
    "highlight": "feet|head|center|hips|knees|torso|none",
    "show_checklist": boolean
  },
  "cooldown_s": number,
  "calibration_patch": null or object,
  "reason": string (optional)
}

Policy:
- Step 1: decide the current mode.
- Step 2: if framing fails, return framing and stop.
- Step 3: if tracking quality fails, return quality and stop.
- Step 4: only then consider safety, depth, knees, encouragement, or summary.
- Pick exactly one priority.
- Prioritize framing, then quality, then safety, then depth, then knees, then encouragement.
- Keep "say" to one short sentence, maximum 14 words.
- Use plain coaching language, not raw variable names.
- Use only the provided structured state.
- If the rep is acceptable, return encouragement.
- Return "calibration_patch" only when mode is "calibrate". Otherwise it must be null.
- Use "reason" for a short internal explanation when helpful.
- Output valid JSON only."""


def live_feedback_instruction_block():
    return """You are a local squat coach running during a live set.
Return strict JSON only.

Schema:
{
  "reason_code": "good_depth|depth_shallow|torso_forward|knees_in|setup_issue",
  "feedback": string,
  "severity": "good|warn|bad|info",
  "speak": boolean
}

Policy:
- Pick the single most useful cue right now.
- Prioritize torso safety, then knee tracking, then depth, then encouragement.
- Return exactly one reason_code.
- Keep "feedback" to one short cue.
- Prefer these exact feedback strings:
  - "Chest up."
  - "Knees out."
  - "Go deeper."
  - "Good depth."
  - "Fix your camera setup."
- Do not explain.
- Do not combine multiple corrections.
- Use only the provided structured state.
- Do not mention raw variable names.
- If there is no important correction, return "good_depth" with "Good depth." and speak false.
- Output valid JSON only."""


def few_shot_examples():
    return [
        {
            "input": {
                "mode": "setup",
                "quality": {
                    "ankles_visible": False,
                    "head_visible": True,
                    "too_close": True,
                    "too_far": False,
                    "off_center": "left",
                    "avg_kpt_conf": 0.45,
                },
                "events_recent": [],
            },
            "output": {
                "say": "Step back until both feet are visible.",
                "priority": "framing",
                "ui": {"highlight": "feet", "show_checklist": True},
                "cooldown_s": 5.0,
                "calibration_patch": None,
                "reason": "Feet visibility is the first setup blocker.",
            },
        },
        {
            "input": {
                "mode": "live",
                "phase": "descending",
                "angles": {"torso_lean_deg": 52, "knee_deg": 126},
                "quality": {
                    "ankles_visible": True,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.84,
                },
                "events_recent": [],
            },
            "output": {
                "say": "Keep your chest up as you lower.",
                "priority": "safety",
                "ui": {"highlight": "torso", "show_checklist": False},
                "cooldown_s": 3.0,
                "calibration_patch": None,
                "reason": "Torso lean is the top safety issue.",
            },
        },
        {
            "input": {
                "mode": "live",
                "phase": "bottom",
                "angles": {"torso_lean_deg": 28, "knee_deg": 118},
                "quality": {
                    "ankles_visible": True,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.88,
                },
                "events_recent": [],
            },
            "output": {
                "say": "Sit a little deeper into the bottom.",
                "priority": "depth",
                "ui": {"highlight": "hips", "show_checklist": False},
                "cooldown_s": 2.5,
                "calibration_patch": None,
                "reason": "Bottom position is shallow.",
            },
        },
        {
            "input": {
                "mode": "live",
                "phase": "bottom",
                "angles": {"torso_lean_deg": 47, "knee_deg": 116},
                "quality": {
                    "ankles_visible": True,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.86,
                },
                "events_recent": [],
            },
            "output": {
                "say": "Keep your chest up into the bottom.",
                "priority": "safety",
                "ui": {"highlight": "torso", "show_checklist": False},
                "cooldown_s": 3.0,
                "calibration_patch": None,
                "reason": "Safety beats depth when torso lean is high.",
            },
        },
        {
            "input": {
                "mode": "live",
                "phase": "descending",
                "angles": {"torso_lean_deg": 24, "knee_deg": 132},
                "quality": {
                    "ankles_visible": True,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.82,
                },
                "events_recent": ["knee_valgus detected"],
            },
            "output": {
                "say": "Push your knees out as you descend.",
                "priority": "knees",
                "ui": {"highlight": "knees", "show_checklist": False},
                "cooldown_s": 3.0,
                "calibration_patch": None,
                "reason": "Knee tracking issue is present.",
            },
        },
        {
            "input": {
                "mode": "live",
                "phase": "descending",
                "angles": {"torso_lean_deg": 18, "knee_deg": 136},
                "quality": {
                    "ankles_visible": True,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.42,
                },
                "events_recent": ["knee_valgus detected"],
            },
            "output": {
                "say": "Hold still so tracking can lock in.",
                "priority": "quality",
                "ui": {"highlight": "none", "show_checklist": True},
                "cooldown_s": 4.0,
                "calibration_patch": None,
                "reason": "Quality gating beats downstream coaching.",
            },
        },
        {
            "input": {
                "mode": "setup",
                "quality": {
                    "ankles_visible": False,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.79,
                },
                "events_recent": [],
            },
            "output": {
                "say": "Step back until both feet are visible.",
                "priority": "framing",
                "ui": {"highlight": "feet", "show_checklist": True},
                "cooldown_s": 5.0,
                "calibration_patch": None,
                "reason": "Framing must be fixed before coaching.",
            },
        },
        {
            "input": {
                "mode": "summary",
                "quality": {
                    "ankles_visible": True,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.84,
                },
                "angles": {"torso_lean_deg": 31, "knee_deg": 102},
                "events_recent": [],
            },
            "output": {
                "say": "Strong set. Keep your chest tall next round.",
                "priority": "summary",
                "ui": {"highlight": "torso", "show_checklist": False},
                "cooldown_s": 0.0,
                "calibration_patch": None,
                "reason": "Summary mode should stay in summary.",
            },
        },
        {
            "input": {
                "mode": "calibrate",
                "quality": {
                    "ankles_visible": True,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.9,
                },
                "calibration": {
                    "depth_target_knee_deg": 100,
                    "torso_warn_deg": 35,
                    "stance_width_norm": 0.35,
                },
                "events_recent": [],
            },
            "output": {
                "say": "Calibration updated for this session.",
                "priority": "calibration",
                "ui": {"highlight": "none", "show_checklist": False},
                "cooldown_s": 2.0,
                "calibration_patch": {"depth_target_knee_deg": 105},
                "reason": "Calibration patches are only valid in calibration mode.",
            },
        },
        {
            "input": {
                "mode": "live",
                "phase": "ascending",
                "angles": {"torso_lean_deg": 23, "knee_deg": 98},
                "quality": {
                    "ankles_visible": True,
                    "head_visible": True,
                    "too_close": False,
                    "too_far": False,
                    "off_center": "none",
                    "avg_kpt_conf": 0.9,
                },
                "events_recent": [],
            },
            "output": {
                "say": "Good rep. Keep that shape.",
                "priority": "encouragement",
                "ui": {"highlight": "none", "show_checklist": False},
                "cooldown_s": 2.0,
                "calibration_patch": None,
                "reason": "No major issue detected.",
            },
        },
    ]


def live_few_shot_examples():
    return [
        {
            "input": {
                "phase": "descending",
                "torso_angle": 52,
                "knee_valgus_ratio": 0.96,
                "knee_angle": 128,
                "depth_history": [146, 138, 128],
            },
            "output": {
                "reason_code": "torso_forward",
                "feedback": "Chest up.",
                "severity": "warn",
                "speak": True,
            },
        },
        {
            "input": {
                "phase": "bottom",
                "torso_angle": 29,
                "knee_valgus_ratio": 0.79,
                "knee_angle": 111,
                "depth_history": [122, 115, 111],
            },
            "output": {
                "reason_code": "knees_in",
                "feedback": "Knees out.",
                "severity": "bad",
                "speak": True,
            },
        },
        {
            "input": {
                "phase": "bottom",
                "torso_angle": 26,
                "knee_valgus_ratio": 0.95,
                "knee_angle": 118,
                "depth_history": [128, 122, 118],
            },
            "output": {
                "reason_code": "depth_shallow",
                "feedback": "Go deeper.",
                "severity": "warn",
                "speak": True,
            },
        },
        {
            "input": {
                "phase": "ascending",
                "torso_angle": 24,
                "knee_valgus_ratio": 0.98,
                "knee_angle": 95,
                "depth_history": [102, 98, 95],
            },
            "output": {
                "reason_code": "good_depth",
                "feedback": "Good depth.",
                "severity": "good",
                "speak": False,
            },
        },
        {
            "input": {
                "phase": "setup",
                "torso_angle": 0,
                "knee_valgus_ratio": 1.0,
                "knee_angle": 180,
                "depth_history": [],
                "side_view_ok": False,
            },
            "output": {
                "reason_code": "setup_issue",
                "feedback": "Fix your camera setup.",
                "severity": "info",
                "speak": True,
            },
        },
    ]


def build_gemma_user_prompt(state, include_examples=True):
    parts = [coach_instruction_block()]
    if include_examples:
        parts.append("Examples:")
        for example in few_shot_examples():
            parts.append("INPUT:")
            parts.append(json.dumps(example["input"], indent=2, sort_keys=True))
            parts.append("OUTPUT:")
            parts.append(json.dumps(example["output"], indent=2, sort_keys=True))
    parts.append("Now review this state.")
    parts.append("INPUT:")
    parts.append(json.dumps(state, indent=2, sort_keys=True))
    parts.append("OUTPUT:")
    return "\n\n".join(parts)


def build_gemma_messages(state, include_examples=True):
    return [{"role": "user", "content": build_gemma_user_prompt(state, include_examples=include_examples)}]


def build_live_feedback_prompt(state, include_examples=True):
    parts = [live_feedback_instruction_block()]
    if include_examples:
        parts.append("Examples:")
        for example in live_few_shot_examples():
            parts.append("INPUT:")
            parts.append(json.dumps(example["input"], indent=2, sort_keys=True))
            parts.append("OUTPUT:")
            parts.append(json.dumps(example["output"], indent=2, sort_keys=True))
    parts.append("Now review this live squat state.")
    parts.append("INPUT:")
    parts.append(json.dumps(state, indent=2, sort_keys=True))
    parts.append("OUTPUT:")
    return "\n\n".join(parts)


def build_live_feedback_messages(state, include_examples=True):
    return [{"role": "user", "content": build_live_feedback_prompt(state, include_examples=include_examples)}]


def has_framing_issue_from_state(state):
    quality = state.get("quality") or {}
    return (
        not quality.get("ankles_visible", True)
        or not quality.get("head_visible", True)
        or bool(quality.get("too_close"))
        or bool(quality.get("too_far"))
        or quality.get("off_center", "none") != "none"
    )


def has_quality_issue_from_state(state):
    quality = state.get("quality") or {}
    confidence = quality.get("avg_kpt_conf")
    return isinstance(confidence, (int, float)) and confidence < 0.55


def has_safety_issue_from_state(state):
    angles = state.get("angles") or {}
    calibration = state.get("calibration") or {}
    torso_lean = angles.get("torso_lean_deg")
    torso_warn = calibration.get("torso_warn_deg")
    return isinstance(torso_lean, (int, float)) and isinstance(torso_warn, (int, float)) and torso_lean >= torso_warn


def has_depth_issue_from_state(state):
    angles = state.get("angles") or {}
    calibration = state.get("calibration") or {}
    knee = angles.get("knee_deg")
    target = calibration.get("depth_target_knee_deg")
    return state.get("phase") == "bottom" and isinstance(knee, (int, float)) and isinstance(target, (int, float)) and knee > target


def has_knees_issue_from_state(state):
    events = " ".join(state.get("events_recent") or []).lower()
    return any(flag in events for flag in ("valgus", "knees_in", "knees in", "knee_valgus"))


def expected_priority_from_state(state):
    mode = state.get("mode")
    if mode == "calibrate":
        return "calibration"
    if mode == "summary":
        return "summary"
    if has_framing_issue_from_state(state):
        return "framing"
    if has_quality_issue_from_state(state):
        return "quality"
    if has_safety_issue_from_state(state):
        return "safety"
    if has_depth_issue_from_state(state):
        return "depth"
    if has_knees_issue_from_state(state):
        return "knees"
    return "encouragement"


def framing_highlight_for_state(state):
    quality = state.get("quality") or {}
    if not quality.get("ankles_visible", True):
        return "feet"
    if not quality.get("head_visible", True):
        return "head"
    if quality.get("off_center", "none") != "none" or quality.get("too_close") or quality.get("too_far"):
        return "center"
    return "center"


def rule_coach_output(state):
    priority = expected_priority_from_state(state)
    if priority == "framing":
        highlight = framing_highlight_for_state(state)
        if highlight == "feet":
            say = "Step back until both feet are visible."
        elif highlight == "head":
            say = "Raise the camera so your head stays in frame."
        else:
            say = "Center yourself in frame before starting the set."
        return {
            "say": say,
            "priority": "framing",
            "ui": {"highlight": highlight, "show_checklist": True},
            "cooldown_s": 5.0,
            "calibration_patch": None,
            "reason": "Framing must be fixed before coaching.",
        }
    if priority == "quality":
        return {
            "say": "Hold still for a second so tracking can lock in.",
            "priority": "quality",
            "ui": {"highlight": "none", "show_checklist": True},
            "cooldown_s": 4.0,
            "calibration_patch": None,
            "reason": "Tracking quality is too low for reliable coaching.",
        }
    if priority == "safety":
        return {
            "say": "Keep your chest up and reduce the forward lean.",
            "priority": "safety",
            "ui": {"highlight": "torso", "show_checklist": False},
            "cooldown_s": 3.0,
            "calibration_patch": None,
            "reason": "Torso safety is the highest-priority live issue.",
        }
    if priority == "depth":
        return {
            "say": "Sit a little deeper to reach your target depth.",
            "priority": "depth",
            "ui": {"highlight": "hips", "show_checklist": False},
            "cooldown_s": 2.5,
            "calibration_patch": None,
            "reason": "Bottom position is too shallow.",
        }
    if priority == "knees":
        return {
            "say": "Push your knees out as you descend.",
            "priority": "knees",
            "ui": {"highlight": "knees", "show_checklist": False},
            "cooldown_s": 3.0,
            "calibration_patch": None,
            "reason": "Knee tracking issue is present.",
        }
    if priority == "calibration":
        calibration = state.get("calibration") or {}
        current_depth = calibration.get("depth_target_knee_deg", 100)
        return {
            "say": "Calibration updated for this session.",
            "priority": "calibration",
            "ui": {"highlight": "none", "show_checklist": False},
            "cooldown_s": 2.0,
            "calibration_patch": {"depth_target_knee_deg": min(115, current_depth + 5)},
            "reason": "Calibration mode allows a patch.",
        }
    if priority == "summary":
        return {
            "say": "Strong set. Keep your chest tall on the next one.",
            "priority": "summary",
            "ui": {"highlight": "torso", "show_checklist": False},
            "cooldown_s": 0.0,
            "calibration_patch": None,
            "reason": "Summary mode should stay high-level.",
        }
    return {
        "say": "Good rep. Keep it consistent.",
        "priority": "encouragement",
        "ui": {"highlight": "none", "show_checklist": False},
        "cooldown_s": 2.0,
        "calibration_patch": None,
        "reason": "No major issue detected.",
    }


def sanitize_coach_output(state, output):
    if not isinstance(output, dict):
        return rule_coach_output(state)
    sanitized = rule_coach_output(state)
    expected_priority = sanitized["priority"]
    if output.get("priority") == expected_priority:
        sanitized["priority"] = output["priority"]
        if isinstance(output.get("cooldown_s"), (int, float)):
            sanitized["cooldown_s"] = float(output["cooldown_s"])
        if isinstance(output.get("reason"), str) and output["reason"].strip():
            sanitized["reason"] = output["reason"].strip()
        ui = output.get("ui")
        keep_model_say = True
        if isinstance(ui, dict):
            if isinstance(ui.get("show_checklist"), bool):
                sanitized["ui"]["show_checklist"] = ui["show_checklist"]
            if ui.get("highlight") == sanitized["ui"]["highlight"]:
                sanitized["ui"]["highlight"] = ui["highlight"]
            else:
                keep_model_say = False
        if expected_priority in {"framing", "quality", "calibration", "summary"}:
            keep_model_say = False
        if isinstance(output.get("say"), str) and output["say"].strip() and keep_model_say:
            sanitized["say"] = output["say"].strip()
        if expected_priority == "calibration" and isinstance(output.get("calibration_patch"), dict):
            sanitized["calibration_patch"] = output["calibration_patch"]
        else:
            sanitized["calibration_patch"] = None
    return sanitized


def rule_live_feedback(snapshot):
    phase = snapshot.get("phase", "standing")
    torso_angle = float(snapshot.get("torso_angle") or 0)
    knee_valgus_ratio = float(snapshot.get("knee_valgus_ratio") or 1)
    knee_angle = float(snapshot.get("knee_angle") or 180)
    recent_depth = snapshot.get("depth_history") or []
    if torso_angle >= 45 and phase != "standing":
        return {"reason_code": "torso_forward", "feedback": "Chest up.", "severity": "warn", "speak": True}
    if knee_valgus_ratio < 0.82 and phase in {"descending", "bottom"}:
        return {"reason_code": "knees_in", "feedback": "Knees out.", "severity": "bad", "speak": True}
    if phase == "bottom" and knee_angle > 100:
        return {"reason_code": "depth_shallow", "feedback": "Go deeper.", "severity": "warn", "speak": True}
    if recent_depth and max(recent_depth[-3:]) > 108:
        return {"reason_code": "depth_shallow", "feedback": "Go deeper.", "severity": "warn", "speak": True}
    return {"reason_code": "good_depth", "feedback": "Good depth.", "severity": "good", "speak": False}


def sanitize_live_feedback(snapshot, output):
    if not isinstance(output, dict):
        return rule_live_feedback(snapshot)
    fallback = rule_live_feedback(snapshot)
    reason_map = {
        "good_depth": ("Good depth.", "good", False),
        "depth_shallow": ("Go deeper.", "warn", True),
        "torso_forward": ("Chest up.", "warn", True),
        "knees_in": ("Knees out.", "bad", True),
        "setup_issue": ("Fix your camera setup.", "info", True),
    }
    sanitized = {
        "reason_code": fallback["reason_code"],
        "feedback": fallback["feedback"],
        "severity": fallback["severity"],
        "speak": fallback["speak"],
    }
    reason_code = output.get("reason_code")
    if reason_code in reason_map:
        feedback, severity, speak = reason_map[reason_code]
        sanitized["reason_code"] = reason_code
        sanitized["feedback"] = feedback
        sanitized["severity"] = severity
        sanitized["speak"] = speak if not isinstance(output.get("speak"), bool) else output["speak"]
        return sanitized
    if output.get("severity") == fallback["severity"]:
        text = (output.get("feedback") or "").strip().lower()
        if "chest" in text:
            sanitized.update({"reason_code": "torso_forward", "feedback": "Chest up.", "severity": "warn"})
        elif "knee" in text:
            sanitized.update({"reason_code": "knees_in", "feedback": "Knees out.", "severity": "bad"})
        elif "deep" in text or "depth" in text:
            if output.get("severity") == "good":
                sanitized.update({"reason_code": "good_depth", "feedback": "Good depth.", "severity": "good", "speak": False})
            else:
                sanitized.update({"reason_code": "depth_shallow", "feedback": "Go deeper.", "severity": "warn"})
        elif "camera" in text or "sideways" in text or "closer" in text or "back" in text:
            sanitized.update({"reason_code": "setup_issue", "feedback": "Fix your camera setup.", "severity": "info"})
        if isinstance(output.get("speak"), bool):
            sanitized["speak"] = output["speak"]
    return sanitized
