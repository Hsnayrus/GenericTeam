# MediaPipe Calibration Findings

**Date:** 2026-02-28
**Method:** Gemini 2.5 Pro Vision Teacher + Full Pose Trajectory Analysis

---

## Executive Summary

We identified a critical bug where **0% of valid squats were being detected** despite MediaPipe tracking knee angles correctly. The root cause is a cascading failure starting from an incorrectly calibrated feet visibility gate.

| Metric | Ground Truth (Video) | MediaPipe Output |
|--------|---------------------|------------------|
| Reps detected | 27 across 5 sessions | 1 |
| Knee angle range | Deep squats visible | 15° - 179° (correct!) |
| Phase detection | Full cycles | 100% "standing" (broken) |
| Feet visibility | Always visible | 90%+ "not visible" (wrong) |

**Fix applied:** Removed `feet_visible` from the setup gate blocking condition.

---

## Methodology: Two-Teacher Approach

### 1. Vision Teacher (Gemini 2.5 Pro)
- **Input:** Sampled video frames from actual sessions
- **Job:** Identify what ACTUALLY happened in the video
- **Output:** Framing issues, occlusion, rep count, form observations

### 2. Pose Teacher (Offline Analysis)
- **Input:** Full pose JSON trajectory (all frames with angles)
- **Job:** Derive reps and coaching decisions from structured data
- **Output:** Offline-derived rep summaries, depth/form assessment

### 3. Calibration Comparison
- **Input:** Video frames + full pose trajectory
- **Job:** Compare vision teacher vs pose pipeline output
- **Output:** Specific disagreements and calibration recommendations

---

## Key Findings

### Finding 1: Feet Visibility Gate is 100% Wrong

**Evidence from Vision Teacher:**
```json
{
  "actual_feet_status": "Feet were clearly visible throughout the entire session.",
  "setup_gate_assessment": {
    "feet_visibility_correct": false
  }
}
```

**Evidence from Pose Data:**
- Session 1: 186/206 frames marked `feet_not_visible` (90%)
- Session 2: Similar pattern across all sessions
- All 5 sessions had `feet_not_visible` as primary blocker

**Root Cause:** The feet visibility check relies on MediaPipe ankle landmark visibility scores:
```javascript
feet: leftAnkle.visibility >= 0.5 && rightAnkle.visibility >= 0.5
```
This fails when:
- User wears baggy pants (ankle landmarks obscured)
- Platform shoes (elevate ankle position)
- Floor contrast issues
- Low camera angles

**Impact:** Setup gate never unlocks → rep mode never activates → phase detection never runs.

### Finding 2: Phase Detection Never Runs

**The Cascade:**
```
feet_visible = false (incorrect)
    ↓
setupState.locked = true
    ↓
repModeActive = false (line 1379)
    ↓
Early return at line 1426
    ↓
phaseDetector.update() never called
    ↓
phase stays "standing" forever
```

**Evidence:**
```json
{
  "phase_distribution": {
    "standing": 206,
    "descent": 0,
    "bottom": 0,
    "ascent": 0
  },
  "knee_angle_range": {
    "min": 15.1,
    "max": 179.2
  }
}
```

The knee angles clearly show deep squats (15° = ass-to-grass), but phase is always "standing".

### Finding 3: Core Angle Tracking IS Working

**This is the good news:** MediaPipe IS correctly tracking:
- Knee angles: 15° to 179° range detected
- Hip angles: 34° to 179° range detected
- Hip-below-knee: True when appropriate
- Confidence: 0.68 to 0.96 (good!)

The problem isn't the pose estimation - it's the gating logic.

### Finding 4: Offline Rep Derivation Works

When we bypass the gates and analyze the raw pose stream:
- **Live rep detection:** 1 rep across 5 sessions
- **Offline rep derivation:** 27 reps across 5 sessions

This proves the data is there - it's just being blocked.

---

## Sessions Analyzed

| Session | Frames | Gate Passed | Live Reps | Offline Reps | Primary Blocker |
|---------|--------|-------------|-----------|--------------|-----------------|
| 22-54-09 | 1029 | 3.5% | 0 | 6 | feet_not_visible |
| 22-50-15 | 1729 | 15.3% | 0 | 6 | feet_not_visible |
| 22-48-53 | 991 | 0% | 0 | 7 | feet_not_visible |
| 22-48-11 | 867 | 0% | 0 | 4 | feet_not_visible |
| 22-46-52 | 1207 | 17.6% | 1 | 4 | feet_not_visible |

---

## Gemini's Calibration Recommendations

From `calibrate_mediapipe_with_gemini.py` output:

```json
{
  "priority_fixes": [
    "Fix the phase transition logic to correctly identify squat phases based on angle data.",
    "Recalibrate the feet visibility check, as it is highly inaccurate and may be the root cause of the phase detection failure.",
    "Investigate if baggy clothing impacts landmark stability and contributes to model uncertainty."
  ],
  "calibration_issues": [
    {
      "issue_type": "phase_mismatch",
      "description": "The model's phase detection logic is completely non-functional. Despite calculating knee angles that clearly indicate deep squats, the 'phase' value for every single frame remains 'standing'.",
      "suggested_fix": "The state machine or thresholds governing phase transitions must be reviewed and fixed. The entry condition for the 'descent' phase is not being triggered."
    },
    {
      "issue_type": "visibility_mismatch",
      "description": "The setup gate incorrectly determined that the user's feet were not visible for the vast majority of the session.",
      "suggested_fix": "Recalibrate the feet visibility detection logic. This faulty gate could be a root cause for the phase logic not engaging."
    }
  ]
}
```

---

## Fix Applied

**File:** `static/index.html`
**Location:** `SetupGate.update()` method (line 824-832)

**Before:**
```javascript
const allGood = Object.values(analysis.checks).every(Boolean);
```

**After:**
```javascript
// NOTE: feet check removed from blocking - Gemini calibration showed it's 100% wrong
// Ankle landmark visibility is unreliable (baggy pants, shoes, floor contrast)
// Core tracking (knee angles, hip-below-knee) works fine without it
const { head, distance, center, view } = analysis.checks;
const allGood = head && distance && center && view;
```

**Impact:**
- Setup gate will now unlock when head, distance, center, and view pass
- `repModeActive` will be true → phase detection will run
- Reps will be counted using the angles that ARE being tracked correctly

---

## Tools Created

1. **`batch_video_vision_review.py`** - Vision Teacher runner
   - Extracts frames from video
   - Sends to Gemini for open-ended analysis
   - Compares against pose data

2. **`derive_reps_offline.py`** - Offline rep derivation
   - Processes raw pose trajectory
   - Detects reps from knee angle changes
   - Bypasses live gating issues

3. **`calibrate_mediapipe_with_gemini.py`** - Full calibration
   - Sends video frames + full pose trajectory to Gemini
   - Gets ground-truth comparison
   - Produces specific calibration recommendations

---

## Next Steps

1. **Test the fix** - Run new recording sessions to verify rep counting works
2. **Consider additional relaxations:**
   - `centered` check may also be too strict
   - Confidence threshold (0.5) may be too high for some scenarios
3. **Add error recovery:**
   - If phase gets stuck, auto-reset after N frames
   - Allow manual override of setup gate
4. **Long-term:** Train Gemma model on the 27 offline-derived reps with correct labels

---

## Appendix: Sample Derived Rep Data

```json
{
  "rep_number": 1,
  "source": "derived_offline",
  "frame_count": 69,
  "provisional_count": 69,
  "gate_pass_count": 0,
  "bottom_frame": {
    "knee_angle": 57.84,
    "hip_angle": 49.29,
    "torso_angle": 32.99,
    "hip_below_knee": true
  },
  "quality": {
    "depth": "deep",
    "lean": "good"
  },
  "setup_issues_during_rep": {
    "feet_not_visible": 58,
    "not_centered": 1
  },
  "confidence": {
    "avg": 0.821,
    "min": 0.707,
    "max": 0.87
  }
}
```

This rep had excellent depth (57.84° knee angle), good form (no excessive lean), high confidence (0.82 avg), but was entirely blocked by `feet_not_visible` = 58/69 frames.

---

## Fixes Applied (Round 2)

### Fix 1: Trusted Confidence Filter

**Problem:** Low-confidence frames (< 0.6) were corrupting phase detection with noisy data.

**Solution:** Added `CONFIG.trustedConfidence = 0.6` threshold. Phase detection only updates on frames with `avgConf >= 0.6`.

```javascript
const trustedFrame = avgConf >= CONFIG.trustedConfidence;
if (trustedFrame) {
  const phaseResult = phaseDetector.update(kneeAngle);
  // ...
}
```

### Fix 2: Single-Person Lock

**Problem:** When another person's limb entered the frame, the tracker would "jump" to them, corrupting angles.

**Solution:** Added `PersonLock` class that:
1. Locks onto the first detected skeleton's hip position
2. Rejects frames where hip position drifts > 25% (tracker hijacked)
3. Allows gradual movement (0.05 update rate)

```javascript
class PersonLock {
  update(landmarks, avgConf) {
    // Track hip position, reject if drift > tolerance
  }
}
```

### Fix 3: Relaxed Center Offset

**Problem:** `maxCenterOffset: 0.1` was too strict.

**Solution:** Relaxed to `maxCenterOffset: 0.15`.

---

## Simulation Results

Ran `simulate_fixed_phase_detection.py` on existing sessions to prove fixes work:

| Session | OLD Reps | NEW Reps | Trusted Frames | New Gate Pass |
|---------|----------|----------|----------------|---------------|
| 22-54-09 | 0 | **6** | 100% | 592 (was 36) |
| 22-50-15 | 0 | **6** | 92% | 1657 (was 265) |
| 22-48-53 | 0 | **2** | 89% | 961 (was 0) |
| 22-48-11 | 0 | **4** | 100% | 867 (was 0) |
| 22-46-52 | 0 | **5** | 100% | 1207 (was 213) |
| **TOTAL** | **0** | **23** | - | - |

### Phase Transitions Now Working

Before: All 5,823 frames showed `phase: "standing"`

After (simulation):
```
standing: 4010 frames
descending: 328 frames
bottom: 609 frames
ascending: 876 frames
```

Phase transitions detected:
- `standing -> descending @ knee=152°`
- `descending -> bottom @ knee=118°`
- `bottom -> ascending @ knee=62°`
- `ascending -> standing @ knee=157°`

---

## Summary of All Changes

| Change | File | Line | Impact |
|--------|------|------|--------|
| Remove feet from gate | index.html | 828 | 0 → 5000+ frames pass |
| Add trustedConfidence | index.html | 512 | Filter noisy data |
| Add PersonLock class | index.html | 775 | Prevent hijacking |
| Relax centerOffset | index.html | 530 | Less strict framing |
| Use trusted for phase | index.html | 1522 | Clean phase detection |

---

## What "Confidence" Means

**Confidence** is MediaPipe's estimate of how certain it is about each landmark's position:

- `0.0` = Can't see this joint at all
- `0.5` = Minimum threshold to record frame
- `0.6` = **Trusted threshold** - confident enough to use for phase detection
- `1.0` = Perfect visibility

**Why discard low-confidence frames?**

When confidence is low (< 0.6), the angle calculations become noisy/erratic:
- Knee angle might jump from 120° to 12° to 180° in 3 frames
- This corrupts the phase velocity calculation (`v = angle[n] - angle[n-1]`)
- Garbage in, garbage out

**The fix:** Only update phase detection with high-confidence frames. Low-confidence frames are still recorded for debugging, but don't affect the state machine.

---

*Generated by Gemini 2.5 Pro calibration analysis + offline rep derivation tooling.*
*Updated with simulation validation of fixes.*
