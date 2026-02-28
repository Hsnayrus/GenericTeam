# Squat State Schema v1

This document locks the compact parameter set for the squat coach model and data pipeline.

## Goal

Use a small, stable set of engineered squat features.

Do not send raw 33-point pose landmarks to the model.

The pose system is responsible for turning video into geometry.
The model is responsible for coaching decisions from that geometry.

## Canonical Live Features

These are the core fields we are locking for v1:

```json
{
  "phase": "standing|descending|bottom|ascending",
  "rep_number": 3,
  "knee_angle": 101.2,
  "hip_angle": 94.6,
  "torso_angle": 35.1,
  "shin_angle": 29.4,
  "hip_below_knee": true,
  "confidence_min": 0.72
}
```

## Field Definitions

- `phase`
  - Current squat phase from the phase detector.
  - Allowed values: `standing`, `descending`, `bottom`, `ascending`.

- `rep_number`
  - Current rep counter for the set.

- `knee_angle`
  - Averaged bilateral knee angle in degrees.

- `hip_angle`
  - Averaged bilateral hip angle in degrees.

- `torso_angle`
  - Torso lean angle in degrees.

- `shin_angle`
  - Averaged bilateral shin angle in degrees.

- `hip_below_knee`
  - Boolean depth indicator.
  - `true` means the hip midpoint is below the knee midpoint in image space.

- `confidence_min`
  - Minimum visibility/confidence across the key joints used for side-view coaching.
  - This is a tracking quality signal, not a squat quality score.

## Confidence Policy

We still want confidence metrics.

Confidence is used for:

- deciding whether pose-derived geometry is trustworthy
- filtering noisy frames during offline cleaning
- keeping bad tracking out of training data

Confidence is not used for:

- judging whether the rep was good
- replacing squat-quality labels

Recommended runtime signal:

- keep `confidence_min`
- optionally also keep `confidence_avg` in exported frame data for debugging

Model-facing default:

- `confidence_min`

## Optional Field

This field is allowed if it is already easy to compute and stable:

- `knee_valgus_ratio`

It is optional for v1 and should not block the rest of the pipeline.

## Fields Explicitly Out Of Scope For v1

Do not include these in the model contract right now:

- raw 33 landmarks
- raw x/y/z coordinates
- landmark trajectories
- velocities
- accelerations
- camera metadata
- user preference metadata
- complex timing features
- large rep-history payloads

These may be useful later, but they are noise for the current stage.

## System Boundary

The architecture boundary is:

```text
video -> MediaPipe -> deterministic squat state -> local model -> coaching JSON
```

That means:

- video understanding stays outside the model
- biomechanics feature extraction stays outside the model
- the model sees only compact squat state

## Runtime vs Offline

### Runtime

For live inference, send the compact squat state above.

### Offline Export

Offline JSON can keep a little more data for debugging and cleaning, such as:

- `confidence_avg`
- setup/framing flags
- rep summaries
- timestamps

But the training and inference contract should still reduce to the same compact v1 feature set.

## Near-Term Extension

If we move from single snapshots to short temporal input, keep the same fields and repeat them over a short window.

Example:

```json
{
  "rep_number": 3,
  "window": [
    {
      "phase": "descending",
      "knee_angle": 142.3,
      "hip_angle": 128.7,
      "torso_angle": 26.4,
      "shin_angle": 21.8,
      "hip_below_knee": false,
      "confidence_min": 0.84
    },
    {
      "phase": "bottom",
      "knee_angle": 101.2,
      "hip_angle": 94.6,
      "torso_angle": 35.1,
      "shin_angle": 29.4,
      "hip_below_knee": true,
      "confidence_min": 0.79
    }
  ]
}
```

This keeps the schema narrow while allowing temporal reasoning later.

## Decision

For v1, the compact squat-state parameters are locked as:

- `phase`
- `rep_number`
- `knee_angle`
- `hip_angle`
- `torso_angle`
- `shin_angle`
- `hip_below_knee`
- `confidence_min`

Optional:

- `knee_valgus_ratio`
