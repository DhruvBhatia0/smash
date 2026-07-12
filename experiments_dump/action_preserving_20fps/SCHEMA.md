# `ACTION_PRESERVING_20FPS_V1` schema

## Alignment contract

Slippi pre-frame input at source frame `f` precedes the post-frame state at `f`. Consequently,
the transition from post state `f-1` to post state `f` uses pre-frame input `f`:

```text
post(f-1) -> pre-input(f) -> post(f)
```

For a full 60 -> 20 transition starting at source frame `s`, `microActions` contains source frames
`s+1`, `s+2`, and `s+3`, in that order. `actionSourceFrames` makes the alignment explicit.

## `observations.jsonl`

One object per retained image/state anchor:

```json
{
  "observationIndex": 0,
  "sourceIndex": 0,
  "sourceFrame": -39,
  "videoIndex": 83,
  "timestampSeconds": 0,
  "stateSha256": "...",
  "state": {
    "frame": -39,
    "players": [],
    "items": [],
    "stageEvents": [],
    "frameStart": null
  }
}
```

`timestampSeconds` is relative to the first extracted state. When a calibrated
`--video-first-slp-frame` mapping is supplied, `videoIndex` refers to the normalized 60 FPS video
timeline, not necessarily the source container's stored packet ordinal. Otherwise it is `null`.

## `transitions.jsonl`

One object per pair of observations. Important fields:

- `startSourceFrame`, `endSourceFrame`: exact boundary states.
- `sourceSteps`: 3 normally for 60 -> 20; 1 or 2 only for an endpoint-preserving tail.
- `durationSeconds`: `sourceSteps / manifest.sourceFps`.
- `microActions`: variable-length list containing only real ordered controls.
- `paddedMicroActions`: always length 3 for tensor batching.
- `validActionMask`: authoritative mask for `paddedMicroActions`.
- `actionSourceFrames`, `actionVideoIndices`: exact source identities.
- `actionBefore`: pre-frame input at the starting frame, included only so first-substep button edges
  are derivable. It produced the starting post state and must not be applied as a fourth action.
- `buttonEdges`: pressed/released masks derived independently for `buttons` and
  `physicalButtons` at every micro-step.

Each micro-action has one entry per present leader/follower:

```json
{
  "sourceFrame": -38,
  "players": [{
    "playerIndex": 1,
    "isFollower": false,
    "joystickX": 0,
    "joystickY": 0,
    "rawJoystickX": 128,
    "cStickX": 0,
    "cStickY": 0,
    "trigger": 0,
    "physicalLTrigger": 0,
    "physicalRTrigger": 0,
    "buttons": 0,
    "physicalButtons": 0
  }]
}
```

Button masks are unsigned 32-bit values represented by exact JavaScript/JSON integers. Analog
values are not averaged or quantized by this experiment.

## `manifest.json`

The manifest records replay SHA-256, selected source/video indices, endpoint hashes, full/partial
counts, and both source/reconstructed action hashes. The action-hash domain is every transition
input at source frames `firstFrame+1..lastFrame`; the input that produced the first extracted post
state is context and intentionally excluded. A valid shard must satisfy:

```text
actionRoundTripExact == true
sourceActionSha256 == reconstructedActionSha256
first observation state hash == firstStateSha256
last observation state hash == lastStateSha256
```

`strictCfrEndpointCompatible` says whether all intervals have
`sourceStepsPerFullTransition / sourceFps` duration. If false, the final transition is short and
must be masked/duration-conditioned rather than treated as an ordinary full step.

State hashes cover canonical, JSON-sanitized Slippi telemetry, not a Dolphin savestate or pixel
hash. Undefined and non-finite replay values become `null`. Pixel endpoints require a separate
decoded-frame hash check.

## Model input

Preferred tensorization:

```text
analog:          float32 [T, R, P, 8]
buttons:         uint32  [T, R, P]
physicalButtons: uint32  [T, R, P]
validActions:    bool    [T, R]
actorPresent:    bool    [T, R, P]
analogValid:     bool    [T, R, P, 8]
```

V1 requires `R = 3`, `sourceFps = 60`, and `targetFps = 20`. The eight analog features are
joystick X/Y, raw joystick X, C-stick X/Y, processed trigger, and physical L/R triggers. Build `P`
from stable `(playerIndex, isFollower)` keys and use both actor-presence and analog-validity masks;
older Slippi versions can expose an analog field as `null`. Embed substeps separately with their
intra-bin position; do not collapse them using OR, mean, majority vote, or endpoint selection.
