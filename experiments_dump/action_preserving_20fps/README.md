# Action-preserving 60 -> 20 FPS experiment

## Result

Do **not** downsample controls with the video. Keep one visual/state observation every three
source frames, but attach all three ordered 60 Hz controller samples to the resulting 20 Hz
transition:

```text
state x[t]                                      state x[t+3]
    |                                               |
    +-- action u[t+1] -- u[t+2] -- u[t+3] ----------+

20 Hz observation: x[t]
20 Hz macro-action: [u[t+1], u[t+2], u[t+3]]
20 Hz target:       x[t+3]
```

For source dynamics `x[t+1] = f(x[t], u[t+1])`, the dataset row is therefore:

```text
y[k]   = x[3k]
U[k]   = [u[3k+1], u[3k+2], u[3k+3]]
y[k+1] = x[3k+3]
```

For full bins, this gives a 3x reduction in image tokens without deleting, averaging, or reordering
a single input. The recommended model representation is `[transition, 3, player, action_features]`.
Embed each sub-action separately with a 0/1/2 position embedding, then concatenate/project or
use three action tokens.

The experiment is CPU-only. It parses `.slp` data with `slippi-js` and uses FFmpeg for optional
lossless video selection; it does not rerun Dolphin or need a GPU.

## What the MIRA/Rocket League work does

[MIRA](https://arxiv.org/abs/2607.05352) is the closest public precedent. Its raw Rocket League
collection has 30 FPS video, 120 Hz physics, and 15 Hz Nexto actions. The preprocessing described
by the [official project](https://mira-wm.com/blog-post/) transcodes video to 20 FPS and samples
the currently held action at each video timestamp. Because each 15 Hz action lasts about 66.7 ms,
longer than a 20 FPS interval (50 ms), no held interval should be skipped completely. Its timestamp
is still quantized to the video grid, so this is not a lossless resampling rule and is unsafe for a
one-frame Melee press lasting about 16.7 ms.

The useful capability in the released MIRA implementation is that video and action rates are not
required to match. Its [dataset loader](https://github.com/mira-wm/mira/blob/db25448391d42547161673a77950a1154d8b5f1f/src/mira/data/dataset.py#L351-L402)
can retain multiple ordered action samples per video frame. Its
[action encoder](https://github.com/mira-wm/mira/blob/db25448391d42547161673a77950a1154d8b5f1f/src/mira/world_model/layers/action_encoder.py#L179-L203)
embeds action samples separately and performs learned, order-sensitive pooling to the latent frame
rate. Released training configs set both video and action target rates to 20 FPS; the
three-60-Hz-actions-per-20-Hz-transition design here is an adaptation of the loader's
higher-action-rate capability, not a claim about what MIRA trained. MIRA also starts generation
from source context but does not constrain a generated terminal state to match the source.

MIRA's simpler lower-rate loader also
[ORs binary controls inside a window](https://github.com/mira-wm/mira/blob/db25448391d42547161673a77950a1154d8b5f1f/src/mira/data/actions.py#L66-L115).
That keyboard-only OR path preserves whether a binary key appeared, but destroys ordering,
duration, and releases and can invent impossible simultaneous-key combinations. It does not
operate on analog controls. This experiment intentionally does not copy that behavior.

The other cited world-model systems avoid this problem by holding one chosen action for an entire
frame-skip interval: [GameNGen](https://arxiv.org/abs/2408.14837) uses four engine frames,
[DIAMOND](https://github.com/eloialonso/diamond/blob/5bcd1599755b4f2fae8e5e079e02f0728e174965/src/envs/atari_preprocessing.py#L66-L93)
repeats one Atari action four times and max-pools its last two images, and
[DreamerV3](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml#L22-L34)
uses Atari action repeat 4. That is valid when collecting new policy rollouts, but cannot losslessly
represent an existing replay whose action changes inside the interval. The distinction is also
formalized in [An Analysis of Frame-skipping in Reinforcement Learning](https://arxiv.org/abs/2102.03718):
coarse sensing needs an open-loop sequence of actions, while ordinary action repeat restricts the
sequence to copies of one action.

## Why common shortcuts fail

- Keeping only every third input deletes one-frame taps and shifts the timing of longer holds.
- OR-ing buttons loses order and releases. `A -> B` and `B -> A` become the same value.
- Averaging sticks can create a direction that the player never selected.
- Majority vote deletes any input shorter than two source frames.
- Keeping only the first or last input cannot represent changes inside the 50 ms interval.
- Async file writing can improve I/O overlap, but it does not repair the information loss caused
  by choosing the wrong temporal representation.

Button press/release edges are included as derived convenience features. The three raw micro-actions
remain authoritative.

## Exact endpoints

Both the original first and last states fit a strict constant 20 Hz grid only when the number of
source transitions is divisible by three. When it is not, `binTicks` retains the real last state
and writes one 1/60 s or 2/60 s final transition with:

- the actual `sourceSteps` and `durationSeconds`;
- a fixed-width `paddedMicroActions[3]` tensor;
- `validActionMask` identifying which entries are real;
- `isPartial: true`.

Repeated padding is never semantic. A trainer must use the mask. The alternative is to crop or
rerender one/two boundary frames; silently pretending the tail is 50 ms changes the trajectory.
The matching video helper writes conventional CFR 20 FPS when the endpoint is aligned, or a
variable-duration final interval so an unaligned exact last image is also retained.

## Verified replay result

The committed summary in `realtime-playable-summary.json` was generated from
`../replays/realtimeTest.slp`, using playable SLP states `-39..2181`. The normalized renderer video
maps SLP frame `-122` to video index 0, so these observation anchors are video indices
`83, 86, ..., 2303`.

```text
source observations:       2,221 at 60 Hz
source transitions/actions: 2,220
output observations:          741 at 20 Hz
output transitions:           740
partial transitions:            0
action round trip:            exact (same SHA-256)
first/last telemetry:         identical canonical extracted-state hashes
```

A naïve every-third-frame action stream would place only 74 of 234 physical-button rising edges
on their true source frame. It would lose the exact timing of 160 presses (68.4%) and completely
hide two press/release pulses. The ordered micro-action representation retains all 234 at their
original frame.

The video helper was also run over 100 locally available CPU-renderer frames. It produced 34
lossless CFR frames tagged `20/1`, starting at timestamp zero. The decoded selected-source and
output streams had the same SHA-256 (`17e7ce51...3485c1`).

## Files

- `binning.mjs`: pure binning, hashing, edge derivation, endpoint handling, and exact action
  round-trip assertion.
- `extract-and-bin-slp.mjs`: strict replay extraction with leaders, followers, rich post state,
  item/stage events, and controller data.
- `downsample-video-vfr.sh`: normalizes a packet-irregular source to a 60 FPS timeline, selects the
  matching lossless observation frames, and keeps an exact non-grid endpoint when necessary.
- `binning.test.mjs`: synthetic pulse/order/endpoint/error tests.
- `video.test.sh`: synthetic non-grid-tail frame-count and decoded-pixel hash test.
- `SCHEMA.md`: precise JSONL contract and training guidance.
- `ONLINE_60HZ_INPUT_20HZ_MODEL.md`: online controller polling, action-token alignment, and latency
  contract.
- `realtime-playable-summary.json`: small reproducible result record; large generated JSONL files
  are ignored under `runs/`.

## Run it

From `experiments_dump/`:

```bash
node action_preserving_20fps/extract-and-bin-slp.mjs \
  --input replays/realtimeTest.slp \
  --output-dir action_preserving_20fps/runs/realtime-playable \
  --first-frame -39 \
  --last-frame 2181 \
  --video-first-slp-frame -122
```

The output is `manifest.json`, `observations.jsonl`, and `transitions.jsonl`. The extractor fails on
missing source frames or incomplete active-player pre/post records. V1 rejects PAL rather than
silently mishandling 50 -> 20 FPS, which needs alternating two- and three-step action groups.
V1 deliberately rejects rates other than NTSC 60 -> 20.

To select the matching images from a rendered video whose normalized index 0 is SLP frame `-122`:

```bash
action_preserving_20fps/downsample-video-vfr.sh \
  input.avi output-20fps.mkv 83 2303 3
```

The FFmpeg filter begins with `fps=60`. This is required for the verified CPU renderer artifact:
its AVI has fewer stored packets than 60 Hz timeline slots, so selecting raw packet number would
misalign later SLP frames. The output is lossless FFV1/Matroska. `videoIndex` is authoritative only
when `--video-first-slp-frame` came from a calibrated renderer mapping. A synthetic 11 -> 5 frame
tail test produced identical decoded-frame SHA-256 values before and after selection. The test also
checks non-zero phase, exact manually enumerated source frames, zero-based PTS, and aligned 20 FPS
container metadata.

## Integration recommendation

Keep the core renderer at its fastest native output and perform this cheap CPU post-process. For
training, store one observation image/state per anchor and three actions per transition. Validate
every shard by flattening valid micro-actions and comparing its canonical hash to the original
60 Hz action stream over source frames `firstFrame+1..lastFrame`. Also compare first/last telemetry
or pixel hashes and verify explicit
`videoIndex -> sourceFrame` mappings.

The state hashes in this experiment cover canonical, JSON-sanitized Slippi telemetry. Undefined and
non-finite fields become `null`; these are not Dolphin savestates. The video helper selects original
decoded images, but the committed replay summary does not claim an independent pixel-endpoint check
against a current renderer artifact.

A transient one-frame visual flash can still disappear from a 20 FPS observation stream even when
its action is retained. If such events are prediction targets, preserve them as auxiliary 60 Hz
event labels or render at the native rate for that task.

The CPU rendering investigation and emulator-speed learnings are documented separately in
`../fast_replay_probe/pixel-video-replay-learnings.md`.
