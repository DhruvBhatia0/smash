# Pixel Video Replay Learnings

Date: 2026-07-09

This note summarizes the replay-speed investigation for Slippi `.slp` files. It is intentionally
verbose: the goal is to preserve the operational recipe, the dead ends, the measured baselines, and
the intuition for what is probably bottlenecking us.

The short version:

- If we need **semantic frame data** only, we can skip Dolphin entirely and parse `.slp` files
  directly in well under `1s`.
- If we need **full pixel frames**, the best verified method so far is a patched headless EGL
  Playback Dolphin build on an RTX 4090, dumping **rawvideo AVI** at internal 1x resolution.
- Best verified pixel/video time on `realtimeTest.slp`: **`18.783s`** for a full frame-by-frame
  video.
- The GPU is actually used in that recipe, but not saturated. The remaining bottleneck looks like
  Dolphin emulation/render/readback/frame handoff, not just video encoding.

## Objective And Definitions

Original long-term target:

```text
1,000,000 SLP emulations in 24 hours ~= 11.57 SLP/s globally
With enough workers, target budget is about 1 second per SLP per worker.
```

There are two very different interpretations of "frame data":

1. **Semantic frame data**: positions, inputs, action states, percent, stocks, hitlag, animation
   index, etc. This already exists in `.slp`.
2. **Pixel frame data**: actual rendered frames, either as PNGs or a video containing every frame.

The semantic path is solved for the `1s` target. The pixel path is not.

The rest of this note focuses mostly on pixel/video output, but includes the semantic result for
context because it changes how we should design the larger pipeline.

## Main Artifacts

Experiment root:

```text
experiments_dump/fast_replay_probe/
```

Important files:

- `findings.md`: running measurements and short conclusions.
- `render-ffv1-replay.sh`: wrapper used for FFV1/rawvideo/H.264 video runs.
- `Dockerfile.headless-egl`: source-build image experiment for headless EGL Playback Dolphin.
- `ishiiruka-headless-egl-playback.patch`: source patch for no-GUI/headless Playback Dolphin.
- `runpod-headless-egl-test.py`: helper for RunPod source-build testing.
- `extract-frame-state-binary.mjs`: semantic frame-state extractor.
- `extract-frame-state-batch.mjs`: semantic batch extractor.
- `bench-frame-state-process.mjs`: semantic fresh-process benchmark.
- `verify-frame-state-summary.mjs`: semantic summary verifier.

Important run artifacts:

```text
experiments_dump/fast_replay_probe/runs/runpod-headless-egl-source-build-secure/
experiments_dump/fast_replay_probe/runs/runpod-gpu-video/
experiments_dump/fast_replay_probe/runpod-ffv1-png-manifest.json
```

Primary sample replay:

```text
experiments_dump/replays/realtimeTest.slp
SLP frame range: -123..2182
SLP frame entries: 2306
Frame log entries during playback: 2307, range -123..2183
```

The video outputs typically report `2303` frames. That appears to be the expected frame count after
playback/video dump details shake out, but I still treated shorter H.264 outputs as suspect.

## Best Pixel Recipe Found

Best verified full visual/video result:

```text
RunPod: RTX 4090 secure pod
Build: patched Project Slippi Ishiiruka / Playback Dolphin source build
Backend: headless EGL + OGL, no Xvfb
Output: rawvideo AVI
Resolution: 642x528
Frames reported by ffprobe: 2303
Duration reported by ffprobe: 38.383333s at 60 fps
Wall time: 18.783137607s
Output size: 1,168,032,254 bytes
```

Run artifact:

```text
experiments_dump/fast_replay_probe/runs/runpod-headless-egl-source-build-secure/remote-out/render-20260709T052513Z-rawvideo-internal-1x/
```

Manifest summary:

```json
{
  "dolphinExitStatus": 0,
  "currentFrameLogEntries": 2307,
  "currentFrameRange": { "first": -123, "last": 2183 },
  "targetEndFrame": 2182,
  "videoBackend": "OGL",
  "dolphinCpuCore": "1",
  "audioBackend": "Null",
  "useFFV1": "False",
  "dumpCodec": "rawvideo",
  "dumpFormat": "avi",
  "internalResolutionFrameDumps": "True",
  "efbScale": "2"
}
```

FFprobe summary:

```text
codec: rawvideo
container: AVI
pix_fmt: yuv420p
resolution: 642x528
fps: 60
nb_frames: 2303
duration: 38.383333s
size: 1,168,032,254 bytes
bitrate: ~243 Mbps
```

GPU evidence for this run:

```text
EGL vendor: NVIDIA
GPU: NVIDIA GeForce RTX 4090
nvidia-dmon samples: 17
max SM: 40%
avg SM: 19.35%
max power: 63 W
encoder util: 0% (expected for rawvideo)
```

So the recipe does use the GPU for OpenGL rendering. It is not the bad Xvfb/llvmpipe path. But GPU
utilization is modest, so this is not a pure "buy bigger GPU, get linear speedup" situation.

## Exact Recipe

This recipe assumes the patched source-build image/setup from `Dockerfile.headless-egl` and
`ishiiruka-headless-egl-playback.patch`.

The high-level flow:

1. Build Playback Dolphin from source with no-GUI CLI playback support.
2. Patch the OGL backend so it can initialize headlessly via EGL without Xvfb.
3. Package `Binaries/Sys`, including `Sys/GameSettings`, so the Melee ISO boots correctly.
4. Prepare a playback JSON with `command`, `replay`, `startFrame`, and `endFrame`.
5. Run Dolphin with unlimited emulation speed, OGL, null audio, video dumping enabled.
6. Dump a video through Dolphin/FFmpeg AVIDump.
7. Watch `[CURRENT_FRAME]` logs and gracefully terminate Dolphin when `endFrame` is reached so the
   AVI header/trailer finalizes.

The wrapper invocation shape:

```bash
SLIPPI_USE_FFV1=False \
SLIPPI_DUMP_CODEC=rawvideo \
SLIPPI_DUMP_FORMAT=avi \
SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS=True \
SLIPPI_EFB_SCALE=2 \
render-ffv1-replay.sh \
  --replay-json /workspace/headless-src/input/playback.json \
  --iso /workspace/headless-src/input/melee.iso \
  --output-dir /workspace/headless-src/out/render-rawvideo-internal-1x \
  --timeout-seconds 120 \
  --video-backend OGL \
  --cpu-core 1 \
  --audio-backend Null \
  --no-xvfb
```

Despite the script name, `render-ffv1-replay.sh` became the generic video wrapper. The key env vars
override it away from FFV1 and into rawvideo.

The wrapper writes Dolphin configs before launching:

```ini
# Dolphin.ini
[Core]
EmulationSpeed = 0.00000000
CPUCore = 1
CPUThread = True
Fastmem = True
GFXBackend = OGL

[Movie]
DumpFrames = True
DumpFramesSilent = True

[DSP]
Backend = Null
DumpAudio = False
DumpAudioSilent = True
```

And:

```ini
# GFX.ini
[Settings]
DumpFramesAsImages = False
InternalResolutionFrameDumps = True
EFBScale = 2
UseFFV1 = False
DumpCodec = rawvideo
DumpFormat = avi
BitrateKbps = 2500
```

Important details:

- `EmulationSpeed = 0.00000000` means unlimited speed, not realtime.
- `CPUCore = 1` is JIT64.
- `audioBackend = Null` avoids audio output work.
- `DumpFramesAsImages = False` avoids direct PNG dumping.
- `InternalResolutionFrameDumps = True` is required for usable full-resolution frame dumps.
- `EFBScale = 2` forces Dolphin's 1x internal scale in this build/config. Without this, Linux
  defaulted to 2x in one run and cost more.
- `UseFFV1 = False` plus `DumpCodec = rawvideo` avoids FFV1 compression.
- `--no-xvfb` matters because this is a headless EGL build. The old Xvfb path fell back to software
  rendering in one GPU test.

The Dolphin command shape generated by the wrapper:

```bash
"$DOLPHIN_BIN" \
  -u "$USER_DIR" \
  -i "$REPLAY_JSON" \
  -e "$ISO" \
  --hide-seekbar \
  --cout \
  --batch \
  -v OGL
```

The end-frame watcher:

- Reads `render-ffv1.log`.
- Finds the last `[CURRENT_FRAME] <n>` line.
- When `n >= targetEndFrame`, sends `TERM` only to the Dolphin process matching the intended
  binary, user dir, and replay JSON.
- This fixed earlier zero-byte video failures where the watcher killed the parent `timeout` process
  and prevented AVIDump from finalizing the file.

## Why Rawvideo Won

FFV1 is lossless and sane for storage, but the encoder is expensive. On the same headless EGL 4090
source build:

| Path | Wall time | Frames | Resolution | Output | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FFV1 internal 1x | `53.399s` | `2303` | `642x528` | `1.035GB` | lossless BGRA, CPU/IO heavy |
| rawvideo internal 1x | `18.783s` | `2303` | `642x528` | `1.168GB` | fastest verified full frame-by-frame video |
| H.264 default internal 1x | `15.200s` | `2265` | `642x528` | `58.7MB` | fastest wall time, but short frames |
| H.264 NVENC internal 1x | `15.902s` | `2297` | `642x528` | `37.3MB` | NVENC works, smaller output, still short frames |

Rawvideo won because it removes most video compression CPU work. It still writes a huge file, but
for this sample the disk path could keep up.

The H.264 results are tantalizing but not accepted as "best" because the frame counts were short.
They might become viable with deeper AVIDump/H.264 finalization debugging, but right now rawvideo is
the fastest full-frame result I trust.

## Baselines

### Local Mac Direct PNG

```text
Wall: 188.56s, timed out at 180s but reached frame 2183
PNG files: 4118
PNG bytes: 784,474,405
Frame log entries: 2307
PNG mtime span: 170.894s
PNG write rate: about 24.09 PNG/s
Duplicate factor: 4118 / 2307 = 1.78 PNGs per frame-log entry
```

Learnings:

- Direct PNG output is very expensive.
- The current image dump path writes more PNGs than gameplay frames.
- Slippi-specific frame gating exists in parts of AVIDump, but `Renderer::DumpFrameToImage` writes
  PNGs whenever frame dumping is enabled.
- PNG file count and compression both hurt.

### Docker Desktop Baseline

```text
Wall: 91.631s
Exit: Dolphin 134
PNG files: 3
Current frames: -123..-122
```

Learnings:

- Docker Desktop on the local Mac was an amd64-on-arm64 wiring check, not useful performance data.
- It proved the wrapper plumbing could start, but not a meaningful speed baseline.

### RunPod CPU Direct PNG

```text
Pod type: cpu3c
Extra overhead: about 5 minutes to receive the 1.4GB ISO
Observed PNGs: 16,311 for only 2,307 frame-log entries before timeout/failure
```

Learnings:

- The ungated PNG path was catastrophically over-dumping after replay end.
- Fixing end-frame stop and deduping PNG writes was necessary before any direct-PNG path could be
  taken seriously.

### RunPod CPU FFV1-To-PNG

```text
Pod type: cpu3c
Requested frame range: -123..2182
Current-frame log range: -123..2183
Dolphin FFV1 render time: about 192s
FFV1 AVI: 462,030,564 bytes
Extracted PNGs: 2303
Extracted PNG bytes: 668,561,367
PNG extraction: FFmpeg compression_level=1,pred=none
```

Learnings:

- Video-first avoids the direct PNG duplicate explosion.
- Cheap CPU rendering is still nowhere near usable.
- Even after video-first, PNG materialization writes hundreds of MB for a 38s sample.

### RTX PRO 6000 Xvfb/OGL

```text
Host: RTX PRO 6000 Blackwell Server Edition pod
Video backend: OGL under Xvfb
Wall: 58.734s
Output: FFV1 AVI, 380x312, bgra, 461,635,584 bytes
GPU render utilization: 0%
glxinfo: Mesa llvmpipe, Accelerated: no
```

Learnings:

- This looked like a GPU pod but did not use GPU rendering.
- The improvement over cheap CPU came from host CPU, not the RTX card.
- Xvfb + OGL is dangerous for this workload because it can silently route through software.

### RTX PRO 6000 Vulkan Retest

```text
Requested frame range: -123..300
Wall: 120.455s
Current-frame entries: 0
Video output: none
GPU max SM: 0%
GPU max memory util: 0%
GPU max power: 36W
```

Learnings:

- Vulkan was visible to system tools, but the Slippi Playback build did not advance into replay
  playback through this wrapper path.
- The playback JSON `"mode": "queue"` was not the cause.
- For this codebase/build, Vulkan needs source-level work before it is a viable path.

### RTX 4090 Headless EGL Source Build

This is the path that produced the best result.

Learnings:

- Building from source was necessary to get no-GUI CLI playback plus headless EGL.
- The pod used NVIDIA EGL rather than llvmpipe.
- GPU rendering was active, but moderate.
- Encoder choice mattered a lot, but even rawvideo stayed at ~19s.

## Optimizations That Worked

### 1. Avoid Direct PNG During Emulation

This is the largest conceptual win.

Direct PNG means:

- thousands of independent files,
- PNG compression work,
- filesystem metadata overhead,
- duplicate/out-of-window frame hazards,
- huge output volume.

Video-first is a better intermediate for pixel data. It does not solve final PNG extraction, but it
separates "render the game" from "materialize individual image files."

### 2. Stop At End Frame Cleanly

The wrapper now watches `[CURRENT_FRAME]` and terminates Dolphin when the target end frame is
reached.

This fixed:

- dumping past replay end,
- long timeout waits,
- stale output confusion,
- zero-byte AVI files from bad process killing.

Important detail: kill the actual Dolphin process, not the parent `timeout` command.

### 3. Headless EGL Instead Of Xvfb

This avoided the software-rendering trap seen on the RTX PRO 6000 pod.

Evidence:

- Bad path: Xvfb/OGL reported llvmpipe and GPU utilization 0%.
- Good path: EGL vendor NVIDIA; rawvideo run had max SM 40%.

### 4. Internal 1x Resolution

`SLIPPI_EFB_SCALE=2` forces 1x in this Dolphin config.

This matters because:

- Linux defaulted to a higher internal scale in one run.
- Higher internal scale increases render/readback/video payload.
- 1x still produced `642x528`, which is usable for full gameplay frames.

### 5. Rawvideo Instead Of FFV1

Measured improvement:

```text
FFV1 internal 1x: 53.399s
rawvideo internal 1x: 18.783s
```

Why:

- FFV1 is lossless but CPU-heavy.
- Rawvideo avoids compression.
- The output file is larger, but writing 1.17GB was still faster than compressing FFV1.

### 6. Null Audio

The wrapper sets audio backend to `Null` and disables audio dumping.

This is not individually benchmarked, but it is a sensible default: we only care about visual frame
data.

### 7. Clear Stale Framedumps

The wrapper deletes old `framedump*.avi/mkv/mp4/mov/nut` before each run.

This mattered because one earlier H.264 measurement accidentally picked up stale rawvideo output.

## Optimizations That Helped Only Partially

### PNG Compression Knobs

A small PNG probe over 100 existing gameplay PNGs:

| Encoder settings | Wall time | Output bytes |
| --- | ---: | ---: |
| FFmpeg default PNG | `4.14s` | `35,743,519` |
| `compression_level=0`, `pred=none` | `0.76s` | `70,181,400` |
| `compression_level=1`, `pred=none` | `1.76s` | `42,983,070` |

This helps PNG extraction/materialization, but not enough. For full replays, PNG output volume is
still massive.

### NVENC

NVENC worked in the sense that `h264_nvenc` produced a small file and `nvidia-dmon` showed encoder
activity:

```text
H.264 NVENC internal 1x: 15.902s
Frames: 2297
Output: 37.3MB
GPU encoder max: 4%
```

But wall time barely improved versus default H.264 and the frame count was short. It reduced file
size, not the true pipeline bottleneck.

### H.264 Default

```text
H.264 default internal 1x: 15.200s
Frames: 2265
Output: 58.7MB
```

This was the fastest nominal wall time, but it was not accepted because the output was short.

Possible explanations to investigate:

- AVIDump finalization issue.
- Encoder delay/B-frame/flushing issue.
- Frame counting mismatch from container metadata.
- H.264 path dropping/duplicating due to timing or termination.

Until verified, rawvideo remains the fastest trustworthy full-frame recipe.

## Optimizations That Did Not Work

### Xvfb On GPU Pod

It did not use GPU rendering:

```text
Renderer: llvmpipe
GPU utilization: 0%
```

### Vulkan In Existing Playback Build

The wrapper could see Vulkan/NVIDIA, but replay playback never advanced. No frame logs, no video.

### Generic Compression Of PNG Archives

Existing PNG files did not compress much with tar+zstd:

```text
300 PNGs original: 91.17 MiB
tar+zstd of PNGs: 90.93 MiB
```

Already-compressed PNGs are a bad target for generic archive compression.

## Pros And Cons Of The 18.8s Rawvideo Recipe

### Pros

- Fastest verified complete pixel/video path so far.
- Produces full frame-by-frame video: `2303` frames at 60 fps.
- Avoids PNG file explosion during emulation.
- Avoids FFV1 compression CPU cost.
- Uses actual NVIDIA EGL/GPU rendering, unlike the Xvfb path.
- Clean end-frame stop prevents long timeout tails.
- Rawvideo is simple to post-process with FFmpeg.
- Good intermediate if downstream can consume video or extract selected frames.

### Cons

- Still far from the `1s` target.
- Still above the user's `10s` unacceptable threshold.
- Output is huge: `1.17GB` for a 38s sample.
- At scale, rawvideo output would create extreme IO/storage pressure.
- GPU is used but underutilized; simply renting a bigger GPU may not help much.
- Rawvideo is an intermediate, not a compact archival format.
- If the final product must be PNGs, PNG extraction/materialization still has to happen later.
- H.264/NVENC is smaller and slightly faster, but currently not frame-complete enough to trust.
- Requires a patched source build, not a stock Slippi Playback Dolphin image.
- Requires careful environment/config setup; easy to accidentally fall back to software rendering.

## What The 18.8s Number Means

The sample replay's video duration is `38.383s`.

The rawvideo render took `18.783s`.

So the best verified pixel path is about:

```text
38.383 / 18.783 ~= 2.04x realtime
```

That is a real improvement over the CPU and PNG baselines, but it is not the order-of-magnitude
breakthrough needed for million/day pixel-video extraction.

For a 3-4 minute game, if scaling were linear, this path might be roughly:

```text
3 min game: 180 / 2.04 ~= 88s
4 min game: 240 / 2.04 ~= 118s
```

That estimate is rough. Some fixed overheads disappear on longer games, but output size and frame
count scale linearly.

## Bottleneck Hypothesis

Current best guess:

1. Dolphin emulation and replay playback are still CPU-heavy.
2. OpenGL render happens on GPU, but GPU is not saturated.
3. Internal-resolution frame dumping likely forces GPU readback/synchronization.
4. AVIDump/frame handoff serializes parts of the pipeline.
5. Rawvideo removes encoder compression bottleneck, revealing render/readback/handoff as the floor.

Evidence:

- Rawvideo cut FFV1 from `53.399s` to `18.783s`, so compression was a big piece.
- NVENC reduced output size dramatically but did not reduce wall time much, so after compression is
  cheap the remaining path is elsewhere.
- GPU SM maxed at `40%` and averaged `19.35%` in the rawvideo run, so the GPU is not fully loaded.
- PCIe receive/transmit numbers were active in `nvidia-dmon`, consistent with frame transfer/readback
  being part of the cost.

## Semantic Frame-State Path

This is not pixel output, but it is important.

Direct `.slp` parsing with `@slippi/slippi-js` writes fixed-width binary rows:

```text
format: SLPFRAMESTATEv2
record size: 132 bytes
sample records: 4612 player-frame rows
sample output: 608,784 bytes
```

Verified results:

```text
slippi-js public fixture corpus, warm process:
  60 files, zero failures, max 0.342092s

slippi-js public fixture corpus, fresh process per file:
  60 files, zero failures, max 0.704498s

Hugging Face tournament Bowser sample, warm process:
  25 files, zero failures, max 0.256620s

Hugging Face tournament Bowser sample, fresh process per file:
  25 files, zero failures, max 0.397421s
```

Verifier checks:

- zero failures,
- zero occupied player/follower frame records skipped,
- binary size equals `rowCount * recordBytes`,
- metadata last frame matches where present,
- no extraction over `1s`.

The semantic path is the only path currently meeting the `1s` per SLP worker target.

## External Context That Shaped The Work

Dolphin/TAS video dumping guidance recommends unlimited speed and FFV1 lossless video dumps instead
of individual image files for normal capture workflows:

```text
https://tasvideos.org/EncodingGuide/VideoDumping/Dolphin
```

Project Slippi's Ishiiruka repo documents Playback builds and command-line playback usage:

```text
https://github.com/project-slippi/Ishiiruka
```

Community Slippi video tools use Playback Dolphin plus FFmpeg rather than a separate fast renderer:

```text
https://github.com/davisdude/slp2mp4
https://github.com/kevinsung/slp-to-video
https://github.com/NunoDasNeves/slp-to-mp4
```

The Hugging Face public dataset card documents a large public Slippi corpus and states that `.slp`
files contain complete game state and controller inputs for every frame:

```text
https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7
```

FFmpeg PNG encoder docs confirmed that PNG has `compression_level` and prediction knobs:

```text
https://ffmpeg.org/ffmpeg-codecs.html#png
```

## Recommended Next Experiments

### 1. Debug H.264 Frame Completeness

H.264 is currently the most promising output format if we can make it frame-complete:

```text
H.264 default: 15.200s but 2265/2303 frames
H.264 NVENC: 15.902s but 2297/2303 frames
```

Things to test:

- Longer graceful shutdown delay after end frame.
- Explicit AVIDump flush/finalize patch.
- H.264 settings with no B-frames and intra-only / low-latency mode.
- Compare actual decoded frame count with `ffprobe nb_frames`.
- Let Dolphin run a few frames past `endFrame`, then trim video afterward.

If H.264 can be made frame-complete, it may beat rawvideo while reducing output size by ~30x.

### 2. Pipe Raw Frames To FFmpeg Directly

Current path writes via Dolphin AVIDump. A custom pipe might reduce overhead:

- Dolphin renders/readbacks frames.
- Write raw frames to stdout or a named pipe.
- FFmpeg handles muxing/encoding.

Potential benefit:

- Avoid AVI/container overhead inside Dolphin.
- More control over encoder flush.
- Easier to test rawvideo, FFV1, H.264, NVENC, AV1, etc.

Risk:

- Requires more invasive source patching.
- Readback remains expensive.

### 3. GPU Readback Optimization

If frame readback is the floor, investigate:

- async PBO readback,
- double/triple buffering readback,
- avoiding synchronous `glReadPixels` stalls,
- dumping from the final internal framebuffer before extra copies,
- direct GPU-to-encoder path if feasible.

The GPU is not saturated. Reducing synchronization may matter more than faster GPU hardware.

### 4. Test Longer Real Tournament Games

The main sample is only ~38s. Need a longer corpus:

- 3-4 minute games,
- multiple stages,
- singles/doubles,
- low/high item and effect activity,
- different character combinations.

Track:

- wall time,
- video frames,
- output bytes,
- GPU SM,
- CPU utilization,
- disk write throughput,
- replay frame count.

### 5. Keep Workers Warm

At scale:

- do not transfer ISO per replay,
- do not build Dolphin per worker job,
- keep pod/container warm,
- keep ISO and Dolphin user dir local,
- copy only `.slp` input and final output,
- avoid per-file SSH setup overhead.

### 6. Decide The Real Output Contract

The biggest architectural decision:

- If downstream can use semantic frame state, use direct `.slp` parsing.
- If downstream needs visual frames but can use video, use rawvideo/H.264 intermediate.
- If downstream truly needs every PNG, expect huge storage and extraction cost.

For PNG-at-scale:

```text
2306 frames * ~190KB average PNG ~= 438MB for a 38s sample
14400 frames * ~190KB ~= 2.7GB for a 4 minute game
1,000,000 games ~= petabytes/day
```

That is a storage/product decision, not just an emulator optimization problem.

## Current Bottom Line

Best trusted pixel/video path:

```text
18.783s per sample replay
rawvideo AVI
headless EGL OGL
RTX 4090
642x528
2303 frames
1.17GB output
```

This path uses the GPU, but only moderately. The remaining problem is likely Dolphin render/readback
and frame handoff. The next best bet is not simply "bigger GPU"; it is making the video dump path
more asynchronous and/or making H.264/NVENC frame-complete.
