# CPU-only Slippi replay rendering

Tested 2026-07-11/12 on a GPU-free Daytona Linux container limited to 4 vCPUs and
8 GiB RAM. The host exposed an AMD EPYC 9275F. The test replay was
`experiments_dump/replays/realtimeTest.slp`, with playback bounds `-123..2182`.

## Result

The final CPU build renders and stores the replay in **16.227 seconds**. The
captured game timeline is 38.400 seconds, so this is **2.37x real time**, or
about **142 game frames/second**.

The output is a 142,132,534-byte rawvideo AVI with:

- 224x184 YUV420P frames
- 60 Hz CFR timeline
- 2,304 decoded frames
- 2,298 stored image packets (six held timestamps are materialized by CFR decode)

The benchmark script calculates the expected frame count from the playback JSON
and fails if FFprobe does not report exactly 2,304 frames.

## Primary bottleneck

The main cost is Mesa llvmpipe rasterization plus Dolphin's final render/readback
path, not the emulated 60 Hz timer and not rawvideo encoding. `EmulationSpeed=0`
already means unlimited host execution.

The stock renderer did unnecessary offline work on every frame:

1. draw the XFB to the window/default framebuffer;
2. draw it a second time to the internal-resolution dump FBO;
3. render OSD/debug state;
4. call the window-system swap path;
5. read pixels back and hand one borrowed buffer to the writer.

The final build renders only the dump image, uses `glFlush()` instead of a window
swap, and performs no window/OSD render in dump-only mode. Fractional EFB scaling
with a 204x168 logical headless backbuffer produces the required 224x184
aspect-corrected dump directly.

## Source changes

The reproducible patch is
`patches/ishiiruka-cpu-fast-frame-dump.patch`, based on Ishiiruka commit
`e7711b104b339a99385f2bb12b472d46140a7bc7`.

It contains five relevant changes:

1. **Bounded owned frame queue.** Eight RGBA slots replace the single borrowed
   handoff. A backend may unmap/reuse its PBO immediately; the writer drains in
   order and shutdown waits for the queue. The running flag is changed under the
   queue mutex to avoid a lost-wakeup shutdown race.
2. **Playback state captured at enqueue.** AVI/image eligibility and Slippi frame
   number travel with the pixels. The old worker read mutable global playback
   state later and could emit frames outside the requested interval.
3. **OpenGL dump-only path.** `SLIPPI_DUMP_ONLY=1` skips the default-framebuffer
   draw, OSD/debug draw, `Swap()`, and window clear. It renders once to the
   existing dump FBO and calls non-blocking `glFlush()` to submit surfaceless GL
   work.
4. **Container-safe fastmem.** Linux uses `memfd_create` for Dolphin's 64.25 MiB
   shared arena, with the previous POSIX shared-memory path as fallback. This
   avoids a SIGBUS on Daytona's 64 MiB `/dev/shm` while preserving JIT fastmem.
5. **Slippi-timeline AVI timestamps.** Playback AVI output accepts only strictly
   increasing captured Slippi frame IDs and derives PTS from their 60 Hz deltas.
   Repeated terminal VIs and rewind/savestate traffic are discarded before
   scaling or encoding; non-playback dumps retain the original tick timestamps.

The headless EGL initialization also sets explicit logical backbuffer dimensions;
a surfaceless context otherwise reports 0x0 and falls into incorrect size/restart
behavior.

## Benchmarks

| Configuration | Wall time | Effective FPS | Output |
|---|---:|---:|---|
| Stock GUI/Xvfb, native CPU render | 41 s | 56 | native |
| Stock, tuned fractional/backbuffer capture | 28 s | 82 | 224x184 |
| Patched queue, native internal dump | 29 s | 79 | 642x528 |
| Patched dump-only, native internal dump | 24 s | 96 | 642x528 |
| Patched dump-only, fractional internal dump | **16.227 s** | **142** | **224x184** |

The first two measurements came from the earlier Daytona CPU allocation; the
source-level native queue/dump-only comparison and final run came from the same
4-vCPU EPYC 9275F allocation. The direct native dump-only win is modest under a
surfaceless context; the large final gain comes from removing window work and
reducing llvmpipe's actual raster target.

Rejected probes:

- `MESA_GLTHREAD=true`: 25 s instead of 24 s at native resolution.
- `LP_NUM_THREADS=3` plus `MESA_NO_ERROR=1`: 17 s instead of 16 s at model size.
- `CPUThread=True`: no measurable improvement (16 s), although its 2,304 frame
  hashes matched single-core mode.
- Direct raw RGBA handoff to FFmpeg: still 16 s but increased the file from
  142 MB to 362 MiB, so the experiment was reverted.
- Emulated CPU/GPU clock overrides: not used, because they can alter game timing
  rather than merely removing host throttling.

## Correctness proof

Ishiiruka's established AVI interval is strict at both ends:

```text
startFrame < frame < endFrame
```

For `-123..2182`, the correct CFR count is therefore:

```text
2182 - (-123) - 1 = 2304
```

The earlier 2,303 result was one missing timeline position, not the correct
target. The old asynchronous writer could also produce 2,478 timeline frames by
observing later global playback state. Capturing eligibility with the queued
pixels fixes both errors.

The production regression replay
`c0934ba2773d518f__diamond-diamond-4000c2f74c02661b77738028.slp` exposed a
second timestamp edge. Its semantic interval contains 4,131 frames, but the old
AVI path emitted 4,375, 4,414, and 4,456 decoded frames across three runs while
the LRAS terminal frame remained displayed during shutdown grace. Snapshot
`smash-cpu-renderer-e7711b1-v3` fixes this at the AVI boundary by suppressing
non-increasing Slippi IDs and assigning accepted frames start-anchored,
frame-ID-based PTS. Core
validation now rejects any raw packet count above the semantic interval rather
than tolerating an extra two seconds.

Validation performed:

- final FFprobe count: 2,304 frames;
- full FFmpeg CFR decode: 2,304 frames;
- all 2,304 native-resolution frame MD5s matched between normal presentation
  and dump-only rendering;
- all 2,304 224x184 frame MD5s matched across repeated final runs;
- process exited normally and the queue drained before AVI trailer close.

## Why the old path was not “just async”

Dolphin added a dump thread, but it owned one handoff slot. The graphics backend
had to call `FinishFrameData()` before reusing the mapped staging/PBO memory.
OpenGL/Vulkan staging made GPU readback asynchronous; it did not create a
multi-frame CPU encode/write queue.

Upstream history confirms the design and the same solution space:

- threaded single-slot dumping: https://github.com/dolphin-emu/dolphin/pull/4345
- in-place encoding and its required stall: https://github.com/dolphin-emu/dolphin/pull/4432
- async OpenGL PBO readback: https://github.com/dolphin-emu/dolphin/pull/4344
- buffered Vulkan readback: https://github.com/dolphin-emu/dolphin/pull/4435
- headless dumping that skips screen render/swap and calls `glFlush()`:
  https://github.com/dolphin-emu/dolphin/pull/6216
- staging/readback performance and final-frame correctness fixes:
  https://github.com/dolphin-emu/dolphin/pull/6193
- FFV1 described as slow and replaced upstream:
  https://github.com/dolphin-emu/dolphin/pull/13233

## Windows conclusion

Windows is not the fast CPU-only path for this Ishiiruka version:

- Daytona's usable Windows test sandbox had one vCPU and a service-session
  virtual display; Slippi could not create its render window.
- D3D11 always selects an enumerated adapter with
  `D3D_DRIVER_TYPE_UNKNOWN`; it has no explicit WARP path.
- D3D12 also lacks `EnumWarpAdapter` and explicitly waits for queued work before
  each mapped dump frame.
- DX9 requests a hardware device; its fallback is software vertex processing,
  not CPU rasterization.
- all Windows backends require an HWND/swapchain and retain synchronous readback
  plus presentation work.

A useful Windows implementation would require a new WARP-specific, no-HWND,
no-swapchain dump backend with a rotating staging-texture ring. Linux
EGL/llvmpipe already provides the controllable CPU-only pixel path, so credits
were not spent building that larger Windows rewrite.

## Reproduction

`Dockerfile.headless-egl` is pinned to the tested Ishiiruka commit and applies
the full CPU patch. No new image was needed for the benchmark; the binary was
built once in the Daytona CPU sandbox with:

```bash
cmake -S Ishiiruka -B Ishiiruka/build-headless \
  -DLINUX_LOCAL_DEV=true \
  -DIS_PLAYBACK=true \
  -DENABLE_HEADLESS=true \
  -DENABLE_ANALYTICS=false \
  -DCMAKE_BUILD_TYPE=Release
cmake --build Ishiiruka/build-headless --target dolphin-emu-nogui -- -j6
```

`daytona-source-benchmark.sh LABEL` now defaults to the validated fast CPU
settings: dump-only, rawvideo, four llvmpipe workers, single-core Dolphin,
fractional EFB scale, and a 204x168 logical backbuffer. Use
`extract-cfr-frames.sh VIDEO OUTPUT_DIR` only when individual PNG files are
actually required; otherwise decode the AVI directly to avoid thousands of
small-file writes.

## `core/data` handoff

The existing frame recorder already calls `/opt/slippi/dolphin-emu-nogui`, uses
rawvideo, and decodes video frames through PyAV. No core files were changed in
this experiment. To exercise the fast renderer with that pipeline, build/use
the patched headless image and set `SMASH_VIDEO_EFB_SCALE=0`; the image supplies
the dump-only, single-core, no-window, and 204x168 logical-backbuffer defaults.
Replacing RunPod provisioning with Daytona CPU provisioning is a separate
orchestration change from the emulator bottleneck fixed here.

The end-to-end worker-shape benchmark, including source handling and final MP4
creation, favored one render process per vCPU:

| Daytona shape | Render processes | Measured throughput |
|---|---:|---:|
| 1 vCPU | 1 | 63.36 replays/hour |
| 2 vCPU | 2 | 164.60 replays/hour |
| 4 vCPU | 4 | **369.67 replays/hour** |

The 4-vCPU shape was fastest both per sandbox and per allocated vCPU. Production
therefore uses 23 GPU-free 4-vCPU/8-GiB workers plus one GPU-free
2-vCPU/4-GiB coordinator. The committed artifact contract is a 252x208, 20 Hz
CFR H.264 MP4 with no audio. Results are streamed back as bounded 100-result
`tar.zst` batches containing the source SLP, MP4 (or an explicit no-playable-frame
skip), and metadata; the laptop carries control traffic only.

Live production measurements favor one Drive upload stream with 512 MiB chunks,
100-result batches, a 64-result flush floor, a 2 GiB archive cap, and a 9 GiB
logical spool guarded independently by physical free space. A 30-minute window
rendered 38.85 results/minute and uploaded 39.88 results/minute, so the serial
uploader drained backlog while minimizing requests against the shared Drive
OAuth project's quota. The full run saw 11 quota pauses; every one recovered
through the persisted global backoff without a component or upload failure.

The production fleet completed 10,326 of 10,327 indexed replays in 5:11:58. The
only holdout was a malformed replay whose Dolphin frame progress deterministically
stopped at frame 7200 even though its SLP declared frame 9204. A bounded CPU-only
repair accepted the independently validated stable prefix after 60 seconds with
no frame progress, retained the original SLP, and recorded the discarded 2,004
frame tail explicitly. Its output is a 2,414-frame, 120.700-second, 252x208 CFR20
H.264 MP4 with no audio. The final strict source/manifest audit covers all 10,327
references: eight are explicit `no_playable_frames` skips and 10,319 have video
artifacts. Those videos contain exactly 31,359,489 CFR20 frames, or
435 hours, 32 minutes, 54.45 seconds. GPU count was zero for every seed,
coordinator, benchmark, repair, and production worker.

## Cleanup

The benchmark sandbox `smash-emulator-cpu-build` was deleted after validation.
The final Daytona sandbox list was empty, no custom `smash`/`slippi` snapshot
existed, and local Docker Desktop was stopped.
