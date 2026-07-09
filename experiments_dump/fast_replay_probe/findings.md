# Fast Replay Findings

## Baseline

Sample: `../replays/realtimeTest.slp`.

- SLP frame range: `-123..2182` (`2,306` entries from `@slippi/slippi-js`).
- Parser-only control: `0.026545s` for the full file.
- Local macOS Playback Dolphin baseline: `188.56s` wall, timed out at `180s` but reached frame `2183`.
- Baseline output: `4,118` PNGs, `784,474,405` bytes.
- Replay frame log entries: `2,307`.
- PNG creation mtime span: `170.894s`, or about `24.09` PNG/s during active writes.

The current image dump path is doing more PNG work than replay-frame work:

```text
4,118 PNG files / 2,307 CURRENT_FRAME entries = 1.78 PNGs per replay frame
```

This is consistent with the source: Slippi-specific gating exists in `AVIDump::Start` and
`AVIDump::AddFrame`, but `Renderer::DumpFrameToImage` always writes a PNG when frame dumping is
enabled.

## Validated External Context

- Dolphin's frame dumping guidance recommends unlimited speed and, on Linux, FFV1 lossless video
  frame dumps instead of individual image files for normal capture workflows:
  https://tasvideos.org/EncodingGuide/VideoDumping/Dolphin
- Project Slippi's Ishiiruka repo documents separate playback builds and command-line playback
  usage for Linux:
  https://github.com/project-slippi/Ishiiruka
- Community Slippi video tools (`slp2mp4`, `slp-to-video`, `slp-to-mp4`) use Playback Dolphin and
  FFmpeg rather than a separate fast renderer:
  https://github.com/davisdude/slp2mp4
  https://github.com/kevinsung/slp-to-video
  https://github.com/NunoDasNeves/slp-to-mp4
- The Hugging Face `erickfm/slippi-public-dataset-v3.7` card describes a CC0 public corpus of
  about `95,000` raw tournament `.slp` files and notes that `.slp` files contain complete game
  state and controller inputs for every frame:
  https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7
- A 2025 CS231n Slippi frame project independently reports that existing frame-dump
  infrastructure requires re-simulating matches and was only practical for a small number of
  replays:
  https://cs231n.stanford.edu/2025/papers/text_file_840589945-Parsing_Super_Smash_Bros__Melee_Frames.pdf
- FFmpeg's PNG encoder exposes `compression_level` from `0..9` and prediction options; these are
  lossless speed/size knobs:
  https://ffmpeg.org/ffmpeg-codecs.html#png

## Compression Probe

Using 100 existing gameplay PNGs:

| Encoder settings | Wall time | Output bytes | Notes |
| --- | ---: | ---: | --- |
| FFmpeg default PNG | `4.14s` | `35,743,519` | smallest of tested outputs |
| `compression_level=0`, `pred=none` | `0.76s` | `70,181,400` | ~5.4x faster, ~2.0x bytes |
| `compression_level=1`, `pred=none` | `1.76s` | `42,983,070` | ~2.4x faster, ~1.2x bytes |

Compression settings help materially, but cannot by themselves make a full replay sub-second.

An existing `../png_compression_probe` experiment explored post-processing storage. Its `probe-300`
run found that `300` original PNGs were `91.17 MiB`; tar+zstd of the PNG files was still
`90.93 MiB`, so already-compressed PNGs do not benefit much from generic archival compression. Its
small `smoke-12` run showed lossless video can compress better (`x264rgb_qp0_veryslow` at `2.35x`
vs PNG, `FFV1` at `1.28x`), but that changes the intermediate format and still requires PNG
materialization if full PNGs are the final contract.

## Prototype Patch

Patch artifact: `ishiiruka-fast-png-frame-dump.patch`.

It changes only the experiment clone under `fast_replay_probe/source/Ishiiruka`:

- Gate PNG image dumps to active Slippi playback frames.
- Deduplicate image dumps by `currentPlaybackFrame`.
- Add env-controlled PNG knobs:
  - `SLIPPI_PNG_COMPRESSION_LEVEL=0..9`
  - `SLIPPI_PNG_FILTER_NONE=1`

Expected impact on the sample:

- PNG count should fall from `4,118` toward `2,306`.
- PNG encode time can drop further with `SLIPPI_PNG_COMPRESSION_LEVEL=1` or `0`.
- File count remains one PNG per replay frame, so filesystem overhead remains high.

## Practical Mitigations

1. First patch: stop duplicate/out-of-window PNGs in `Renderer::DumpFrameToImage`.
2. Use low PNG compression for generation speed, then optionally recompress later offline.
3. If downstream can tolerate a two-stage format, dump FFV1/lossless video first and materialize
   PNGs later with FFmpeg. This follows Dolphin/TAS and Slippi community practice and avoids
   thousands of synchronous PNG creates during emulation, but it does not remove the final PNG
   materialization cost.
4. Keep RunPod workers warm and localize ISO/replay/output storage; per-file SSH/rsync and ISO
   transfer will dominate once render time comes down.

## Docker And RunPod Baselines

Fresh renderer image pushed for native RunPod tests:

```text
ttl.sh/smash-slippi-renderer-23e29240-5b1d-4d03-bf25-84aa001b7ba8:24h
```

Docker Desktop is running (`29.5.3`, Linux VM on arm64), but the renderer image is amd64. The
Docker Desktop baseline is therefore a wiring check, not a performance source. It failed after
`91.631s` with Dolphin exit `134`, only `3` PNGs, and current frames `-123..-122`.

Native RunPod direct-PNG baseline on `cpu3c` / JIT64:

- The one-off pod spent about `5m` just receiving the `1.4 GB` ISO.
- Dolphin reached the requested end frame, but the ungated PNG path kept dumping after replay end.
- Remote observation showed `16,311` PNGs for only `2,307` frame-log entries before timeout/failure.
- This confirms the direct PNG path must be gated/deduped in the emulator before more queue work.

## FFV1-To-PNG RunPod Probe

Derived image used for the measured RunPod probe:

```text
ttl.sh/smash-slippi-renderer-ffv1-png-cbf39c1b-a684-422d-a3ae-268374a47714:24h
```

Result manifest: `runpod-ffv1-png-manifest.json`.

On the same `cpu3c` RunPod CPU pod:

```text
Requested frame range: -123..2182
Current-frame log range: -123..2183 (2,307 entries)
Dolphin FFV1 render time to end frame: about 192s
FFV1 AVI: 462,030,564 bytes
Extracted PNGs: 2,303 files, 668,561,367 bytes
Pipeline: FFV1 video, then FFmpeg PNG extraction with compression_level=1,pred=none
```

This avoids the massive duplicate PNG dumping seen in the direct-PNG path, but it is still nowhere
near the `10s` per SLP requirement on this cheap CPU setup. FFmpeg extraction alone wrote about
`669 MB` of PNGs; that output contract creates a hard floor even if emulation becomes much faster.

The experimental `render-ffv1-replay.sh` wrapper now watches `[CURRENT_FRAME]` and gracefully stops
Dolphin after `endFrame`, so the manual stop used in this probe should be repeatable in the next
image build. Fresh image containing that wrapper patch:

```text
ttl.sh/smash-slippi-renderer-ffv1-png-ce72e713-81e5-42d3-af30-0aa6cd9846ae:24h
```

## GPU RunPod Video Probe

Requested cheapest available among RTX 4090, RTX 5090, and RTX PRO 6000. RunPod pricing listed
RTX 4090 at `$0.69/hr`, RTX 5090 at `$0.99/hr`, and RTX Pro 6000 at `$2.09/hr`; 4090 and 5090 had
no capacity, so the API allocated an RTX PRO 6000 Blackwell Server Edition pod at `$1.69/hr`.

Pod result artifacts:

- `runpod-gpu-video-pod.json`
- `runpod-gpu-video-remote-summary.txt`
- `runs/runpod-gpu-video/`
- `runpod-gpu-vulkan-retest-pod.json`
- `runs/runpod-gpu-vulkan-retest/`

Backend checks:

- `xvfb-run glxinfo -B` reports Mesa `llvmpipe`, `Accelerated: no`.
- `vulkaninfo --summary` sees `NVIDIA RTX PRO 6000 Blackwell Server Edition`.
- The Slippi/Ishiiruka Vulkan smoke test consumed GPU, but never reached `[CURRENT_FRAME]` before
  the `120s` timeout, so it produced no video.
- The working OGL path under Xvfb produced video, but GPU monitoring showed `max_sm=0` and
  `avg_pwr_w=36`, confirming it was not using the RTX card.

Follow-up Vulkan retest:

```text
Replay JSON: minimal command/replay/startFrame/endFrame, no queued mode
Requested frame range: -123..300
Elapsed wall time: 120.455s
Exit status: 1
Current-frame entries: 0
Video output: none
GPU dmon samples: 115
GPU max SM: 0%
GPU max memory util: 0%
GPU max power: 36 W
```

This rules out the earlier `"mode": "queue"` playback JSON as the cause. In this image, Vulkan is
available to system tools but this Slippi Playback build does not advance into replay playback via
the wrapper's Vulkan path. The existing source also warns that Playback builds have had Vulkan
issues, so the next real GPU attempt should be source/build work rather than more pod shape tests.

Full video-only OGL/Xvfb result on the RTX PRO 6000 host:

```text
Replay: realtimeTest.slp
Frame range: -123..2182
Frame-log entries: 2,306
Elapsed wall time: 58.734s
FFV1 AVI: 461,635,584 bytes
Video stream: ffv1, 380x312, bgra, 60 fps
Host CPU: AMD EPYC 9334, 128 visible CPUs
GPU render utilization: 0%
```

This is faster than the cheap CPU pod's `~192s`, but the improvement comes from the host CPU, not
GPU rendering. It is about `0.65x` realtime for the sample, still far from the `10s` target.

The end-frame watcher had a `set -euo pipefail` bug: the background watcher could exit before the
first `[CURRENT_FRAME]` line because `grep` returned no matches. The local wrapper now guards that
initial no-match case, and `latest-ffv1-png-image.txt` points at a rebuilt image with the fix.

## RTX 4090 Headless EGL Source-Build Probe

RunPod allocation request was cheapest available among RTX 4090, RTX 5090, and RTX PRO 6000. The
community cloud had no matching capacity, so a secure RTX 4090 pod was allocated at `$0.69/hr`.
The pod was deleted after the probe; `runs/runpod-headless-egl-source-build-secure/cleanup.json`
shows HTTP `204` delete success and no remaining pods.

This probe built a patched Playback Dolphin from source with:

- no-GUI playback CLI support for `-e`, `-i`, `-u`, `-v`, `--batch`, and `--cout`;
- EGL/headless OGL initialization without Xvfb;
- packaged `Sys/GameSettings` playback Gecko codes;
- clean end-frame termination so AVIDump writes headers/trailers;
- configurable video dump codec/container/bitrate/EFB scale in `render-ffv1-replay.sh`;
- `AVIDump` encoder-by-name support, which allows `DumpCodec=h264_nvenc`.

Important bug fixes from this run:

- Without `Binaries/Sys`, the source build could not boot the Melee ISO.
- The first end-frame watcher killed both `timeout` and Dolphin because it matched the parent
  command line; that produced zero-byte AVIs. Matching only the Dolphin argv fixed clean shutdown.
- The wrapper now clears stale `framedump*` outputs and recognizes `avi`, `mkv`, `mp4`, `mov`, and
  `nut`; one earlier H.264 row was invalid because it copied a stale rawvideo AVI.
- Headless backbuffer dumping without internal-resolution frame dumps is fast but unusable:
  `15.8s`, `2303` frames, but only `16x12`.
- Linux defaulted to `EFBScale=4` (`2x`); setting `SLIPPI_EFB_SCALE=2` forces `1x`.

Valid full-visual-video results on the sample:

| Path | Wall time | Frames reported by video | Resolution | Output | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FFV1, internal 1x | `53.399s` | `2303` | `642x528` | `1,035,392,728` bytes | lossless BGRA, too much CPU/IO |
| rawvideo, internal 1x | `18.783s` | `2303` | `642x528` | `1,168,032,254` bytes | avoids compression, huge I/O |
| H.264 default encoder, internal 1x | `15.200s` | `2265` | `642x528` | `58,744,456` bytes | fast/small but video frame count is short |
| H.264 NVENC, internal 1x | `15.902s` | `2297` | `642x528` | `37,313,106` bytes | NVENC works, but wall time barely improves |

The key result is that changing the video encoder gets the visual path from `53s` to about `15s`,
but not below that. At this point the bottleneck is mostly Dolphin render/readback/frame handoff,
not FFV1 compression. NVENC reduces output size but does not change the wall-clock floor much.

Non-pixel frame-state extraction is a different story. `extract-frame-state-binary.mjs` parses the
same full SLP into fixed-width binary player-frame records in-process in tens of milliseconds; the
20-run process-level benchmark has median wall time around `0.120s` and p95 around `0.160s`. If the
contract can be "game state per frame" rather than pixels, the under-`1s` target is already met on
this sample.

## Direct SLP Frame-State Extraction

The currently reliable under-`1s` path is to skip emulation for frame data that already exists in
the `.slp` file. `extract-frame-state-batch.mjs` writes fixed-width binary player-frame records:

```text
format: SLPFRAMESTATEv2
record size: 132 bytes
sample input: 535,610 bytes
sample frames: -123..2182 (2,306 frame entries)
sample records: 4,612 player-frame rows
sample output: 608,784 bytes
```

Each record contains the frame number, player slot, follower flag, pre-frame controller/input
fields, pre/post action state, positions, percent, shield/stocks/jumps, hitlag, animation index,
and basic self-induced speed fields. This is semantic game-state frame data, not pixels.
`v2` widens `post.instanceHitBy` to 16 bits because public Slippi edge-case replays include stage
hazard/platform hit instances above `255`.

Validation runs:

| Benchmark | Scope | Wall time |
| --- | --- | ---: |
| Single smoke | 1 SLP, one process | `0.039284s` in-process |
| Local batch | 21 local `.slp` paths x 3 repeats, one process | `1.074016s` total for 63 runs |
| Local batch per-file | same 63 runs | median `0.016108s`, p95 `0.020125s`, max `0.036994s` |
| Fresh process | 20 separate Node invocations on sample | median `0.119945s`, p95 `0.127213s`, max `0.401799s` |
| Verified public corpus | 60 `project-slippi/slippi-js` sample SLPs, one warm process | `2.608616s` total, zero failures |
| Verified public corpus per-file | same 60 runs | median `0.011232s`, p95 `0.174650s`, max `0.342092s` |
| Verified public corpus fresh-process | same 60 SLPs, one Node process per SLP | `10.158975s` total, zero failures |
| Verified public corpus fresh-process per-file | same 60 runs, includes startup/spawn | median `0.129632s`, p95 `0.283442s`, max `0.704498s` |
| HF tournament sample | first 25 `BOWSER/` SLPs from `erickfm/slippi-public-dataset-v3.7`, one warm process | `2.302894s` total, zero failures |
| HF tournament sample per-file | same 25 runs | median `0.080840s`, p95 `0.143579s`, max `0.256620s` |
| HF tournament sample fresh-process | same 25 SLPs, one Node process per SLP | `5.925945s` total, zero failures |
| HF tournament sample fresh-process per-file | same 25 runs, includes startup/spawn | median `0.224542s`, p95 `0.367061s`, max `0.397421s` |

Artifacts:

- `runs/frame-state-batch-smoke/batch-summary.json`
- `runs/frame-state-batch-local-3x/batch-summary.json`
- `runs/frame-state-batch-process-20x/process-bench-summary.json`
- `runs/frame-state-batch-slippi-js-corpus-v2-verified/batch-summary.json`
- `runs/frame-state-process-slippi-js-corpus-v2-verified/process-bench-summary.json`
- `runs/hf-bowser-25/replays/BOWSER/`
- `runs/frame-state-batch-hf-bowser-25-v2/batch-summary.json`
- `runs/frame-state-process-hf-bowser-25-v2/process-bench-summary.json`

Verification gate:

```text
node verify-frame-state-summary.mjs --max-seconds 1 \
  runs/frame-state-batch-slippi-js-corpus-v2-verified/batch-summary.json \
  runs/frame-state-process-slippi-js-corpus-v2-verified/process-bench-summary.json \
  runs/frame-state-batch-hf-bowser-25-v2/batch-summary.json \
  runs/frame-state-process-hf-bowser-25-v2/process-bench-summary.json
```

The verifier checks zero failures, zero occupied player/follower frame records skipped, valid
`binaryBytes == rowCount * recordBytes`, matching `metadataLastFrame` where present, and no
per-file run over `1s`. Latest result:

```text
warm maxObservedSeconds: 0.342092, errors: []
cold maxObservedSeconds: 0.704498, errors: []
hf warm maxObservedSeconds: 0.256620, errors: []
hf cold maxObservedSeconds: 0.397421, errors: []
```

Caveat: the early extra `.core_runs` SLP files discovered locally were byte-identical to the sample
(`sha1=a129183350917200ec5a9ad2de580bdefc54ceed`). The later `slippi-js` corpus run covers more
diverse public edge cases, including incomplete games, teams/FFA placements, stage hazards, items,
rollback/finalized-frame cases, PAL/NTSC, nametags, and doubles. This still is not a million-file
production corpus, but it is enough to show that the parser path is not just a lucky single-file
result.

## Hard Constraint

Full PNGs for every frame imply very large output. Even after removing duplicate dumps, this sample
projects to roughly:

```text
2,306 frames * 190 KB average PNG ~= 438 MB for a 37s game
14,400 frames * 190 KB ~= 2.7 GB for a 4 minute game
1,000,000 games ~= 2.7 PB
```

That volume is possible only with a storage plan designed around petabytes/day scale or by changing
when/how PNGs are materialized.
