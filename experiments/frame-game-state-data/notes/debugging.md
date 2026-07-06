# Debugging Replay Image Export

Current goal: render a `.slp` replay into real Melee pixels, align those pixels with Slippi frame numbers, then join image paths back onto game-state rows for training.

## Working Export Path

1. Use Slippi Playback Dolphin, not the Homebrew/netplay Slippi Dolphin app.
2. Use the patched local Playback build at:
   `/Users/dhruv/code/smash/experiments/frame-game-state-data/tools/patched-playback/Slippi Dolphin.app`
3. Verify the app bundle has `Contents/Resources/Sys/GameSettings/GALE01r2.ini` containing `$Required: Slippi Playback`.
4. Launch Playback Dolphin with:
   - `-u <isolated user dir>`
   - `-i replay-playback.capture-test.json`
   - `-e <Melee ISO>`
   - `--hide-seekbar --cout --batch`
5. Enable Dolphin image dumping with:
   - `Dolphin.ini` `[Movie] DumpFrames = True`
   - `Dolphin.ini` `[Movie] DumpFramesSilent = True`
   - `GFX.ini` `[Settings] DumpFramesAsImages = True`
6. Copy the raw dump to `frames/<replay command>/`.
7. Align `framedump_*.png` order with `[CURRENT_FRAME]` log order into `game-frames/<command id>/`.
8. Join aligned image paths into per-player rows in `data/*.image-rows.jsonl`.

## Root Cause

The installed Playback Dolphin could replay the `.slp`, but it produced no frame media files on macOS. The source showed why in `Source/Core/VideoCommon/RenderBase.cpp`: `Renderer::IsFrameDumping()` only returned `true` for `[Movie] DumpFrames` when the build had `HAVE_LIBAV` or was Windows.

The macOS Playback Dolphin binary contains the existing PNG fallback strings but is not linked against libav/ffmpeg, so the old guard prevented the fallback from ever running.

The tracked patch in `patches/ishiiruka-macos-png-frame-dump.patch` removes that guard:

```cpp
if (SConfig::GetInstance().m_DumpFrames)
    return true;
```

That lets the existing `DumpFramesAsImages` path write `framedump_*.png` files to `User/Dump/Frames`.

## Bundle Pitfall

The first patched source build dumped PNGs but booted to Online Play. The binary was compiled with `-DIS_PLAYBACK=true`, but the app bundle still had netplay `GameSettings` under `Contents/Resources/Sys/GameSettings`.

`scripts/prepare-playback-build.sh` fixes that by copying base `Data/Sys`, replacing `Sys/GameSettings` with `Data/PlaybackGeckoCodes`, and verifying `$Required: Slippi Playback`. It expects a local source checkout at `source/Ishiiruka`; the cleaned folder keeps only the compact patched app bundle, not the full 1.4 GB source/build tree.

`scripts/render-replay-debug.sh` now preflights this too. If the playback Gecko code is missing, it exits instead of dumping menu frames.

## Logs

Useful local logs:

- Render wrapper log, generated after `pnpm render`:
  `/Users/dhruv/code/smash/experiments/frame-game-state-data/logs/render-replay-debug.log`
- Dolphin log, generated after `pnpm render`:
  `/Users/dhruv/code/smash/experiments/frame-game-state-data/playback-debug-user/Logs/dolphin.log`

The render wrapper requires both PNG output and `[CURRENT_FRAME]` log entries. PNGs alone are not success because the wrong app/resources can dump menu frames.

## Current Verified Outputs

- Aligned game frames:
  `/Users/dhruv/code/smash/experiments/frame-game-state-data/game-frames/realtimeTest-capture-test`
- Frame mapping:
  `/Users/dhruv/code/smash/experiments/frame-game-state-data/game-frames/realtimeTest-capture-test/frames.jsonl`
- State/input rows with image paths:
  `/Users/dhruv/code/smash/experiments/frame-game-state-data/data/realtimeTest.capture-test.image-rows.jsonl`

The verified capture-test run produced:

- 1504 raw `framedump_*.png` files.
- 1025 Playback `[CURRENT_FRAME]` entries for replay frames `-123..901`.
- 1025 aligned training images in `game-frames/realtimeTest-capture-test`.
- 2050 image-backed state rows, two players per frame.
- 479 trailing raw dump frames dropped because they are post-window `Waiting for game` frames.

The raw `frames/` dump is generated and was removed during cleanup. Re-running `pnpm render` recreates it, and `pnpm align` recreates the aligned set.

## Commands

From `/Users/dhruv/code/smash/experiments/frame-game-state-data`:

```bash
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm render
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm align
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm attach:images
```

If rebuilding the patched app from source, clone Ishiiruka under `source/Ishiiruka`, apply `patches/ishiiruka-macos-png-frame-dump.patch`, build the Playback target, then run:

```bash
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm prepare:playback-build
```

The Melee ISO currently used by the render script is:
`/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso`

## Replay Download and Metadata Ingestion

The batch ingestion path is class-based:

- `DownloadManifest` loads and validates replay URL manifests.
- `ReplayDownloadJob` normalizes ids, filenames, output paths, and optional hashes.
- `ConcurrentDownloader` downloads with a fixed worker pool and `.part` files.
- `SlpMetadataExtractor` extracts game, player, winner, and skill-signal metadata.
- `ReplayIngestionPipeline` coordinates download, metadata extraction, and aggregate index writing.

Run it with:

```bash
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm ingest:slp -- --manifest download-manifests/slippi-js-samples.json --concurrency 4
```

Generated output:

- Downloaded `.slp` files: `replays/downloaded/`
- Metadata JSON and index: `metadata/replays/`

True player rank/MMR is not normally available in `.slp` files. The metadata stores `skillSignals` from replay stats, including inputs per minute, conversions, damage per opening, neutral-win ratio, action counts, and final stocks when available.

## Corpus Frame Queue

`scripts/process-frame-queue.py` runs the dataset queue:

- `SlpCorpusProducer` is the single producer thread. It discovers `.slp` files and blocks on `queue.put` when the bounded queue is full.
- `FrameProcessingConsumer` creates its runtime in `__init__`, then consumes replay jobs until it receives a stop sentinel.
- `FrameJobProcessor` extracts structured state, invokes the runtime, aligns rendered frames, and attaches image paths back to frame-state rows.
- Runtime choices are `plan`, `local-macos`, `docker`, and `runpod`.
- `RunPodCpuEmulatorRuntime` creates CPU Pods through the RunPod REST API. The RunPod API key is read from `RUNPOD_API_KEY`; it should not be written into manifests.
- `DockerEmulatorRuntime` requires Docker Desktop to be running and a renderer image that implements `/opt/slippi-renderer/render-replay.sh`.

Decision note: RunPod is the first scalable runtime to pursue. Docker is installed locally, but the Docker daemon was not reachable during this setup, and the only renderer proven to emit screenshots is the patched macOS Playback Dolphin app. A Docker path is still useful later because the same Linux renderer image can be used locally and on RunPod.

Validation commands run:

```bash
python3 -m py_compile scripts/process-frame-queue.py scripts/clean-up-runpod-cpu-pods.py scripts/frame_queue/*.py
python3 scripts/process-frame-queue.py --runtime plan --consumers 2 --queue-size 1 --max-jobs 2 --run-id test-plan-queue replays/downloaded/slippi-js-samples
python3 scripts/process-frame-queue.py --runtime docker --dry-run --plan-only --docker-image slippi-renderer:local --consumers 2 --queue-size 1 --max-jobs 2 --run-id test-docker-plan replays/downloaded/slippi-js-samples
python3 scripts/process-frame-queue.py --runtime runpod --dry-run --plan-only --runpod-image slippi-renderer:runpod --consumers 2 --queue-size 1 --max-jobs 2 --run-id test-runpod-plan replays/downloaded/slippi-js-samples
python3 scripts/process-frame-queue.py --runtime local-macos --consumers 1 --queue-size 1 --max-jobs 1 --start-frame -123 --end-frame 10 --timeout-seconds 45 --run-id test-local-render replays/realtimeTest.slp
python3 scripts/clean-up-runpod-cpu-pods.py
```
