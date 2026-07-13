# Frame Game-State Data Experiment

This folder is a first pass at a Slippi replay dataset pipeline.

The production Fox-versus-Captain-Falcon video corpus is documented in
[`frame-video-dataset.md`](frame-video-dataset.md). That document is the handoff
for data locations, schemas, frame alignment, integrity, and regeneration.

- `replays/realtimeTest.slp` is a public sample replay downloaded from the official `project-slippi/slippi-js` repository.
- `scripts/extract-slp.mjs` replays the `.slp` through `@slippi/slippi-js`, walking every frame in numeric order.
- `patches/ishiiruka-macos-png-frame-dump.patch` is the small source patch needed for macOS PNG frame dumping.
- `tools/patched-playback/Slippi Dolphin.app` is the compact local patched Playback Dolphin app bundle. It is intentionally gitignored.
- `game-frames/realtimeTest-capture-test/` contains the current aligned training image set: one PNG per Slippi replay frame for frames `-123..901`. It is generated and gitignored.
- `data/realtimeTest.capture-test.image-rows.jsonl` contains the extracted per-player state/input rows filtered to frames with images and populated with `screenshotPath`. It is generated and gitignored.

Important constraint: `.slp` files do not contain rendered images. To add per-frame screenshots, replay the same `.slp` in Slippi Playback Dolphin with a Melee ISO, dump frames, align Dolphin dump frame order with Playback Dolphin's `[CURRENT_FRAME]` log, and then join image paths to rows by replay frame.

Run state extraction again from this folder:

```bash
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm install
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm extract
```

Render with the compact patched Playback Dolphin bundle, align frames, and attach image paths:

```bash
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm render
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm align
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm attach:images
```

Only run `pnpm prepare:playback-build` when `source/Ishiiruka` has been cloned and rebuilt locally; the cleaned folder does not keep the full source tree.

Download replay batches and extract replay-level metadata:

```bash
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm ingest:slp -- --manifest download-manifests/slippi-js-samples.json --concurrency 4
```

The ingestion output is generated and gitignored:

- Downloaded replays: `replays/downloaded/`
- Per-replay metadata and aggregate index: `metadata/replays/`

Metadata includes stage/map, players, display names/connect codes when present, inferred winner/placements when final stocks are available, and replay-derived skill signals. True player rank/MMR is not normally present in `.slp` files, so `skillSignals.rank` is intentionally `null`.

## Frame Queue Processing

The corpus processing path is class-based and uses a single producer thread with `queue.Queue(maxsize=1000)` by default. The producer discovers `.slp` files and blocks on `put` when the queue is full. Consumer threads provision a runtime during initialization, then render jobs and save state rows, raw frame dumps, aligned frames, image-backed rows, and per-job results under `processed-frame-queues/`.

Safe planning run:

```bash
python3 scripts/process-frame-queue.py --runtime plan --consumers 2 --queue-size 1000 replays/downloaded
```

Local integration run with the patched macOS Playback Dolphin app:

```bash
python3 scripts/process-frame-queue.py --runtime local-macos --consumers 1 --start-frame -123 --end-frame 900 replays/realtimeTest.slp
```

RunPod is the first scalable runtime target. Docker is wired behind the same runtime interface, but local Docker is only useful after a Linux renderer image exists with `/opt/slippi-renderer/render-replay.sh`; the currently verified renderer is a patched macOS app. RunPod also needs that renderer image plus an explicit remote ISO strategy before real pod processing should be enabled.

RunPod planning run:

```bash
python3 scripts/process-frame-queue.py --runtime runpod --dry-run --plan-only --runpod-image slippi-renderer:runpod --consumers 2 replays/downloaded
```

Cleanup dry run:

```bash
python3 scripts/clean-up-runpod-cpu-pods.py
```

Delete this experiment's CPU worker Pods:

```bash
python3 scripts/clean-up-runpod-cpu-pods.py --confirm
```

Delete all CPU Pods in the account:

```bash
python3 scripts/clean-up-runpod-cpu-pods.py --all-cpu --confirm
```
