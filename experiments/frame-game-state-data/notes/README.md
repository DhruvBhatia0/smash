# Frame Game-State Data Experiment

This folder is a first pass at a Slippi replay dataset pipeline.

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
