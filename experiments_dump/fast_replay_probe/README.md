# Fast Replay Probe

Goal: find or disprove a path to process one complete SLP replay's frame data in under one
second per worker. The visual-frame path is currently being tested as video first, with PNG
materialization deferred.

Constraints from the main task:

- Keep all scratch work in this folder.
- Do not edit `core`.
- Parser-only Slippi frame state is a timing control and is already sub-second.
- Visual frames remain the hard path; full PNG output was the original contract, but the current
  experiment writes video first and treats PNG extraction as a later stage.

Current sample:

- `../replays/realtimeTest.slp`
- Frames: `-123..2182`, 2,306 SLP frame entries.
- Playback JSON: `playback.realtime.full.json`.
- Node dependency: `@slippi/slippi-js@9.1.2` from `../package.json` / `../node_modules`.

Key artifacts:

- `findings.md`: measurements, external validation, and current recommendation.
- `frame-state-schema.md`: fixed-width `SLPFRAMESTATEv2` binary row layout.
- `extract-frame-state-binary.mjs`: sub-second Slippi state extractor control.
- `extract-frame-state-batch.mjs`: batch version of the fixed-width binary frame-state extractor.
- `bench-frame-state-process.mjs`: fresh-process benchmark that spawns one Node extraction per SLP.
- `verify-frame-state-summary.mjs`: threshold/completeness verifier for benchmark summaries.
- `ishiiruka-fast-png-frame-dump.patch`: experimental Dolphin/Ishiiruka patch for Slippi-frame-gated PNG dumping and PNG compression knobs.
- `ishiiruka-headless-egl-playback.patch`: experimental headless EGL Playback Dolphin patch.
- `render-ffv1-replay.sh`: video render wrapper with end-frame watcher and codec knobs.
- `render-ffv1-to-png.sh`: two-stage FFV1-to-full-PNG wrapper.
- `Dockerfile.ffv1-to-png`: derivative renderer image that adds FFmpeg and the FFV1 wrapper.
- `Dockerfile.headless-egl`: native headless EGL source-build image experiment.
- `runpod-headless-egl-test.py`: RunPod helper for a prebuilt headless image.
- `runpod-ffv1-png-manifest.json`: manifest copied from the native RunPod FFV1-to-PNG probe.
- `runpod-gpu-video-remote-summary.txt`: RTX PRO 6000 video-only probe summary.
- `runpod-gpu-video-pod.json`: pod allocation metadata for the GPU probe.
- `runpod-gpu-vulkan-retest-pod.json`: pod allocation metadata for the minimal Vulkan retest.
- `runs/runpod-gpu-vulkan-retest/`: copied logs from the minimal Vulkan retest.
- `runs/runpod-headless-egl-source-build-secure/`: RTX 4090 source-build logs, copied summaries,
  and cleanup proof. Large video files are intentionally excluded from the local copy.
- `runs/frame-state-batch-local-3x/batch-summary.json`: 63 in-process semantic frame-state
  extractions, max `0.036994s` per SLP.
- `runs/frame-state-batch-process-20x/process-bench-summary.json`: 20 fresh-process semantic
  frame-state extractions, p95 `0.127214s`, max `0.401800s`.
- `runs/frame-state-batch-slippi-js-corpus-v2-verified/batch-summary.json`: 60 diverse public
  Slippi sample SLPs in one warm process, zero failures, no skipped occupied frame records, p95
  `0.174650s`, max `0.342092s`.
- `runs/frame-state-process-slippi-js-corpus-v2-verified/process-bench-summary.json`: same corpus
  with one fresh Node process per SLP, zero failures, no skipped occupied frame records, p95
  `0.283442s`, max `0.704498s`.
- `runs/hf-bowser-25/replays/BOWSER/`: first 25 `BOWSER/` replays from the public Hugging Face
  `erickfm/slippi-public-dataset-v3.7` tournament corpus.
- `runs/frame-state-batch-hf-bowser-25-v2/batch-summary.json`: HF sample in one warm process,
  zero failures, no skipped occupied frame records, p95 `0.143579s`, max `0.256620s`.
- `runs/frame-state-process-hf-bowser-25-v2/process-bench-summary.json`: same HF sample with one
  fresh Node process per SLP, zero failures, no skipped occupied frame records, p95 `0.367061s`,
  max `0.397421s`.
