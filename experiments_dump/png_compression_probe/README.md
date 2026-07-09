# PNG Compression Probe

This experiment explores CPU-decodable ways to store Dolphin PNG frame dumps more compactly.

The production `core` package is treated as read-only. The probe works from an existing frame
directory and writes all derived archives, manifests, and reports under this folder.

## Quick Start

```bash
/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  experiments_dump/png_compression_probe/png_compression_probe.py \
  --frames-dir experiments_dump/fast_replay_probe/runs/local-full-baseline-20260709T025734Z/dolphin-user/Dump/Frames \
  --limit 1504 \
  --run-dir experiments_dump/png_compression_probe/runs/midrun-1504 \
  --verify
```

The script benchmarks:

- PNG files compressed as a tar stream with `zstd`.
- Full padded RGB frame streams compressed with `zstd`/`xz`.
- Inter-frame XOR and modulo-difference RGB streams.
- Sparse changed-tile streams.
- Lossless CPU video encoders available through local `ffmpeg`.

Each result reports output size, ratio against the original PNG bytes, encode time, and optional
pixel round-trip verification.

