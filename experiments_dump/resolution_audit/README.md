# Smash recording resolution audit

## Result

Use **256x208 at 60 FPS** as the first training candidate. Both dimensions are divisible by 16,
the output contains 6.37x fewer pixels than the current recording, and the corresponding `/16`
latent grid is only `16x13 = 208` spatial positions.

Intermediate checks at `240x208` and `224x192` show increasingly soft character, nametag,
platform-edge, and HUD detail. `192x160` crosses the obvious failure boundary, and `128x112` is not
usable. Treat 256x208 as the practical visual floor pending a downstream model ablation, not as
proof that a trained model will tie the higher-resolution baseline.

## Source

- Hugging Face dataset: [`DhruvBhatia0/smash-battlefield-fox`](https://huggingface.co/datasets/DhruvBhatia0/smash-battlefield-fox)
- Pinned revision: `2c94351b82c2c65a31fb39fe52a34ff905b6abcf`
- Sample: `slp_with_video/1455/video.avi`
- Sample SHA-256: `b881fcf6c9dfda731c690e2fd404b01a300989209d0fc557d18b8465ecfe68b0`
- Source format: raw YUV420P AVI, 642x528, 60 FPS
- Sample duration: 14.9167 seconds
- Sample bytes: 454,607,918
- Actual decodable frames: 894. The AVI header reports 895, but `ffprobe -count_frames` finds
  894. Every output retains those 894 frames and reports `r_frame_rate=60/1`.

Five HF recordings across the corpus size distribution, including the largest recording, were
range-probed and all reported 642x528 YUV420P at 60 FPS.

## Ladder

All variants use aspect-preserving Lanczos scaling, minimal black padding to the requested
multiple-of-16 canvas, H.264/yuv420p at CRF 18, and the source timestamps without an FPS filter.

| Canvas | Pixels | Pixel reduction | `/16` grid | 14.9 s size | VMAF |
|---|---:|---:|---:|---:|---:|
| 512x416 | 212,992 | 1.59x | 32x26 = 832 | 4.22 MB | 96.21 |
| 384x320 | 122,880 | 2.76x | 24x20 = 480 | 2.80 MB | 92.07 |
| 320x256 | 81,920 | 4.14x | 20x16 = 320 | 2.00 MB | 86.65 |
| **256x208** | **53,248** | **6.37x** | **16x13 = 208** | **1.45 MB** | **78.22** |
| 240x208 | 49,920 | 6.79x | 15x13 = 195 | 1.35 MB | 75.09 |
| 224x192 | 43,008 | 7.88x | 14x12 = 168 | 1.20 MB | 71.54 |
| 192x160 | 30,720 | 11.03x | 12x10 = 120 | 0.95 MB | 61.53 |
| 128x112 | 14,336 | 23.64x | 8x7 = 56 | 0.50 MB | 30.10 |

VMAF was measured after removing each variant's padding and Lanczos-upscaling its content back to
642x528. It measures conventional perceptual similarity, not whether a world model can still infer
the correct game dynamics. The sharp score drop below 256x208 agrees with visual inspection.

The source aspect ratio is 1.2159, so a direct 256x192 resize would introduce substantial shape
distortion. The 256x208 output instead contains a 252x208 aspect-preserved image plus two black
columns on each side. This changes content aspect by only about 0.36% and keeps the final tensor
dimensions divisible by 16.

Large source and generated artifacts are ignored under `runs/hf-1455/`, including the full
resolution ladder, VMAF logs, source frame, and comparison sheet.
