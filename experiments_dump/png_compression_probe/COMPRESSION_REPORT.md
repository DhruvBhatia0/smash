# PNG Frame Compression Report

## Scope

I read the production flow in `core/data` but did not edit it. All code and generated artifacts for this pass live in `experiments_dump/png_compression_probe`.

The core pipeline currently seeds SLPs to Hugging Face, uses CPU RunPod workers to render `framedump_*.png`, and uploads those frames under `slp_with_frame/<sample id>/`. That makes compression a clean post-render storage format question: can we store the same pixels more compactly and recover them with CPU-only work during training?

## Online Validation

- The W3C PNG spec confirms PNG is a lossless static/animated raster format and that APNG stores later frames as PNG-like `fdAT` frame data, not video-style motion prediction: <https://www.w3.org/TR/png-3/>.
- FFmpeg documents `libx264rgb` as the RGB-input version of `libx264`; `libx264` supports lossless mode, B-frames, references, motion search, and adaptive transforms: <https://ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb>.
- RFC 9043 defines FFV1 as a lossless intra-frame video format, which explains why it is safer/preservation-oriented but weaker for this temporal dataset: <https://datatracker.ietf.org/doc/rfc9043/>.
- zstd is a fast lossless compressor with streaming APIs, but it only sees cross-frame redundancy if we expose pixels rather than already-deflated PNG byte streams: <https://facebook.github.io/zstd/zstd_manual.html>.
- RFC 3284/VCDIFF validates the general delta-compression framing idea, including portable compact deltas and CPU-efficient decoding, but the local data did not have enough exact unchanged pixels for simple deltas to win: <https://datatracker.ietf.org/doc/html/rfc3284>.
- x265 documentation confirms HEVC lossless still uses inter/intra prediction with lossless residual coding, so I tested it too: <https://x265.readthedocs.io/en/stable/lossless.html>.

## Corpus Facts

The most important local sample is the midpoint-sized slice the user called out:

- Frames: 1,504
- PNG bytes: 496,412,608 bytes, 473.42 MiB
- Canvas used for experiments: 460x380 RGB
- Raw padded RGB: 788,697,600 bytes, 752.16 MiB
- Source image modes: RGBA, but alpha sampled as fully opaque
- Dimensions: 1,502 frames at 460x380 and 2 frames at 460x344, so archives need a manifest/crop table
- Exact padded duplicate frames: 1
- Pixel change rate vs previous frame: mean 91.2%, median 94.0%

That last point is the main surprise. The frames feel visually repetitive, but exact same-position pixels are not the dominant redundancy. Camera movement, animations, antialiasing, and particles make most pixels differ at the same coordinate. This is why simple "store changed pixels" codecs lose, while block-motion video wins.

## Experiment Harness

Code:

- `experiments_dump/png_compression_probe/png_compression_probe.py`
- `experiments_dump/png_compression_probe/README.md`

Key implementation details:

- Reads PNGs with Pillow, drops alpha only after verifying it is opaque, pads to a fixed 460x380 RGB canvas, and records original frame sizes in JSON manifests.
- Verifies losslessness by hashing reconstructed RGB pixels with frame number and original width/height included.
- For video formats, pipes padded `rgb24` into FFmpeg and decodes back to `rgb24` for verification.
- Keeps manifests beside each archive so a training loader can recover frame numbers and crop dimensions.

## Main Results: 1,504 Frames

| Rank | Strategy | Output | Ratio vs PNG | Encode | Verified |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `x264rgb_qp0_placebo` | 301.19 MiB | 1.5718x | 261.8 s | yes |
| 2 | `x264rgb_qp0_veryslow` | 301.58 MiB | 1.5698x | 83.3 s | yes |
| 3 | `ffv1_level3` | 376.40 MiB | 1.2577x | 11.6 s | yes |
| 4 | `sub8_zstd19_long` | 435.33 MiB | 1.0875x | 205.8 s | yes |
| 5 | `png_tar_zstd19_long` | 472.77 MiB | 1.0014x | 94.1 s | n/a |

The best absolute ratio was `x264rgb_qp0_placebo`, but it saved only 0.39 MiB over `veryslow` while taking 178.6 extra seconds. I would not use placebo by default.

## Full Available Local Corpus

I also ran the recommended practical strategy on the full local frame folder:

- Frames: 4,118
- PNG bytes: 784,474,405 bytes, 748.13 MiB
- Padded RGB: 2,159,479,200 bytes, 2,059.44 MiB
- `x264rgb_qp0_veryslow`: 494,306,935 bytes, 471.41 MiB
- Ratio: 1.5870x
- Encode: 123.1 s
- Decode/verify: 16.3 s
- Verified: yes

## What Worked

The best practical storage format is one Matroska file per replay using lossless H.264 RGB:

```bash
ffmpeg -f rawvideo -pix_fmt rgb24 -s 460x380 -r 60 -i - \
  -c:v libx264rgb -qp 0 -preset veryslow -pix_fmt rgb24 \
  -x264-params keyint=9999:min-keyint=9999:scenecut=0 \
  frames.mkv
```

Store it with a small `frames.manifest.json`:

- original PNG filenames / frame numbers
- original width and height for crop recovery
- padded canvas width/height
- RGB hash or per-frame hashes if desired
- codec command and version

At training time, CPU workers can decode with FFmpeg into raw `rgb24` tensors directly:

```bash
ffmpeg -v error -i frames.mkv -f rawvideo -pix_fmt rgb24 -
```

If actual PNG files are needed, decode frames and crop according to the manifest, then encode PNGs on CPU. For training, I would skip re-PNGing and feed raw decoded RGB arrays.

## What Did Not Work

- Compressing PNGs as-is barely moved the needle. The PNG/DEFLATE layer hides the pixel-level temporal redundancy; tar+zstd was only 1.0014x on 1,504 frames.
- Simple raw RGB plus zstd/xz helped a little on small samples, but did not beat video.
- XOR deltas were actively bad on the 300-frame pass. XOR destroys byte locality when colors change slightly.
- Modulo subtraction deltas were correct and clean, but only 1.0875x on 1,504 frames.
- Exact changed tiles and changed row spans did not work well because median exact changed-pixel rate was 94.0%.
- A custom global-motion residual helped compared with simple deltas on small samples, but only reached 1.25x on 300 frames. One global shift cannot model players, camera zoom, particles, HUD, and stage motion together.
- FFV1 is nice and very fast, but it is essentially intra-frame for our purpose and reached only 1.26x.
- VP9 lossless was verified but worse than x264rgb on 300 frames: 1.61x vs 1.72x, and 217 s vs 13-17 s.
- x265 lossless GBR was verified but slightly larger and much slower than x264rgb on 300 frames: 1.72x at 179 s vs x264rgb 1.72x at 17-45 s.

## Recommendation

Use `libx264rgb -qp 0 -preset veryslow` in MKV plus a manifest. It is the best balance I found:

- Lossless RGB pixels, verified by decode/hash
- CPU-only encode/decode with commodity FFmpeg
- Exploits block motion and inter-frame prediction that the custom exact-delta attempts could not capture
- About 37% storage reduction on the full local corpus
- Much faster and nearly the same size as placebo

The loader should decode to raw RGB tensors, not reconstruct PNG files unless some downstream tool truly requires PNGs. That avoids wasting CPU on PNG encoding during training.

## Future Bets

If we can tolerate non-bit-exact images, a visually lossless or ML-tolerance-tested codec setting would probably beat this by a lot. For strict lossless, the next creative direction would be block-level learned/background-aware prediction before entropy coding, but at that point we are rebuilding pieces of a video codec. The experiments strongly suggest the off-the-shelf video codec is already capturing the useful structure in these Dolphin frame dumps.

