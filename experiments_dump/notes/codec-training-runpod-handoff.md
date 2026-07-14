# Codec training and streamed dataset handoff

Status: implementation and infrastructure validation completed on 2026-07-14.
The production GPU run was intentionally stopped before its first training step,
and its RunPod Pod and network volume were deleted at the user's request.

This note records the complete investigation, implementation, validation, and
operational outcome for training the video codec from the Fox–Captain Falcon
Battlefield corpus stored on Google Drive. The dataset itself is documented in
[`frame-video-dataset.md`](frame-video-dataset.md). The implementation was
developed in [PR #10](https://github.com/DhruvBhatia0/smash/pull/10).

## Executive summary

- The Google Drive objects are `tar.zst` batches, not ordinary independent
  `.gz` samples. They are sequential archives and should be transferred once,
  scanned once, and consumed sequentially.
- Directly mounting Google Drive or a cross-cloud GCS bucket on an
  unpredictably located RunPod GPU would add avoidable latency, egress, and
  failure modes. The validated design stages the compressed corpus once onto a
  RunPod network volume in the same data center as the GPU Pod.
- CPU workers decompress Zstandard and decode H.264 proactively. They spool
  complete MP4s ahead of demand, emit overlapping clips from a frame ring, and
  queue pinned batches ahead of the GPU. After startup, measured GPU data wait
  was approximately `0.05-0.09 ms`.
- Training uses eight-way data parallelism through PyTorch DDP, not tensor
  parallelism. Each H100 holds a complete model and processes a different local
  batch. The validated local batch is 18, for a global batch of 144.
- A compiled 10-step, 8xH100 smoke test passed at approximately 75 clips/s,
  71,068 MiB reported device memory per GPU, finite losses, and no loader stall.
- The full corpus contains 3,101,190 overlapping two-second clips after skips.
  The deterministic whole-archive split contains 2,922,684 training clips and
  178,506 evaluation clips.
- The production launch was stopped during startup on request. No production
  step or epoch completed and no checkpoint was uploaded. The transient RunPod
  Pod and 500 GB network volume were then deleted.

## Authoritative dataset and edge cases

Only this Drive prefix is authoritative:

```text
smash-drive:hal-fox-captain-falcon-battlefield/recordings-252x208-20fps-slippi-pts-v3/batches
```

The neighboring smoke, old-timeline, legacy, and orphan-quarantine folders are
not training data. The final source inventory is:

| Item | Value |
|---|---:|
| Indexed replay references | 10,327 |
| MP4 videos | 10,319 |
| Explicit `no_playable_frames` skips | 8 |
| Source video frames | 31,359,489 |
| Source video duration | 435 h 32 m 54.45 s |
| Committed objects | 111 archives + 111 manifests |
| Compressed bytes | 147,359,770,145 B / 137.239 GiB |

Every normal video is H.264 in MP4, `yuv420p`, 252x208, constant 20 FPS, and
contains one video stream with no audio. Training must use actual MP4 frame
counts. Replay-semantic duration and endpoint fields do not have the same
semantics as stored-video duration.

The edge cases are handled as follows:

- Eight `no_playable_frames` records contain no MP4. Manifest parsing excludes
  them from video and clip counts, and archive streaming simply sees no
  `video.mp4` member for them.
- One committed batch contains only a skipped record. This explains why there
  are 111 committed archive/manifest pairs but 110 usable video archives.
- Sample 10317 uses the documented `stable-render-prefix-v1` tail recovery and
  has a valid 2,414-frame MP4. The loader trusts its indexed MP4 frames and does
  not compare its endpoint with the original replay endpoint.
- Some retained SLPs contain concatenated games and have replay-normalization
  metadata. This matters for future state/action alignment, but not for
  codec-only pixel reconstruction.
- `firstSelectedSlpFrame` is usually `-39` but is not assumed by this loader.
  Codec training operates only on decoded video-frame order.
- The Drive OAuth warning about rclone's shared client being retired in 2026
  matters for a future restage. A private Google OAuth client should be used
  for the next large Drive transfer.

## Storage decision

The investigated options were:

1. Download and expand everything onto each GPU node. This no longer fits the
   intended node lifecycle and unnecessarily duplicates data.
2. Stream individual samples repeatedly from Google Drive. This is a poor fit
   because Drive is not a training object store and `tar.zst` is sequential;
   reopening an archive for each MP4 repeatedly transfers and decompresses its
   prefix.
3. Put the data in GCS and mount it through Cloud Storage FUSE. This can work
   when compute is in a known, colocated GCP region, but RunPod placement was
   not known before provisioning. Cross-cloud traffic would expose training to
   egress charges, higher latency, and an Internet dependency.
4. Stage the compressed archives once onto a RunPod network volume and attach
   it to a GPU Pod in the same RunPod data center. This was selected and
   validated.

At the time of the test, RunPod documented network volumes from 1 to 4,000 GB.
A 500 GB volume was enough for the 137.239 GiB compressed corpus, indexes,
model caches, logs, and approximately 6.8 GB per checkpoint. The volume was in
`EUR-IS-3`, colocated with the 8xH100 Pod.

The complete Drive stage copied all 222 files in about 21 minutes. A subsequent
`rclone check --size-only --one-way` reported 222 matching files and zero
differences.

## Indexing the sequential archives

[`core/training/codec/index.py`](../../core/training/codec/index.py) builds a
small sidecar next to each usable archive:

```text
batch-<key>.tar.zst.codec-index.json
```

Each sidecar records:

- schema and completion status;
- the archive byte size, used to reject a stale index; and
- the exact frame count of every MP4 in archive order.

The indexer streams Zstandard and tar sequentially, spools each MP4, and reads
container frame metadata without decoding video pixels. It writes a partial
file and atomically renames it only after a successful scan. It also requires
the number of indexed MP4s to match the non-skipped manifest count.

Measured indexing results:

- Four real 100-video archives on a four-CPU Daytona machine: 9.8 seconds.
- The full RunPod copy: 110 usable archives indexed with 32 processes in about
  84 seconds.

For a video with `N` frames, a 40-frame clip and 10-frame stride produce:

```text
max(0, 1 + floor((N - 40) / 10))
```

A real 1,855-frame source was checked explicitly. It produced 182 clips, and
consecutive clips had the expected 30-frame overlap.

## Proactive streaming loader

[`core/training/codec/data.py`](../../core/training/codec/data.py) treats the
manifest as the commit record. It accepts only positive-size archive/manifest
pairs whose manifest rows all have `status: complete`, and subtracts explicit
skips from the video count.

The runtime pipeline is:

```text
RunPod network volume
  -> sequential Zstandard decompression
  -> sequential tar scan
  -> complete MP4 spooled by CPU producer
  -> PyAV H.264 decode into a frame ring
  -> 40-frame clip every 10 frames
  -> rank-local pinned batch queue
  -> non-blocking H100 transfer
```

Important implementation details:

- Each DataLoader worker has a producer thread that keeps four complete MP4s
  ahead by default.
- `tempfile.SpooledTemporaryFile` keeps small videos in memory and spills large
  ones to the configured fast spool location. RunPod used its 704 GB `/dev/shm`.
- A bounded 40-frame deque emits the first two-second clip and then another
  every 10 decoded frames, so overlapping clips do not trigger repeated video
  decoding.
- Six DataLoader workers run per DDP rank, with PyTorch prefetch factor four.
- A rank-process producer collates and pins four complete batches ahead of the
  training loop.
- Queues are bounded, so prefetching uses available RAM without growing without
  limit or loading the whole corpus.
- Producer exceptions are forwarded to the consumer instead of silently
  hanging a training rank.
- Mean and maximum time spent waiting for the next batch are reduced across
  ranks, printed, and logged to W&B.

Measured loader performance:

| Environment | Result |
|---|---:|
| Daytona, 4 CPUs | 24.06 clips/s and 144.32 MiB/s |
| RunPod host, 12 workers, concurrent Drive stage | 82.33 clips/s and 493.88 MiB/s |
| 8xH100 compiled training after startup | `0.05-0.09 ms` mean data wait |

The loader therefore stayed ahead of the measured model throughput. The first
batch took several seconds because worker processes, archive streams, and video
queues had to start. That one-time startup cost should not be interpreted as a
steady-state data stall.

### Epoch semantics caveat

`steps_per_epoch` is exact as a global step budget:

```text
floor(training clips / global batch)
```

Training ranks and workers own archive subsets and cycle their private subsets
indefinitely to prevent DDP deadlock when archive sizes differ. Consequently,
an epoch is not guaranteed to visit every individual clip exactly once: a
worker with fewer assigned clips can begin another cycle while a worker with
more clips has not finished its first cycle. Archive order is reshuffled
between cycles, so coverage changes over time, but exact once-per-epoch
semantics would require clip-count-balanced worker ranges or an explicit
distributed epoch plan. This is the main known data-scheduling follow-up.

## Clip schedule and split

The requested clip configuration is implemented directly:

| Setting | Value |
|---|---:|
| Resolution | 252x208 |
| Source frame rate | 20 FPS |
| Frames per clip | 40 |
| Clip duration | 2 seconds |
| Stride | 10 frames / 0.5 seconds |
| Overlap | 30 frames / 1.5 seconds |

The split is deterministic and operates on whole archives, preventing clips
from one replay/archive from leaking across train and evaluation. For the 110
usable archives, rounding the requested 95:5 split produced:

| Split | Archives | Clips |
|---|---:|---:|
| Train | 104 | 2,922,684 |
| Evaluation | 6 | 178,506 |
| Total | 110 | 3,101,190 |

At local batch 18 on eight ranks, the global batch is 144. The resulting
schedule is 20,296 optimizer steps per epoch and 142,072 steps over seven
epochs.

## Data parallelism, not tensor parallelism

The tested launch uses:

```text
torchrun --nproc-per-node=8
```

PyTorch DistributedDataParallel creates one complete model replica per H100.
Each rank consumes different archive/clip work, computes gradients for its
local batch, and synchronizes gradients over NVLink/NCCL. This is data
parallelism.

Tensor parallelism was unnecessary because one complete codec model and local
batch fit on an 80 GB H100. Tensor parallelism would shard individual model
operations across GPUs, add communication and implementation complexity, and
would not address storage or decoding. DDP is the cleanest scaling mode for
this model and dataset.

## MIRA codec alignment

The implementation was compared with
[mira-wm/mira](https://github.com/mira-wm/mira). The intentional differences
are the dataset's native 252x208 resolution and this repository's DINOv3-S+
encoder. The remaining codec recipe was aligned as follows.

### Model

- Frozen `dinov3_vits16plus` encoder.
- Intermediate DINO layers 1, 4, 7, 9, 10, and 11.
- Selected features aggregate as `mean(features) + features[-1]`.
- Decoder width 1,152, depth 28, 16 attention heads, spatial patch 16, and
  temporal patch 2.
- Spatial bidirectional attention, causal temporal attention, QK normalization,
  gated MLPs, residual scaling, RoPE, MIRA-style initialization, and decoder
  activation checkpointing.
- Native 252x208 output is preserved. Input is padded internally to 256x224
  only where the stride-32 encoder/decoder path requires it, then cropped back.

### Reconstruction objective

[`core/training/codec/loss.py`](../../core/training/codec/loss.py) implements:

```text
L1 + adaptive_weight(LPIPS) * VGG-LPIPS
   + adaptive_weight(DINO) * normalized DINO feature consistency
```

- L1 is evaluated on every reconstructed pixel.
- LPIPS uses the frozen VGG network.
- LPIPS and DINO each sample 25% of video frames independently.
- DINO consistency averages MSE across normalized selected-layer features.
- Adaptive weights compare each auxiliary loss's gradient norm with the L1
  gradient norm at the decoder's final projection layer, then detach and clamp
  the ratio.
- Frozen DINO parameters still allow gradients from reconstructed features to
  flow back to reconstructed pixels.
- Target DINO features already computed by the codec encoder are reused, so the
  target does not require a redundant second DINO pass.

### Optimization

- AdamW at `1e-4`, betas `(0.9, 0.95)`, weight decay `0.1`.
- Linear warmup for 1,000 steps, then cosine decay to `1e-6`.
- Bias-corrected model EMA with decay `0.9999`.
- Latent mean and standard-deviation EMA with decay `0.99`, synchronized across
  ranks.
- EMA model weights are used for evaluation and saved for downstream consumers.

The 1,000 warmup steps are only about 0.7% of the complete 142,072-step
schedule. Warmup does not discard samples; it performs optimizer updates at a
smaller learning rate to protect the large randomly initialized decoder during
its least stable phase.

## Evaluation, checkpoints, W&B, and Hugging Face

- Rank zero evaluates 1,024 clips at each epoch boundary.
- Each evaluation writes and logs a two-second side-by-side
  `target | reconstruction` MP4 to W&B under `eval/reconstruction`.
- Train/eval loss terms, adaptive weights, learning rate, latent statistics,
  epoch progress, throughput, and loader-wait metrics are logged.
- A checkpoint is written at every epoch boundary. Smoke checkpoints were
  approximately 6.8 GB each.
- Checkpoints contain EMA model weights, optimizer and scheduler states, step,
  epoch, steps per epoch, complete config, and latent normalization statistics.
- When `--hf-repo` is set, every checkpoint upload is queued asynchronously so
  a multi-gigabyte upload does not block continued training. Shutdown waits for
  outstanding uploads.

The stopped production launch created these tracking locations:

- W&B: <https://wandb.ai/dhruvbhatia0/smash-codec/runs/4vrrjw3e>
- Hugging Face: <https://huggingface.co/DhruvBhatia0/smash-codec>

The run was stopped before its first production step and before any production
checkpoint upload. The W&B run is therefore only an aborted startup record,
and the Hugging Face repository did not receive an epoch checkpoint.

## Daytona validation

A CPU-only Daytona sandbox was used before spending H100 time:

| Field | Value |
|---|---|
| Name | `smash-codec-loader-benchmark` |
| Sandbox ID | `5b43f55b-ed19-4fb5-9b4e-c18981e34c91` |
| Storage | 100 GB |
| GPU count | 0 |
| Final state | retained/running when this note was written |

Validation included:

- finite objective values and nonzero decoder/input gradients;
- adaptive loss weights;
- target DINO feature reuse;
- scheduler endpoints;
- model and latent EMA behavior;
- a small decoder forward/backward pass;
- real VGG-LPIPS inference;
- real six-layer DINOv3-S+ feature extraction with gradients to input;
- SHA-256-verified DINO and VGG weight caches;
- exact clip overlap on real video; and
- the four-CPU proactive loader benchmark reported above.

The user explicitly requested that this Daytona machine not be shut down. It
was not deleted during RunPod cleanup.

## RunPod validation

The transient GPU environment was:

| Field | Value |
|---|---|
| Pod ID | `aqxvxlx5jhd2xk` |
| GPU | 8x NVIDIA H100 80 GB HBM3/SXM |
| Data center | `EUR-IS-3` |
| CPU/RAM | 160 vCPUs / approximately 1.5 TB RAM |
| Shared memory | 704 GB |
| Network volume | 500 GB, ID `2wpr0tl8dn` |
| Image | `runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404` |
| PyTorch | 2.9.1 + CUDA 12.8 |
| Price observed | $23.92/hour total |
| Final state | Pod and network volume deleted |

The machine had full NVLink connectivity between all eight H100s. The full
Drive copy, indexes, source archives, DINO/VGG caches, compile cache, logs, and
smoke checkpoints lived on the deleted network volume.

### Batch-size probes

| Local batch / GPU | Mode | Observed peak | Result |
|---:|---|---:|---|
| 4 | compiled | 26,486 MiB | passed 10 steps; roughly 42-50 clips/s |
| 12 | eager | 53,168 MiB | passed; approximately 33 clips/s |
| 16 | eager | 65,914 MiB | passed |
| 18 | eager | 72,276 MiB | passed |
| 18 | compiled | 71,068 MiB | passed 10 steps; approximately 75 clips/s |

Batch 20 was not attempted after later batch-16 phases revealed that its likely
peak would approach the 80 GB device limit. Batch 18 preserved about 10.6 GiB
of reported device-memory headroom in compiled execution and was selected as
the largest sensible unattended configuration.

The batch-18 compiled smoke took about 7 minutes 40 seconds to build/load its
first graph. After compilation, steps 3-10 sustained approximately 75 clips/s.
The ten losses were finite and ended at `loss_total=1.59443`. Steady loader wait
was approximately `0.05-0.08 ms`; the first batch's startup wait was about 3
seconds. The smoke also produced a valid 740 KB reconstruction MP4 and 6.8 GB
checkpoint.

At 75 clips/s, 142,072 steps with global batch 144 represent approximately
75.8 hours of pure optimizer-step time. At the observed $23.92/hour Pod rate,
that is roughly $1,814 before epoch evaluation, compile, checkpoint, upload, or
throughput variance. This is an estimate, not a completed-run measurement.

## Dependency and operational learnings

- The RunPod image already contained PyTorch 2.9.1 + CUDA 12.8. An unconstrained
  torchvision resolution attempted to pull an incompatible/newer Torch/CUDA
  stack. Pinning `torchvision==0.24.1` retained the correct PyTorch pairing.
- DINO's Torch Hub repository and pretrained weights must both be cached.
  `SMASH_DINO_WEIGHTS` accepts an exact file, while
  `SMASH_DINO_WEIGHTS_DIR` searches a directory and rejects ambiguous matches.
- The cached DINO file used was
  `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`; VGG used
  `vgg16-397923af.pth`.
- `PYTORCH_CUDA_ALLOC_CONF` is deprecated in this runtime. New launches should
  use `PYTORCH_ALLOC_CONF` and explicitly unset the old inherited variable.
- The remote code was copied without `.git`, so W&B warned that its configured
  Git root did not exist. That warning was benign and did not affect logging.
- Compile caches are shape-sensitive. Changing from batch 4 to batch 18 caused
  another expensive first-step compile, but that cache could then be reused by
  a production launch with the same model and input shape.
- A retained network volume makes Pod replacement cheap, but deleting that
  volume also deletes staged data and compile/model caches. A future launch now
  requires recreating the volume, restaging 137.239 GiB, rebuilding the 110
  frame indexes, and restoring model weights.
- The network volume should be created in a data center that currently has the
  desired GPU capacity; RunPod network volumes are data-center-local.

## Local visual samples

Twenty source MP4s were extracted from one committed archive, copied to the
user's laptop, and verified with ffprobe. All 20 report H.264, 252x208, and
20/1 FPS. They occupy about 229 MB at:

```text
/Users/dhruv/Downloads/smash-dataset-mp4-samples
```

The copied sample IDs are 6533, 6549, 6551, 6574, 6632, 6644, 6654, 6664,
6699, 6710, 6723, 6800, 6886, 6894, 6978, 7036, 7055, 7071, 7100, and 7130.
These local viewing files are deliberately not committed to Git.

## Code delivered

PR #10 changes:

- [`core/training/codec/codec.py`](../../core/training/codec/codec.py): MIRA-size
  decoder, initialization, attention/MLP details, activation checkpointing, and
  reusable DINO target features.
- [`core/training/codec/dinov3_wrapper.py`](../../core/training/codec/dinov3_wrapper.py):
  selected intermediate layers and portable weight discovery.
- [`core/training/codec/loss.py`](../../core/training/codec/loss.py): MIRA
  reconstruction objective.
- [`core/training/codec/optimization.py`](../../core/training/codec/optimization.py):
  warmup/cosine scheduling plus model and scalar EMA utilities.
- [`core/training/codec/index.py`](../../core/training/codec/index.py): fast
  multiprocessing frame-index generation.
- [`core/training/codec/data.py`](../../core/training/codec/data.py): committed
  archive discovery, exact overlapping clip counts, proactive MP4 production,
  and frame-ring clip emission.
- [`core/training/codec/train.py`](../../core/training/codec/train.py): eight-way
  DDP training, pinned batch prefetch, exact step schedule, epoch evaluation and
  checkpointing, W&B video/metric logging, and asynchronous Hugging Face uploads.
- [`requirements.txt`](../../requirements.txt): LPIPS support and the compatible
  torchvision pin.

Local validation before merge included Ruff checks on the changed codec files,
Ruff format checks, Python bytecode compilation, and `git diff --check`. The
repository has no required PR status checks configured.

## Recreating the validated launch

After recreating and filling a RunPod network volume, generating all codec
indexes, caching DINO/VGG, and exporting W&B/Hugging Face credentials, the
validated training shape is:

```bash
python -m torch.distributed.run \
  --nproc-per-node=8 \
  --master-port=29502 \
  -m core.training.codec.train \
  /workspace/codec-data/batches \
  --batch-size 18 \
  --epochs 7 \
  --checkpoint-dir /workspace/codec-checkpoints/codec-7ep-b18 \
  --wandb-project smash-codec \
  --wandb-name codec-7ep-b18-8xh100 \
  --hf-repo DhruvBhatia0/smash-codec \
  --hf-private
```

Before an unattended relaunch, decide whether step-budget epoch semantics are
acceptable or whether the worker assignment should be changed to exact
clip-balanced epoch plans. Everything else in the launch path was exercised on
the target 8xH100 hardware.
