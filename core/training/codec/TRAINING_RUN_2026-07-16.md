# Codec full-dataset training run — 2026-07-16

This note records how the first full codec run was staged, what actually ran, and what we learned. The corresponding experiment is [W&B run `0i8htlom`](https://wandb.ai/dhruvbhatia0/smash-codec/runs/0i8htlom). The final model is [Hugging Face checkpoint `step-0016182.pt`](https://huggingface.co/DhruvBhatia0/smash-codec/blob/main/checkpoints/step-0016182.pt).

## Dataset and access

The source dataset lives in Google Drive as 448 `batch-<id>.tar.zst` archives and 448 matching `batch-<id>.manifest.jsonl` files:

- [Finished batch folder](https://drive.google.com/drive/folders/1hnHMZtNVpZVLKiywWo9-9Ev_lxqigEha)
- [Dataset root](https://drive.google.com/drive/folders/11oL6WTBLU0w13GkgGy7slhyFijdmvfTR)

```text
smash-drive:hal-fox-captain-falcon-battlefield/
recordings-642x528-20fps-slippi-pts-v4/batches
```

The complete compressed dataset is 896 files and 561,825,182,493 bytes. It contains 10,327 rendered replays at 642×528 and 20 FPS. Each manifest is the commit record for its archive and preserves the replay's source reference and checksum.

The RunPod host downloaded the committed archive/manifest pairs directly with `rclone copy`. The loader itself does **not** read from Google Drive: training reads the local copy on the pod. OAuth material was removed after the download and was never committed. An 800 GB pod volume was sufficient for the roughly 562 GB compressed dataset. We never extracted the entire dataset, which would have exceeded the available storage.

After download, 64 CPU processes indexed the MP4 frame counts from container metadata. This produced the clip counts needed to size the run without decoding every frame in advance.

## How streaming worked

Every loader worker owns a subset of the training archives. For each archive it:

1. Streams Zstandard decompression into a sequential tar reader.
2. Spools one MP4 member at a time instead of unpacking the shard.
3. Decodes with PyAV/FFmpeg.
4. AREA-downsamples the complete frame to 252×208; it never crops.
5. Emits contiguous 40-frame windows with a 40-frame stride.

PyTorch `DataLoader` provides the bounded queue: workers keep prefetched batches ready, the trainer consumes them, and consumed batches are released. This is the standard producer/consumer pattern we wanted without maintaining a custom ring-buffer implementation.

The archive containing the final dataset sample, `batch-ea63cff5629dc0ae4fb8.tar.zst`, was held out in full. Rank zero decoded and cached 1,024 evaluation clips once, using about 6 GiB of CPU RAM. Evaluation therefore did not repeatedly decompress its archive.

## Run configuration

| Setting | Value |
| --- | ---: |
| GPUs | 8× H100 80 GB |
| Parallelism | 8-way DDP |
| Batch | 18 per GPU / 144 global |
| Resolution | 252×208 |
| Clip | 40 frames at 20 FPS / 2 seconds |
| Stride | 40 frames / 2 seconds / no overlap |
| Training archives | 447 |
| Indexed training clips | 776,847 |
| Evaluation archive | final archive, held out completely |
| Epochs | 3 |
| Steps per epoch | 5,394 |
| Total steps | 16,182 |
| Evaluation | every 1,000 steps, roughly every 32 minutes |
| Checkpoints | steps 5,394, 10,788, and 16,182 |
| Precision | BF16 autocast |
| Decoder memory strategy | activation checkpointing for all 28 blocks |
| Input workers | 6 per rank with bounded PyTorch prefetch |

The model used the existing reconstruction objective: pixel L1, VGG/LPIPS perceptual loss, and normalized DINO feature consistency. VGG and DINO remained frozen; gradients flowed through them into the codec. The training `*_auto_w` metrics are gradient-balancing multipliers, not additional losses.

The step budget uses `training_clips // global_batch`, leaving a remainder of 111 indexed clips per nominal epoch. More importantly, the input is an iterable, worker-sharded stream that cycles to prevent uneven archives from deadlocking DDP. All 447 training archives were in scope, but an “epoch” is a global step budget rather than a persisted guarantee that every individual clip appeared exactly once. Exact clip-level coverage would require explicit sample IDs and resumable sampler state.

## Outcome

The run completed normally at step 16,182 and exited with status zero. The final checkpoint upload and W&B synchronization both completed before the pod was terminated.

- End-to-end training and evaluation took about nine hours.
- Steady throughput was approximately 75.5 clips/second globally.
- Final reported mean data wait was 0.17 ms; maximum was 1.07 ms.
- All eight GPUs remained at 100% utilization during steady training.
- GPU memory settled around 73–75 GB per device.
- The pod cost was approximately $230, including setup, compilation, evaluation, checkpoint uploads, and teardown time.

The final evaluation summary was:

| Metric | Value |
| --- | ---: |
| Total reconstruction loss | 0.19162 |
| MAE | 0.05378 |
| LPIPS | 0.13672 |
| DINO consistency | 0.00025 |

All evaluation curves decreased over the run. The data path was not a meaningful bottleneck; the one multi-second wait occurred during initial worker startup, after which waits were negligible compared with the roughly 1.9-second training step.

## Inference check

The final EMA checkpoint was loaded locally on CPU and tested on five exact training windows from five different non-evaluation replays. The windows were aligned to the 40-frame training grid and sampled near the start, quarter, middle, three-quarter, and end of their source replays.

Each CPU reconstruction took roughly 13–14 seconds. All outputs were verified as 40 frames, 20 FPS, two seconds, and 252×208 per panel. The codec preserved stage geometry, HUD state, characters, and broad motion. The visible weakness was softness and some ghosting around fast-moving characters, while static scene elements were reconstructed more cleanly.

## Main lessons

1. **Use the framework's queue.** PyTorch's worker prefetch kept the GPUs fed without custom queue threads. Simpler code was also faster and easier to reason about.
2. **Never unpack the full corpus.** Streaming one compressed shard and one MP4 at a time made a dataset larger than local capacity trainable on a fixed-volume pod.
3. **Resolution dominates cost.** Original-resolution training was dramatically less efficient. Making resolution configurable and downsampling in the loader let the model stay resolution-agnostic while retaining the practical 252×208 default.
4. **Activation checkpointing bought useful batch capacity.** Recomputing decoder blocks during backward allowed a local batch of 18 within 80 GB. We did not run a controlled throughput comparison without checkpointing, so this should be treated as a capacity result rather than proof that recomputation itself was faster.
5. **The checkpoint is inference-complete, not resume-complete.** It contains the EMA codec model, frozen DINO tensors, optimizer, scheduler, configuration, and latent statistics. An exact resume would additionally need the live non-EMA model, EMA accumulator/count, RNG state, and dataloader/sampler position. The current constructor also bootstraps DINO before loading the included DINO state.
6. **Export a smaller inference artifact next.** The 7.28 GB training checkpoint includes Adam state that inference does not need. A model-only `safetensors` export plus a direct loader would be smaller, safer, and easier to deploy.
