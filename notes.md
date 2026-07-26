# Full d-transformer run notes

Updated: 2026-07-16 America/Los_Angeles

## Objective

Train the MIRA-aligned Smash diffusion transformer for exactly four epochs on the complete committed
Drive dataset using one 8×H100 RunPod node. Use a bounded PyTorch `DataLoader` queue, preserve all
three ordered 60 Hz controller states per 20 FPS transition, keep W&B concise, and verify the final
recipe before committing the expensive run.

## Confirmed before launch

- Source: 448 committed `tar.zst` archives, 10,327 replays, 561.8 GB compressed.
- Video is 20 FPS; SLP inputs are contiguous 60 Hz; every video transition maps to three ordered
  inputs per player.
- Clip shapes: video `(40, 3, 208, 252)`, actions `(39, 3, 2, 56)`, normalized codec latent
  `(20, 32, 7, 8)`.
- Final one-sample train=eval latent test: fixed loss `1.97379 -> 0.06297` in 1,000 steps; six-step
  same-clip rollout MSE `0.00487`.
- Final small held-out recipe uses full-width per-player action pooling. Ordered 60 Hz controls
  improved autoregressive prediction versus last-only and no-action controls.
- RunPod A4000 ablation pod was deleted; no live GPU remained at the start of this task.

## Required launch recipe

- Exactly 4 epochs (increased from 2 by the user immediately before launch).
- 8×H100 node.
- Frozen final codec checkpoint `step-0016182.pt`.
- Full dataset local to the pod; never unpack the corpus.
- Precompute resumable, sharded frozen-codec latents once so the codec/video decoder is not repeated
  for both transformer epochs.
- PyTorch worker queue with bounded prefetch for source materialization and latent training.
- MIRA 1B defaults: width 2048, 16 layers, 16 Q heads, 4 KV heads, temporal attention every 4,
  clean-past control run, ordered 60 Hz actions, no action dropout.
- Empirically selected local batch 16 / global batch 128. Keep the released optimizer defaults;
  batch size changes throughput, not the flow-matching objective.
- W&B essentials only: train/EMA eval loss, LR, grad norm, examples/epoch, throughput/data wait,
  action zeroing delta, and short/final rollout MSE. Rely on W&B system telemetry for GPU memory.

## Work log

- Created this continuity note before implementation/provisioning.
- Audited the proven codec input pipeline: rank-then-worker archive sharding, sequential `tar.zst`
  reads, bounded pinned `DataLoader` prefetch, and nonblocking GPU copies. The transformer pipeline
  will retain those properties and cache each frozen-codec result exactly once.
- Provisioned RunPod pod `lu5b5srnu3rnrm`: 8x H100 80 GB HBM3, Secure Cloud `AP-IN-1`, 224 vCPU,
  2,013 GB RAM, 900 GB persistent pod volume, observed GPU price `$23.92/hour` total. The known
  `EUR-IS-3` pool was unavailable; its unused temporary network volume was immediately deleted.
- Started the complete Drive copy with 12 parallel transfers and four streams per large file;
  observed roughly 450 MB/s.
- Added resumable per-archive BF16 latent shards, deterministic exact-coverage rank/worker plans,
  bounded prefetch, two-epoch DDP training, epoch checkpoints, fixed EMA evaluation, and a dedicated
  minimal-metric W&B project. Static checks, a two-rank queue coverage test, and a synthetic
  flow-matching backward pass passed on the node.
- Real 1.207B H100 batch sweep (optimizer + EMA included): batch 14 DDP reached 271.1 global
  clips/s at 69.5 GiB; batch 16 reached 277.9 clips/s at 75.2 GiB; batch 20 with activation
  checkpointing slowed to 224.1 clips/s at 33.9 GiB. Selected batch 16 as the fastest fixed-shape
  run with usable headroom.
- Full copy completed: 448 archives + 448 manifests, 523.24 GiB in 13m38s. All 448 codec indexes
  completed with 96 CPU workers.
- A full-size one-sample smoke reached raw fixed loss `9.48 -> 1.06` by step 300 with finite,
  nonzero action-encoder gradients. The user explicitly waived the remaining overfit/decoded QA to
  start the full run sooner; the smoke was stopped and its output will not be used for training.
- Initial materialization showed the encoder was CPU-starved. A direct frozen-encoder sweep reached
  about 158 clips/s/GPU at every tested batch from 18 through 96, so larger batches were not useful.
  Relaunched resumably with batch 36, 24 decode workers/rank (192 total), and bounded prefetch 2;
  all eight GPUs then reached 96-100% during encoder bursts and sustained throughput improved.
- Materialization completed successfully: 3,301 shards contain 776,847 training clips and 2,093
  evaluation clips (778,940 total), with latent shape `(20, 32, 7, 8)` and action shape
  `(39, 3, 2, 56)`. A cached-shard audit found finite analog values in `[-1, 1]` and strictly binary
  button channels.
- The user stopped the launch before training began to recheck the Smash-specific input path. The
  old guarded handoff was killed and `/workspace/logs/full-training.log` remains empty.
- Rejected the proposed MIRA keyboard/mouse encoder. Smash keeps its proven full-width controller
  adapter: all eight analog fields and both Slippi button masks, three ordered 60 Hz microsteps,
  separate player identity, learned temporal pooling, shared player projection, and mean combine.
  The beginning-of-sequence action now follows MIRA's wrapper placement before per-player
  projection/combination. Action dropout and all related parameters/configuration were removed.
- Audited the binary parser against current official `project-slippi/slippi-js` source commit
  `626f8fb0dfa08133f3244d58bebe7002bddd44bf`: the seven float offsets, uint32 processed buttons,
  uint16 physical buttons, and signed int8 raw joystick X all match. Corrected one stale experiment
  document that incorrectly illustrated neutral raw joystick X as 128 instead of 0; no cache data
  needed repair.
- Synced the corrected transformer to the pod. Compile and Ruff checks pass. A real cached-batch
  forward/backward smoke is finite, preserves `(batch, 20, width)` action conditioning, changes
  under microstep reversal/zeroing, contains no dropout parameters, and receives nonzero action
  gradients after the zero-initialized AdaLN modulation unlocks on the second optimizer step.
- At that point no trainer was running; the fully materialized cache and 8×H100 pod remained intact
  while launch was paused after the user's explicit stop.
- The user then authorized immediate launch and increased the locked run length from two to four
  epochs. The first launch initialized W&B but exited before optimizer step 1 in the step-zero
  rollout diagnostic: `len(prefix)` returned batch size rather than temporal length. Replaced both
  rollout uses with `prefix.shape[1]`; this changed no training or input semantics.
- Resumed the same W&B run `25co20q1` at
  `https://wandb.ai/dhruvbhatia0/smash-d-transformer-full/runs/25co20q1`, launcher PID `172055`.
  Step-zero evaluation completed and training is live. At step 100: train loss `3.88098`, gradient
  norm `54.59`, throughput `279.35 clips/s`, data wait `0.011 ms`, and all eight GPUs are at
  99–100% utilization. Four epochs are 24,456 steps at the observed epoch plan, implying about
  3.1 hours of pure training and roughly 3.25–3.5 hours including evaluation/checkpoints.
- Aborted that run at the user's request after a transient instability: the 100-step mean loss rose
  to `2.73` at step 700 and `8.91` at step 800 after the pre-clip gradient norm reached `1,127`.
  It recovered to loss `0.802` / norm `22.97` by step 1,200. A deterministic loader-plan audit
  ruled out unshuffled archive blocks: the affected 100-step windows covered 88–90 archives and no
  archive exceeded 3.1% of rank-batches. The earlier step-zero rollout exception was a hard process
  exit followed by a fresh restart, not a silent fallback.
- Reviewed the original production trainer commits and the richer ablation logger. Added a global
  gradient-norm ceiling of `100` on every update with non-finite errors enabled. Restored the useful
  diagnostics at 20-step cadence: loss mean/std/min/max and five diffusion-time bins; pre-clip
  gradient mean/max, clip fraction/minimum scale, and action-encoder gradient; prediction/target
  velocity norms and cosine; raw/EMA/zero-action fixed eval; and correct/zero-action rollout deltas
  at every horizon. Per-replay tables, parameter scans, histograms, and redundant GPU metrics remain
  omitted because they are noisy or too expensive for the 1.2B model.
- Fresh clipped run: `https://wandb.ai/dhruvbhatia0/smash-d-transformer-full/runs/lmyej81l`, launcher
  PID `173285`, output `/workspace/full-run-clipped`. At step 120 loss was `1.148`, pre-clip norm mean
  `53.7` / max `76.3`, clip fraction `0`, throughput `275.4 clips/s`, and data wait `0.012 ms`.
- The clipped run saved a valid epoch-1 checkpoint at optimizer step 6,114, then continued to step
  12,220 (almost two epochs) before the finite loader exhausted and raised `StopIteration`. Epoch 2
  was not checkpointed. The durable epoch-1 model and EMA tensors were checked finite.
- Uploaded the resumable 19,311,263,845-byte epoch-1 checkpoint, SHA-256 manifest, final status,
  complete log, W&B run ID, and failure/resume metadata to the private Hugging Face repository
  `https://huggingface.co/DhruvBhatia0/smash-d-transformer`. Checkpoint SHA-256:
  `7deb8b548739f992327880a131be469e390704fe47ca38108d1ddf6c2785472a`.
- Terminated RunPod pod `lu5b5srnu3rnrm` after verifying the Hugging Face checkpoint commit. The
  RunPod API returned HTTP 204 for deletion and HTTP 404 on the subsequent identity lookup.
- Before any future restart, fix the per-rank RNG seed and resolve the intended forcing recipe:
  this checkpoint used independent per-frame flow times plus an explicit clean-past side channel.
