# MIRA alignment and action-data ablations — 2026-07-16

The complete dashboard is the [W&B `smash-d-transformer` project](https://wandb.ai/dhruvbhatia0/smash-d-transformer).
The final model code passes the exact same-sample train/eval overfit check, and the held-out
ablations support retaining all three ordered 60 Hz controller inputs rather than reducing them to
one 20 Hz value.

## Data and alignment

The Drive dataset has 448 committed `tar.zst` archives, 10,327 rendered replays, and 561.8 GB of
compressed data. Archive metadata and direct SLP inspection agree on:

- source SLP and render simulation rate: 60 Hz;
- encoded video rate: 20 FPS;
- source frame step: 3;
- video clip: `(40, 3, 208, 252)` uint8;
- controller clip: `(39, 3, 2, 56)` float32.

For video frames `x[t]` and `x[t + 1]`, the loader returns the three inputs that actually drive the
transition, in order. Each input contains two players and 56 lossless controller features: eight
analog values, 32 processed-button bits, and 16 physical-button bits. With the codec's temporal
stride of two, every predicted latent after the initial token pools two video transitions, or six
ordered controller inputs per player. The one-transition offset matches MIRA's
`action_temporal_downsampling - 1` alignment.

The 24-replay experiment cache shows that the extra temporal detail is not redundant:

| Data statistic | Fraction |
| --- | ---: |
| Adjacent 60 Hz pair changes for a player | 52.11% |
| Any within-transition change for a player | 65.33% |
| Any within-transition change for either player | 87.61% |
| Analog pair changes | 48.61% |
| Processed-button pair changes | 19.93% |
| Physical-button pair changes | 9.46% |

Replacing every three-input group with its final input has feature MSE `0.00964`; it is a lossy
transformation of a frequently changing stream.

## MIRA parity

The final defaults follow the released MIRA 1B configuration: width 2,048, 16 blocks, 16 query
heads, four KV heads, temporal attention every four blocks, causal time attention, attention
gating, per-head QK LayerNorm, exact temporal and axial spatial RoPE, AdaLN on attention and MLP
sublayers, SwiGLU, and MIRA initialization. The resulting Smash model has 1.207B parameters.

Training uses the same diagonal flow-matching objective:

```text
epsilon ~ N(0, I)
tau ~ Uniform(0, 1), independently per latent frame
z_tau = tau * z_clean + (1 - tau) * epsilon
target_velocity = z_clean - epsilon
loss = MSE(model(z_tau, actions, tau), target_velocity)
```

The codec is frozen, its latents are normalized with checkpoint statistics, and AdamW defaults now
match MIRA (`lr=1e-4`, betas `(0.9, 0.99)`, weight decay `0.1`, 1,000 warmup steps, EMA `0.9999`).
The released `use_clean_past=true` path is the default; the noised-past path remains available as an
ablation.

The exact MIRA keyboard/mouse action encoder cannot be copied unchanged: Smash has two players,
analog sticks/triggers, button bitfields, and three 60 Hz microsteps per video transition. The final
adapter nevertheless follows MIRA's structure: full-transformer-width learned temporal pooling per
player, then a player embedding, shared projection, and mean combination. This replaced an early
narrow joint bottleneck and reduced held-out six-step rollout MSE from `0.6595` to `0.5197`, a 21.2%
reduction.

The intentional codec difference remains DINOv3 small+ rather than DINOv3 large. The training
objective and transformer do not need a special case for it.

## Experiment setup

All final comparisons used the same seed, initialization, 1,000 optimizer steps, 6.05M-parameter
ablation model, and fixed evaluation noise/timesteps. The first 16 cached replays were training data
and the last eight were disjoint held-out evaluation data. Rollout evaluation seeded one true
latent and autoregressively generated six latent steps (about 0.6 seconds) with eight Euler steps.
Only the named variable changed.

The W&B runs log metrics that diagnose training rather than generic parameter dumps:

- train distribution and loss by diffusion-time bin;
- raw and EMA fixed evaluation, per replay, latent position, and diffusion-time bin;
- target/prediction velocity norm and cosine;
- paired zero, shifted, reversed-microstep, player-swap, and last-only action interventions;
- correct and counterfactual autoregressive errors at every horizon;
- action-encoder and global gradient norms, parameter norm, update ratio, and learning rate;
- examples, unique replays, equivalent epochs, source seconds, frames, and player micro-inputs;
- step/clip/frame throughput and peak GPU allocation/reservation.

FID/FVD and weight histograms were intentionally omitted from this 8-replay test. They would be
statistically weak and do not answer whether the model follows controls. The next larger evaluation
should add MIRA's inverse-dynamics action reconstruction ratio and decoded rollout video.

## Final action/history results

Lower loss and rollout MSE are better. `Zero-action delta` is the final-horizon error increase after
zeroing the inputs in the same trained model; positive values show that correct actions help.

| Run | Clean past | Action representation | Dropout | Best fixed eval | Rollout MSE h06 | Zero-action delta |
| --- | :---: | --- | ---: | ---: | ---: | ---: |
| [`miraenc-clean-ordered60`](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/437jh2jd) | yes | ordered 60 Hz | 0% | 0.50988 | 0.51970 | +0.07616 |
| [`miraenc-clean-none`](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/dv0w8gov) | yes | none | 0% | **0.50443** | 0.58068 | — |
| [`miraenc-noisy-ordered60`](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/67jan411) | no | ordered 60 Hz | 0% | 0.58915 | **0.49601** | **+0.12234** |
| [`miraenc-noisy-summary60`](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/ikjuuvfh) | no | invertible mean/first/last | 0% | 0.58898 | 0.51002 | +0.12128 |
| [`miraenc-noisy-last20`](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/tjj5ofor) | no | final input only | 0% | 0.58926 | 0.53438 | +0.10217 |
| [`miraenc-noisy-none`](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/njmve6v0) | no | none | 0% | **0.58402** | 0.61450 | — |
| [`miraenc-noisy-ordered60-drop10`](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/hd3ptfq1) | no | ordered 60 Hz | 10% | 0.60096 | 0.51992 | +0.07663 |

The teacher-forced metric still slightly favors no-action models, but it gives the wrong model
selection answer. With noised past, ordered 60 Hz actions reduce rollout error by 19.3% versus no
actions, 7.2% versus a final-input-only 20 Hz representation, and 2.8% versus the invertible
hand-engineered summary. Under clean-past training, ordered actions reduce rollout error by 10.5%
versus no actions. Reversing the ordered microsteps in the clean model increases final-horizon error
by `0.03697`.

Noised-past training improves ordered-action rollout MSE by 4.6% over the released clean-past
default and produces a larger action counterfactual gap, despite worse one-step loss. This is the
most interesting larger-scale follow-up, but 16 training replays and two rollout replays are not
enough to replace MIRA's released default. Ten-percent action dropout hurts rollout by 4.8% here;
use zero dropout for tiny-data pilots and retest the regularizer at real scale.

## Data-efficiency curve

Each run consumed exactly 1,000 examples, so increasing the replay count reduces repetition while
holding optimizer compute fixed.

| Unique training replays | Equivalent epochs | Best eval (step) | Final eval | Rollout MSE h06 |
| ---: | ---: | ---: | ---: | ---: |
| [1](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/lkek1voe) | 1,000 | 0.93403 (200) | 2.41196 | 0.57664 |
| [4](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/qlds19bh) | 250 | 0.66228 (300) | 1.28045 | 0.54510 |
| [16](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/437jh2jd) | 62.5 | **0.50988 (900)** | **0.51255** | **0.51970** |

Repeated exposure to one or four clips causes severe held-out overfitting. At fixed compute, unique
replay diversity is much more useful than additional epochs over the same trajectories.

## Exact one-sample overfit

The final-code [`miraenc-same-sample-overfit` run](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/oxaco8zb)
uses the same replay clip for the one training item and one evaluation item.

| Step | Fixed eval loss |
| ---: | ---: |
| 0 | 1.97379 |
| 100 | 1.14167 |
| 200 | 0.57139 |
| 500 | 0.10097 |
| 1,000 | **0.06297** |

Loss falls 96.8%, and the same-clip six-step autoregressive rollout MSE is `0.00487`. This verifies
SLP parsing, video/action alignment, codec normalization, latent caching, flow matching, backward,
optimizer updates, EMA, and rollout integration together.

## Readiness and next decision

The architecture and small-scale trainer are ready for a larger pilot. The evidence supports:

1. Keep all three ordered 60 Hz inputs.
2. Keep the full-width per-player MIRA-style action encoder.
3. Use the released clean-past configuration as the control run, with zero action dropout for a
   low-data pilot.
4. Run noised past as the primary research branch; it is the best rollout result here.

Do not launch the expensive full-corpus 1.2B run yet. First convert latent preprocessing from the
small monolithic ablation cache to resumable shards, then repeat the clean/noised-past comparison on
at least thousands of unique replays with decoded rollouts and an inverse-dynamics action probe.
The streaming SLP/video loader itself is ready; the sharded latent materialization is the remaining
production data-path task.

The experiments ran on a RunPod RTX A4000 at `$0.17/hour`. The pod existed for 39.5 minutes, cost an
estimated `$0.112`, synchronized all W&B runs, removed credentials, and was deleted successfully.
