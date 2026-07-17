# One-sample overfit check — 2026-07-16

The end-to-end path was tested on a RunPod RTX A4000 using the final trained codec and the first
40-frame clip in Drive archive `batch-b7bbcb7492d4bc733e75.tar.zst`. The same clip was used for
training and deterministic evaluation.

- Video: `(1, 40, 3, 208, 252)` uint8 at 20 FPS.
- Controller inputs: `(1, 39, 3, 2, 56)` — 39 video transitions, three ordered 60 Hz inputs,
  two players, and 56 lossless analog/button features.
- Normalized codec latent: `(1, 20, 32, 7, 8)`.
- Ablation DiT: width 256, depth 4, 8 query heads, 2 KV heads, 5,739,712 parameters.
- Optimization: 1,000 AdamW steps at `1e-3`; fixed noise and flow times for evaluation.

| Step | Train loss | Fixed eval loss |
| ---: | ---: | ---: |
| 0 | — | 1.93401 |
| 100 | 0.49959 | 0.35429 |
| 500 | 0.06202 | 0.11743 |
| 900 | 0.04371 | 0.08228 |
| 1,000 | 0.24292 | 0.07711 |

The final stochastic training sample is noisy because training resamples Gaussian noise and flow
time every step. The fixed evaluation objective fell by 96.0%, confirming that data parsing, codec
normalization, action conditioning, flow matching, backward, and optimizer updates work together.

## Final aligned rerun

After replacing the narrow joint action bottleneck with MIRA-style full-width per-player temporal
pooling, the final code was rerun with the same replay as both the one-item training set and the
one-item evaluation set. See the [W&B run](https://wandb.ai/dhruvbhatia0/smash-d-transformer/runs/oxaco8zb).

| Step | Fixed eval loss |
| ---: | ---: |
| 0 | 1.97379 |
| 100 | 1.14167 |
| 500 | 0.10097 |
| 1,000 | 0.06297 |

The final-code loss fell 96.8%, and its six-step same-clip autoregressive rollout MSE was `0.00487`.
The full held-out experiment matrix and readiness decision are recorded in
[`ABLATION_RUN_2026-07-16.md`](ABLATION_RUN_2026-07-16.md).
