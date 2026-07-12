# 60 Hz controls with a 20 Hz world model

## Contract

Run the world model and generate images at 20 Hz, but sample the real controller at 60 Hz. Every
20 Hz transition receives the three ordered controller samples that occurred during its 50 ms
interval.

```text
60 Hz controls:    u0 -> u1 -> u2 | u3 -> u4 -> u5
                            |                 |
20 Hz action token:       A0                A1
                            |                 |
20 Hz images:       x0 ---------> x1 ------------> x2
```

The action-token rate remains 20 Hz:

```text
A[k] = ActionEncoder([u[3k+1], u[3k+2], u[3k+3]])
x[3k+3] = WorldModel(x[3k], A[k])
```

Therefore 20 model steps, 20 generated images, and 20 action tokens always represent one real
second. The three controller samples are substeps inside one token, not extra world-model steps.

## Training representation

Use a fixed-width ordered tensor rather than an unordered or variable-length bag:

```text
[batch, world_step, 3_substeps, player, action_features]
```

Give substeps 0, 1, and 2 distinct position embeddings. Encode them with a small order-sensitive
network, then project the result into one action token for the corresponding world-model step.
Do not average sticks, OR buttons, or discard releases: those operations destroy ordering and can
make different control sequences indistinguishable.

Each training transition must align as:

```text
source image/state x[t]
source controls    [u[t+1], u[t+2], u[t+3]]
target image/state x[t+3]
```

## Online inference loop

1. Keep controller polling on a stable 60 Hz clock.
2. Accumulate three successive samples in order.
3. Encode the three-sample window into one action token.
4. Generate the next image/state at 20 Hz.
5. Clear the window and repeat.

Training and inference must use the same feature normalization, controller fields, substep order,
and 60-to-20 Hz phase. Timestamps should be used to detect missed or duplicated polling ticks.

## Latency

The target frame cannot be generated until the controller samples that cause it have arrived.
With a 20 Hz display, input-to-visible-frame quantization is approximately 0–50 ms (about 25 ms on
average), plus model inference and display time. Inference should therefore stay comfortably below
50 ms per generated frame and controller polling must remain independent of model execution.

This design preserves short taps and within-frame ordering while keeping image generation and the
world-model timeline at 20 Hz.
