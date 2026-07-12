# Daytona 100-SLP native-60-to-20-Hz action audit

## Decision

For existing Slippi replays, use **multiple ordered inputs inside each 20 Hz frame**. Encode the
three 60 Hz sub-actions into one 20 Hz action token. Do not keep only one controller snapshot and
pretend it caused the 50 ms transition.

This preserves the desired real-time clock:

```text
20 world-model steps = 20 images = 20 action tokens = 1 real second

action token k = Encoder([u(3k+1), u(3k+2), u(3k+3)])
x(3k) + action token k -> x(3k+3)
```

The token rate is 20 Hz. The encoder merely describes what happened within the token's 50 ms.

If deployment truly reads a human controller only at 20 Hz, the correct dataset is different:
collect or rerun gameplay with each chosen 20 Hz action held for three emulator ticks. A native
60 Hz SLP transition cannot be relabeled truthfully with one sampled action after the fact.

## Corpus and machine

- 100 content-unique tournament singles replays from the CC0
  [`erickfm/slippi-public-dataset-v3.7`](https://huggingface.co/datasets/erickfm/slippi-public-dataset-v3.7)
  corpus.
- Deterministic seed `20260712`; basename deduplication followed by SHA-256 deduplication and
  top-up. The corpus mirrors a match under both players' character directories, so both checks
  matter.
- 94,487 unique-basename candidates from 221,943 `.slp` paths.
- 266,632,576 bytes downloaded; 100 parsed successfully and zero failed.
- 5.0965 match-hours, 366,951 world bins and 733,902 player bins at 20 Hz.
- Daytona `daytona-small`: Linux, 1 CPU, 1 GiB RAM, 3 GB disk, no GPU. It was deleted after the
  results and manifest were copied locally.

The sample is an older competitive-tournament corpus. The result is representative of these
competitive 1v1 replays, not every possible human/player population.

## What “lost” means

For a causal transition from post-state `x[t]` to `x[t+3]`, the real source inputs are
`u[t+1], u[t+2], u[t+3]`. The causal single-sample baseline keeps `u[t+1]` and holds it for the
whole 50 ms interval.

The audit counts full deletion, not merely timing quantization:

- A button pulse is hidden only if none of its pressed ticks intersects a retained sample.
- A stick-direction episode is hidden only if its thresholded direction never intersects a
  retained sample.
- A changed semantic bin contains more than one combination of gameplay buttons, main-stick
  direction, C-stick direction, or trigger state in its three source ticks.
- A high-stakes event is associated with an active combo role, a nearby landed move, an actor or
  opponent in a vulnerable state, a nearby stock end, or an actor state transition within six
  ticks.

The context labels are observational. They establish that an omitted input occurred during a
combo/state sequence, but only a counterfactual emulator rerun can prove how the match would have
diverged without that input.

## Headline results

| Causal 20 Hz metric | Count | Rate | Replay-bootstrap 95% interval |
|---|---:|---:|---:|
| Player bins containing multiple semantic actions | 230,766 / 733,902 | 31.44% | 30.67–32.32% |
| Fully hidden button pulses | 4,148 / 79,160 | 5.24% | 4.38–6.17% |
| Fully hidden stick-direction episodes | 27,443 / 171,479 | 16.00% | 15.18–16.71% |
| World bins with any fully hidden event | 29,045 / 366,951 | 7.92% | 7.50–8.37% |
| World bins with a high-stakes hidden event | 23,416 / 366,951 | 6.38% | 6.06–6.76% |

Moving the retained sample later does not solve deletion. The early, middle, and late phases hide
5.24%, 5.19%, and 5.12% of button pulses, and 16.00%, 15.90%, and 15.94% of direction episodes.
Middle/late sampling also looks into the transition being predicted and is therefore not a causal
live-input label.

The 29,045 affected bins correspond to about 1.58 affected world frames per second across both
players. The high-stakes subset is about 1.28 frames per second.

## Which buttons disappear

| Button | Hidden / total presses | Hidden rate |
|---|---:|---:|
| X (jump) | 1,629 / 13,344 | 12.21% |
| Y (jump) | 1,457 / 17,168 | 8.49% |
| Z (grab) | 131 / 3,560 | 3.68% |
| B (special) | 290 / 9,964 | 2.91% |
| A (attack) | 363 / 13,117 | 2.77% |
| R (shield) | 166 / 12,159 | 1.37% |
| L (shield) | 112 / 9,848 | 1.14% |

X/Y account for 3,086 of 4,148 hidden button pulses: **74.4%**. Every completely hidden press is
short: 1,940 one-tick presses and 2,208 two-tick presses. No press lasting three or more ticks is
fully deleted.

Short does not mean irrelevant. In inspected examples, the hidden X press occurs on the same tick
that the actor changes into action state 24 (jump squat). Several are active-combo attacker events;
one two-tick X press occurs near a landed move. Another tap occurs while the actor is in damage
state and may be an ignored or buffered attempt. The examples therefore include both truly
outcome-producing jumps and inputs that may have no immediate effect—the aggregate is not merely
controller noise.

## Combo association

- 998 of 4,148 hidden button pulses (24.1%) occur while that player is the active combo attacker
  or victim. Within those active-combo roles, 3.96% of button pulses are completely hidden.
- 6,737 of 27,443 hidden direction episodes (24.6%) occur in active combo roles. Within those
  roles, 13.86% of direction episodes are completely hidden.
- 3,588 hidden button pulses are followed by an actor action-state transition within six source
  ticks. This proxy can include transitions not caused by the press, so it is evidence of temporal
  importance rather than causal proof.
- The broader Slippi conversion window gives larger counts but includes a reset grace period; it
  was deliberately not used as the main “during a combo” claim.

## Why one sampled action is a bad training label

Suppose the saved image changes from grounded to airborne over a 50 ms transition, but the one
retained controller snapshot says “no jump.” The world model is trained to produce an airborne
target without seeing its cause. Enough examples like this teach noisy or apparently spontaneous
dynamics. OR-ing all buttons does not fix it: it removes order and releases, cannot represent
analog trajectories, and can invent impossible simultaneous combinations.

The fixed-width ordered macro is the smallest faithful answer for existing SLPs:

1. Keep one image/state every 50 ms.
2. Keep the three ordered controller records that produced the next image.
3. Give substeps 0/1/2 position embeddings.
4. Concatenate or pool them with an order-sensitive encoder into one action token.
5. Feed exactly one image token and one action token per 20 Hz world-model step.

This satisfies “one second in the model feels like one second in the game” while preventing the
model from having to hallucinate omitted causes.

## Local artifacts

Large raw artifacts remain ignored under `runs/daytona-100-audit/`:

- `sample-manifest.json`: source paths, replay hashes, and sizes.
- `analysis-100.json`: per-replay metrics and saved event windows.
- `summary-100.json`: pooled, replay-distribution, and 2,000 replay-level bootstrap replicates.
- `analyze_20hz.mjs`, `download_100.py`, `summarize_analysis.py`: audit scripts used on Daytona and
  locally.

The copied analysis SHA-256 is
`05421eb884d728385a06efe41dd909993768937145148e3bac2eaa5362b40b7b`; the sample-manifest SHA-256
is `f1b84da4aed5a62348cb0b6638ee23f989ba9ce22c9d09b01797d4a10185aff0`.
