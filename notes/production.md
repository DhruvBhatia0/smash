# Production notes

## HF identity

Always set `SMASH_HF_EXPECTED_EMAIL` before real uploads. The storage layer verifies
the bearer token with Hugging Face and fails before writing if the token belongs to
another account.

```bash
pnpm hf:sync -- --expected-email dhruv.bhatia.j@gmail.com whoami
```

## Resume behavior

Frame queue runs are keyed by `run-id`. If a run is interrupted, rerun with the same
`--run-id` and `--resume-existing-run`; jobs with existing `processed` result files
are skipped.

```bash
pnpm queue:frames -- replays/downloaded \
  --runtime runpod \
  --run-id RUN_ID \
  --resume-existing-run
```

## Observability

Each queue run writes:

- `manifest.json`: run-level configuration and final counts.
- `index.jsonl`: one final row per processed job attempt.
- `events.jsonl`: lifecycle events such as run start, job start, job finish, and run finish.

## Local disk bound

Use `--upload-processed-to-hf --delete-local-after-hf-upload` to prune bulky per-job
frame directories after successful upload. The current local peak is still one active
job per consumer; pushing alignment and HF upload into the RunPod worker is the next
step to remove local frame staging.
