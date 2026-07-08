# Hugging Face storage

The dataset storage path is split into two prefixes:

- `raw/slp`: original `.slp` files, optionally mirrored directly from a download manifest.
- `processed/frame-queues`: rendered frame queue outputs, keyed by run id and job id.

Install the Python dependency first:

```bash
python3 -m pip install -r requirements.txt
```

Use `HF_TOKEN` for auth. Set `SMASH_HF_EXPECTED_EMAIL` so real writes fail closed
if the token points at the wrong account. For throughput on large processed-frame
uploads, set `HF_XET_HIGH_PERFORMANCE=1`.

Verify the active token:

```bash
pnpm hf:sync -- --expected-email dhruv.bhatia.j@gmail.com whoami
```

Dry-run a raw upload:

```bash
pnpm hf:sync -- --repo USER/DATASET --dry-run upload-slp replays/realtimeTest.slp
```

Mirror a manifest with bounded temporary disk usage:

```bash
pnpm hf:sync -- --repo USER/DATASET mirror-manifest \
  --manifest download-manifests/slippi-js-samples.json \
  --concurrency 2
```

Upload processed frame outputs after a run:

```bash
pnpm hf:sync -- --repo USER/DATASET upload-processed-run \
  --run-dir processed-frame-queues/RUN_ID
```

Or publish each processed job as it finishes, then delete that bulky job directory locally:

```bash
pnpm queue:frames -- replays/downloaded \
  --runtime runpod \
  --upload-processed-to-hf \
  --hf-processed-repo USER/DATASET \
  --hf-expected-email dhruv.bhatia.j@gmail.com \
  --resume-existing-run \
  --delete-local-after-hf-upload
```
