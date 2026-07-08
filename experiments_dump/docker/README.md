# Slippi Renderer Docker Image

This image packages a Linux Slippi Playback Dolphin build with the frame-dump patch used by the local macOS experiment. The image does not contain a Melee ISO or replay data. Mount those at runtime.

Build the RunPod-target image locally:

```bash
IMAGE_TAG=smash/slippi-renderer:local BUILD_JOBS=2 docker/scripts/build-renderer-image.sh
```

The build defaults to `linux/amd64`, which is the target for RunPod CPU Pods. On Apple Silicon, a full local `linux/amd64` render runs under qemu and may abort inside Dolphin; use local Docker for image/script validation and use a native x86 CPU Pod for render validation.

Validate the local Docker wiring:

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD:/workspace" \
  -v "/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso:/iso/melee.iso:ro" \
  smash/slippi-renderer:local \
  --replay-json /workspace/processed-frame-queues/runpod-live-dry-run-v4/jobs/realtimeTest-ed97860936/playback.json \
  --iso /iso/melee.iso \
  --output-dir /workspace/processed-frame-queues/docker-final-dry-run/raw-frames \
  --cpu-core 0 \
  --dry-run
```

Validate through the queue Docker runtime when running on native x86 Linux:

```bash
IMAGE_TAG=smash/slippi-renderer:local docker/scripts/validate-renderer-image.sh
```

Direct dry run:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -v "/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso:/iso/melee.iso:ro" \
  smash/slippi-renderer:local \
  --replay-json /workspace/replay-playback.short.json \
  --iso /iso/melee.iso \
  --output-dir /workspace/processed-frame-queues/docker-direct-dry-run/raw-frames \
  --dry-run
```

For RunPod, push the image to Docker Hub or another registry and pass that tag as the Pod `imageName` or as `--runpod-image`. Custom Pods need `22/tcp` exposed and an SSH daemon running when the queue will copy inputs/results over SSH; the image includes `start-runpod-worker.sh` for that.

For short-lived testing without registry credentials, an ephemeral public tag works:

```bash
IMAGE_TAG="ttl.sh/smash-slippi-renderer-$(uuidgen | tr '[:upper:]' '[:lower:]'):2h"
docker tag smash/slippi-renderer:local "$IMAGE_TAG"
docker push "$IMAGE_TAG"
```

Cheapest RunPod CPU planning defaults use CPU compute, `cpu3c`, one vCPU, a 10 GB container disk, and no persistent volume. The runtime injects `~/.ssh/id_ed25519.pub` by default when present, uses `~/.ssh/id_ed25519` for SSH/rsync, uploads the configured local ISO to `/workspace/iso/melee.iso`, and deletes the Pod in `close()`.

```bash
python3 scripts/process-frame-queue.py \
  --runtime runpod \
  --dry-run \
  --runpod-image "$IMAGE_TAG" \
  --runpod-cpu-flavor-id cpu3c \
  --runpod-vcpu-count 1 \
  --runpod-volume-gb none \
  --consumers 1 \
  --max-jobs 1 \
  replays/realtimeTest.slp
```

Live one-job smoke test:

```bash
python3 scripts/process-frame-queue.py \
  --runtime runpod \
  --runpod-image "$IMAGE_TAG" \
  --runpod-cloud-type COMMUNITY \
  --runpod-cpu-flavor-id cpu3c \
  --runpod-vcpu-count 1 \
  --runpod-volume-gb none \
  --runpod-container-disk-gb 10 \
  --consumers 1 \
  --queue-size 1 \
  --max-jobs 1 \
  --run-id runpod-live-dry-run \
  --start-frame -123 \
  --end-frame 10 \
  --timeout-seconds 120 \
  replays/realtimeTest.slp
```

Cleanup check:

```bash
python3 scripts/clean-up-runpod-cpu-pods.py
```
