#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-smash/slippi-renderer:local}"
ISO="${ISO:-/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso}"
SLP="${SLP:-$ROOT/replays/realtimeTest.slp}"
RUN_ID="${RUN_ID:-docker-renderer-local}"
START_FRAME="${START_FRAME:--123}"
END_FRAME="${END_FRAME:-10}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-60}"
VIDEO_BACKEND="${VIDEO_BACKEND:-OGL}"
DOLPHIN_CPU_CORE="${DOLPHIN_CPU_CORE:-0}"
DOLPHIN_AUDIO_BACKEND="${DOLPHIN_AUDIO_BACKEND:-Null}"

python3 "$ROOT/scripts/process-frame-queue.py" \
  --runtime docker \
  --docker-image "$IMAGE_TAG" \
  --consumers 1 \
  --queue-size 1 \
  --max-jobs 1 \
  --run-id "$RUN_ID" \
  --start-frame "$START_FRAME" \
  --end-frame "$END_FRAME" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --video-backend "$VIDEO_BACKEND" \
  --dolphin-cpu-core "$DOLPHIN_CPU_CORE" \
  --dolphin-audio-backend "$DOLPHIN_AUDIO_BACKEND" \
  --iso "$ISO" \
  "$SLP"
