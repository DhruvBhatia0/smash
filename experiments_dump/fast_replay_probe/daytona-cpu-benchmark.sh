#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-rawvideo}"
ROOT="/tmp/slippi-probe"
WRAPPER="/tmp/render-ffv1-replay.sh"
RESULT_NAME="${2:-${RESULT_NAME:-$MODE}}"
if [[ -n "${3:-}" ]]; then
  export LP_NUM_THREADS="$3"
fi
if [[ -n "${4:-}" ]]; then
  export SLIPPI_OVERCLOCK_FACTOR="$4"
fi
if [[ -n "${5:-}" ]]; then
  export SLIPPI_CPU_THREAD="$5"
fi
COMMON=(
  --replay-json "$ROOT/playback.json"
  --iso /tmp/melee.iso
  --output-dir "$ROOT/$RESULT_NAME"
  --timeout-seconds 150
  --video-backend OGL
  --cpu-core 1
  --audio-backend Null
)

rm -rf -- "$ROOT/$RESULT_NAME"
SECONDS=0
case "$MODE" in
  emulator-only)
    SLIPPI_DUMP_FRAMES=False \
      SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
      "$WRAPPER" "${COMMON[@]}"
    ;;
  emulator-only-auto)
    SLIPPI_DUMP_FRAMES=False \
      SLIPPI_EFB_SCALE=0 \
      SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
      "$WRAPPER" "${COMMON[@]}"
    ;;
  rawvideo)
    SLIPPI_DUMP_FRAMES=True \
      SLIPPI_USE_FFV1=False \
      SLIPPI_DUMP_CODEC=rawvideo \
      SLIPPI_DUMP_FORMAT=avi \
      SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS=True \
      SLIPPI_EFB_SCALE=2 \
      SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
      "$WRAPPER" "${COMMON[@]}"
    ;;
  rawvideo-backbuffer)
    SLIPPI_DUMP_FRAMES=True \
      SLIPPI_USE_FFV1=False \
      SLIPPI_DUMP_CODEC=rawvideo \
      SLIPPI_DUMP_FORMAT=avi \
      SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS=False \
      SLIPPI_EFB_SCALE=2 \
      SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
      "$WRAPPER" "${COMMON[@]}"
    ;;
  rawvideo-backbuffer-auto)
    SLIPPI_DUMP_FRAMES=True \
      SLIPPI_USE_FFV1=False \
      SLIPPI_DUMP_CODEC=rawvideo \
      SLIPPI_DUMP_FORMAT=avi \
      SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS=False \
      SLIPPI_EFB_SCALE=0 \
      SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
      "$WRAPPER" "${COMMON[@]}"
    ;;
  rawvideo-lowres)
    SLIPPI_DUMP_FRAMES=True \
      SLIPPI_USE_FFV1=False \
      SLIPPI_DUMP_CODEC=rawvideo \
      SLIPPI_DUMP_FORMAT=avi \
      SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS=False \
      SLIPPI_EFB_SCALE=0 \
      SLIPPI_RENDER_TO_MAIN=False \
      SLIPPI_RENDER_WIDTH=320 \
      SLIPPI_RENDER_HEIGHT=264 \
      SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
      "$WRAPPER" "${COMMON[@]}"
    ;;
  rawvideo-modelres)
    SLIPPI_DUMP_FRAMES=True \
      SLIPPI_USE_FFV1=False \
      SLIPPI_DUMP_CODEC=rawvideo \
      SLIPPI_DUMP_FORMAT=avi \
      SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS=False \
      SLIPPI_EFB_SCALE=0 \
      SLIPPI_RENDER_TO_MAIN=False \
      SLIPPI_RENDER_WIDTH=224 \
      SLIPPI_RENDER_HEIGHT=184 \
      SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
      "$WRAPPER" "${COMMON[@]}"
    ;;
  ffv1)
    SLIPPI_DUMP_FRAMES=True \
      SLIPPI_USE_FFV1=True \
      SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS=True \
      SLIPPI_EFB_SCALE=2 \
      SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
      "$WRAPPER" "${COMMON[@]}"
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac
printf '%s\n' "$SECONDS" > "$ROOT/$RESULT_NAME/wall-seconds.txt"
echo "DAYTONA_WALL_SECONDS=$SECONDS"
