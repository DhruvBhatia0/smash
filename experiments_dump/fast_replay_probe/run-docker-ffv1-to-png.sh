#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-smash/slippi-renderer:local}"
RUN_ID="${RUN_ID:-docker-ffv1-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$SCRIPT_DIR/runs/$RUN_ID"
ISO="${ISO:-/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso}"
REPLAY_JSON="${REPLAY_JSON:-$SCRIPT_DIR/playback.realtime.full.json}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
DOLPHIN_CPU_CORE="${DOLPHIN_CPU_CORE:-0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-OGL}"

mkdir -p "$RUN_DIR/video" "$RUN_DIR/png"

started_ns="$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)"

docker run --rm --platform linux/amd64 \
  -v "$ROOT:/workspace" \
  -v "$ISO:/iso/melee.iso:ro" \
  --entrypoint /bin/bash \
  "$IMAGE_TAG" \
  /workspace/fast_replay_probe/render-ffv1-replay.sh \
    --replay-json "/workspace/${REPLAY_JSON#$ROOT/}" \
    --iso /iso/melee.iso \
    --output-dir "/workspace/${RUN_DIR#$ROOT/}/video" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --video-backend "$VIDEO_BACKEND" \
    --cpu-core "$DOLPHIN_CPU_CORE"

render_done_ns="$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)"

video="$(find "$RUN_DIR/video" -maxdepth 1 -type f -name 'framedump*.avi' | sort | head -1)"
/usr/bin/time -lp ffmpeg -hide_banner -loglevel error \
  -i "$video" \
  -compression_level "${PNG_COMPRESSION_LEVEL:-1}" \
  -pred "${PNG_PRED:-none}" \
  "$RUN_DIR/png/frame_%06d.png" \
  > "$RUN_DIR/ffmpeg-extract.log" 2>&1

finished_ns="$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)"

python3 - <<'PY' "$RUN_DIR" "$started_ns" "$render_done_ns" "$finished_ns"
from pathlib import Path
import json
import sys

run_dir = Path(sys.argv[1])
started_ns = int(sys.argv[2])
render_done_ns = int(sys.argv[3])
finished_ns = int(sys.argv[4])
pngs = sorted((run_dir / "png").glob("frame_*.png"))
videos = sorted((run_dir / "video").glob("framedump*.avi"))
summary = {
    "runDir": str(run_dir),
    "renderSeconds": round((render_done_ns - started_ns) / 1_000_000_000, 6),
    "extractSeconds": round((finished_ns - render_done_ns) / 1_000_000_000, 6),
    "totalSeconds": round((finished_ns - started_ns) / 1_000_000_000, 6),
    "videoBytes": sum(path.stat().st_size for path in videos),
    "pngCount": len(pngs),
    "pngBytes": sum(path.stat().st_size for path in pngs),
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
