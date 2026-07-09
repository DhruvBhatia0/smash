#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLAYBACK_JSON="${1:-$SCRIPT_DIR/playback.realtime.full.json}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$SCRIPT_DIR/runs/$RUN_ID"

mkdir -p "$RUN_DIR"

export ROOT
export USER_DIR="$RUN_DIR/dolphin-user"
export FRAME_OUTPUT_DIR="$RUN_DIR/raw-frames"
export LOG_DIR="$RUN_DIR/logs"
export RUN_LOG="$RUN_DIR/logs/render-replay-debug.log"
export PID_FILE="$RUN_DIR/logs/render-replay-debug.pid"
export TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
export KILL_EXISTING_DOLPHIN="${KILL_EXISTING_DOLPHIN:-1}"

started_ns="$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)"

/usr/bin/time -lp "$ROOT/scripts/render-replay-debug.sh" "$PLAYBACK_JSON" \
  2>&1 | tee "$RUN_DIR/render-baseline.time.log"

finished_ns="$(python3 - <<'PY'
import time
print(time.time_ns())
PY
)"

python3 - <<'PY' "$RUN_DIR" "$started_ns" "$finished_ns"
from pathlib import Path
import json
import sys

run_dir = Path(sys.argv[1])
started_ns = int(sys.argv[2])
finished_ns = int(sys.argv[3])
manifest_path = run_dir / "raw-frames" / "manifest.json"
render_manifest = {}
if manifest_path.exists():
    render_manifest = json.loads(manifest_path.read_text())
pngs = sorted((run_dir / "raw-frames").glob("framedump_*.png"))
summary = {
    "runDir": str(run_dir),
    "wallSeconds": round((finished_ns - started_ns) / 1_000_000_000, 6),
    "pngCount": len(pngs),
    "pngBytes": sum(path.stat().st_size for path in pngs),
    "renderManifest": render_manifest,
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
