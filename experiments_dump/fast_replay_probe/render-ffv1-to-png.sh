#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=""
REPLAY_JSON=""
VIDEO_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      VIDEO_ARGS+=("$1" "$2/.ffv1-video")
      shift 2
      ;;
    --replay-json)
      REPLAY_JSON="$2"
      VIDEO_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      VIDEO_ARGS+=("$1")
      if [[ $# -gt 1 && "$2" != --* ]]; then
        VIDEO_ARGS+=("$2")
        shift 2
      else
        shift
      fi
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Missing --output-dir" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/framedump_*.png

"$SCRIPT_DIR/render-ffv1-replay.sh" "${VIDEO_ARGS[@]}"

video="$(find "$OUTPUT_DIR/.ffv1-video" -maxdepth 1 -type f -name 'framedump*.avi' | sort | head -1)"
if [[ -z "$video" ]]; then
  echo "Missing FFV1 video after render" >&2
  exit 3
fi

ffmpeg -hide_banner -loglevel error \
  -i "$video" \
  -compression_level "${SLIPPI_FFMPEG_PNG_COMPRESSION_LEVEL:-1}" \
  -pred "${SLIPPI_FFMPEG_PNG_PRED:-none}" \
  "$OUTPUT_DIR/framedump_%d.png"

python3 - <<'PY' "$OUTPUT_DIR" "$REPLAY_JSON"
from pathlib import Path
import json
import re
import struct
import sys

out_dir = Path(sys.argv[1])
replay_json = Path(sys.argv[2]) if sys.argv[2] else None
video_manifest_path = out_dir / ".ffv1-video" / "manifest.json"
video_manifest = json.loads(video_manifest_path.read_text()) if video_manifest_path.exists() else {}
pngs = sorted(out_dir.glob("framedump_*.png"), key=lambda p: int(p.stem.split("_")[-1]))

def png_size(path: Path):
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return {"width": width, "height": height}

try:
    replay_config = json.loads(replay_json.read_text()) if replay_json else {}
except Exception as error:
    replay_config = {"error": f"{type(error).__name__}: {error}"}

run_log = out_dir / ".ffv1-video" / "render-ffv1.log"
current_frames = []
if run_log.exists():
    for line in run_log.read_text(errors="replace").splitlines():
        match = re.search(r"\[CURRENT_FRAME\]\s+(-?\d+)", line)
        if match:
            current_frames.append(int(match.group(1)))

requested_end_frame = replay_config.get("endFrame")
completed_requested_range = (
    not isinstance(requested_end_frame, int)
    or bool(current_frames and current_frames[-1] >= requested_end_frame)
)

manifest = {
    **video_manifest,
    "replayJson": str(replay_json) if replay_json else "",
    "replayConfig": replay_config,
    "pipeline": "ffv1-video-then-ffmpeg-png",
    "frameCount": len(pngs),
    "pngBytes": sum(path.stat().st_size for path in pngs),
    "currentFrameLogEntries": len(current_frames),
    "currentFrameRange": {
        "first": current_frames[0] if current_frames else None,
        "last": current_frames[-1] if current_frames else None,
    },
    "requestedEndFrame": requested_end_frame,
    "completedRequestedRange": completed_requested_range,
    "samples": [
        {
            "path": str(pngs[index]),
            "dumpFrameNumber": int(pngs[index].stem.split("_")[-1]),
            "png": png_size(pngs[index]),
        }
        for index in sorted(set(i for i in (0, len(pngs) // 2, len(pngs) - 1) if 0 <= i < len(pngs)))
    ],
    "logs": {
        "run": str(run_log),
        "dolphin": str(out_dir / ".ffv1-video" / "dolphin.log"),
    },
    "video": video_manifest.get("videos", []),
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))

if not pngs:
    raise SystemExit("No PNGs were extracted from FFV1 video")
if not completed_requested_range:
    raise SystemExit(
        "Render stopped before requested end frame: "
        f"last={manifest['currentFrameRange']['last']} requested={requested_end_frame}"
    )
PY
