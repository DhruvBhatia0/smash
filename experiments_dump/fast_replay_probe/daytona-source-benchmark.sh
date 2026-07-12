#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: daytona-source-benchmark.sh LABEL}"
root="/home/daytona/slippi-source-probe"
output="$root/$label"
wrapper="/tmp/render-ffv1-replay.sh"

rm -rf -- "$output"
mkdir -p "$output"

started_ns="$(date +%s%N)"
LP_NUM_THREADS="${LP_NUM_THREADS:-4}" \
LIBGL_ALWAYS_SOFTWARE=1 \
EGL_PLATFORM=surfaceless \
MESA_GLTHREAD="${MESA_GLTHREAD:-false}" \
SLIPPI_DOLPHIN_BIN=/opt/slippi/dolphin-emu-nogui \
SLIPPI_DUMP_ONLY="${SLIPPI_DUMP_ONLY:-1}" \
SLIPPI_DUMP_FRAMES=True \
SLIPPI_USE_FFV1="${SLIPPI_USE_FFV1:-False}" \
SLIPPI_DUMP_CODEC="${SLIPPI_DUMP_CODEC:-rawvideo}" \
SLIPPI_DUMP_FORMAT="${SLIPPI_DUMP_FORMAT:-avi}" \
SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS="${SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS:-True}" \
SLIPPI_EFB_SCALE="${SLIPPI_EFB_SCALE:-0}" \
SLIPPI_CPU_THREAD="${SLIPPI_CPU_THREAD:-False}" \
SLIPPI_RENDER_TO_MAIN="${SLIPPI_RENDER_TO_MAIN:-False}" \
SLIPPI_RENDER_WIDTH="${SLIPPI_RENDER_WIDTH:-204}" \
SLIPPI_RENDER_HEIGHT="${SLIPPI_RENDER_HEIGHT:-168}" \
SLIPPI_END_FRAME_POLL_SECONDS=0.05 \
  "$wrapper" \
    --replay-json "$root/playback.json" \
    --iso "$root/melee.iso" \
    --output-dir "$output" \
    --user-dir "$root/dolphin-user" \
    --timeout-seconds 150 \
    --video-backend OGL \
    --cpu-core 1 \
    --audio-backend Null \
    --no-xvfb

elapsed_ns="$(( $(date +%s%N) - started_ns ))"
printf '%d.%03d\n' \
  "$(( elapsed_ns / 1000000000 ))" \
  "$(( elapsed_ns % 1000000000 / 1000000 ))" \
  > "$output/wall-seconds.txt"
ffprobe -v error -count_packets -select_streams v:0 \
  -show_entries stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,nb_read_packets,duration \
  -of json "$output/framedump0.avi" > "$output/ffprobe.json"

python3 - "$root/playback.json" "$output/ffprobe.json" <<'PY'
import json
import sys
from pathlib import Path

playback = json.loads(Path(sys.argv[1]).read_text())
probe = json.loads(Path(sys.argv[2]).read_text())
expected = playback["endFrame"] - playback["startFrame"] - 1
stream = probe["streams"][0]
actual = int(stream["nb_frames"])
if actual != expected:
    raise SystemExit(f"CFR frame count mismatch: expected {expected}, got {actual}")
print(f"validated_cfr_frames={actual}")
PY

cat "$output/wall-seconds.txt"
cat "$output/ffprobe.json"
