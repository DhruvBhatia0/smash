#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  render-ffv1-replay.sh --replay-json <path> --iso <path> --output-dir <path> [options]

Options:
  --user-dir <path>         Dolphin user dir. Defaults to <output-dir>/../dolphin-user.
  --timeout-seconds <n>     Stop Dolphin after n seconds. Defaults to 120.
  --video-backend <name>    Dolphin video backend. Defaults to OGL.
  --cpu-core <n>            Dolphin CPU core. Defaults to 1.
  --audio-backend <name>    Dolphin DSP audio backend. Defaults to Null.
  --no-xvfb                 Run Dolphin directly. Use for EGL/headless builds.

Environment knobs:
  SLIPPI_DUMP_FRAMES=True|False
  SLIPPI_USE_FFV1=True|False
  SLIPPI_DUMP_CODEC=<ffmpeg codec name>
  SLIPPI_DUMP_FORMAT=<container, e.g. avi or mkv>
  SLIPPI_BITRATE_KBPS=<kbps>
  SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS=True|False
  SLIPPI_EFB_SCALE=<Dolphin EFB scale enum, 2 is 1x>
  SLIPPI_OVERCLOCK_FACTOR=<emulated CPU clock multiplier; defaults to 1.0/off>
  SLIPPI_CPU_THREAD=True|False
  SLIPPI_RENDER_TO_MAIN=True|False
  SLIPPI_RENDER_WIDTH=<pixels>
  SLIPPI_RENDER_HEIGHT=<pixels>
  SLIPPI_DUMP_ONLY=0|1 (requires ishiiruka-cpu-fast-frame-dump.patch)
EOF
}

REPLAY_JSON=""
ISO=""
OUTPUT_DIR=""
USER_DIR=""
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
TIMEOUT_KILL_AFTER_SECONDS="${TIMEOUT_KILL_AFTER_SECONDS:-5}"
VIDEO_BACKEND="${VIDEO_BACKEND:-OGL}"
DOLPHIN_CPU_CORE="${DOLPHIN_CPU_CORE:-1}"
DOLPHIN_AUDIO_BACKEND="${DOLPHIN_AUDIO_BACKEND:-Null}"
DUMP_FRAMES="${SLIPPI_DUMP_FRAMES:-True}"
INTERNAL_RESOLUTION_FRAME_DUMPS="${SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS:-False}"
EFB_SCALE="${SLIPPI_EFB_SCALE:-2}"
OVERCLOCK_FACTOR="${SLIPPI_OVERCLOCK_FACTOR:-1.0}"
CPU_THREAD="${SLIPPI_CPU_THREAD:-True}"
RENDER_TO_MAIN="${SLIPPI_RENDER_TO_MAIN:-True}"
RENDER_WIDTH="${SLIPPI_RENDER_WIDTH:-960}"
RENDER_HEIGHT="${SLIPPI_RENDER_HEIGHT:-720}"
USE_FFV1="${SLIPPI_USE_FFV1:-True}"
DUMP_CODEC="${SLIPPI_DUMP_CODEC:-}"
DUMP_FORMAT="${SLIPPI_DUMP_FORMAT:-avi}"
BITRATE_KBPS="${SLIPPI_BITRATE_KBPS:-2500}"
DOLPHIN_BIN="${SLIPPI_DOLPHIN_BIN:-/opt/slippi/Slippi Dolphin}"
MAX_RAW_BYTES="${SLIPPI_MAX_RAW_BYTES:-}"
STALL_SECONDS="${SLIPPI_STALL_SECONDS:-0}"
STALL_MIN_FRAME="${SLIPPI_STALL_MIN_FRAME:-}"
USE_XVFB="${USE_XVFB:-1}"
export LD_LIBRARY_PATH="/opt/slippi:${LD_LIBRARY_PATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --replay-json) REPLAY_JSON="$2"; shift 2 ;;
    --iso) ISO="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --user-dir) USER_DIR="$2"; shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --video-backend) VIDEO_BACKEND="$2"; shift 2 ;;
    --cpu-core) DOLPHIN_CPU_CORE="$2"; shift 2 ;;
    --audio-backend) DOLPHIN_AUDIO_BACKEND="$2"; shift 2 ;;
    --no-xvfb) USE_XVFB=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$REPLAY_JSON" || -z "$ISO" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
USER_DIR="${USER_DIR:-$(dirname "$OUTPUT_DIR")/dolphin-user}"
RUN_LOG="$OUTPUT_DIR/render-ffv1.log"
STALL_MARKER="$OUTPUT_DIR/.stall-frame"
mkdir -p "$USER_DIR/Config" "$USER_DIR/Dump/Frames" "$USER_DIR/Dump/Audio" "$USER_DIR/ScreenShots"
rm -f -- "$STALL_MARKER"
find "$USER_DIR/Dump/Frames" -maxdepth 1 -type f \
  \( -name 'framedump*.avi' -o -name 'framedump*.mkv' -o -name 'framedump*.mp4' -o -name 'framedump*.mov' -o -name 'framedump*.nut' \) \
  -delete

TARGET_END_FRAME="$(python3 - <<'PY' "$REPLAY_JSON"
import json
import sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text()).get("endFrame")
except Exception:
    value = None
print(value if isinstance(value, int) else "")
PY
)"
TARGET_STOP_FRAME="$(python3 - <<'PY' "$REPLAY_JSON"
import json
import sys
from pathlib import Path

try:
    replay = json.loads(Path(sys.argv[1]).read_text())
    value = replay.get("stopFrame", replay.get("endFrame"))
except Exception:
    value = None
print(value if isinstance(value, int) else "")
PY
)"

python3 - <<'PY' "$USER_DIR" "$ISO" "$DOLPHIN_CPU_CORE" "$VIDEO_BACKEND" "$DOLPHIN_AUDIO_BACKEND" "$INTERNAL_RESOLUTION_FRAME_DUMPS" "$EFB_SCALE" "$USE_FFV1" "$DUMP_CODEC" "$DUMP_FORMAT" "$BITRATE_KBPS" "$DUMP_FRAMES" "$OVERCLOCK_FACTOR" "$CPU_THREAD" "$RENDER_TO_MAIN" "$RENDER_WIDTH" "$RENDER_HEIGHT"
import configparser
from pathlib import Path
import sys

user = Path(sys.argv[1])
iso = sys.argv[2]
cpu_core = sys.argv[3]
video_backend = sys.argv[4]
audio_backend = sys.argv[5]
interpreter_mode = cpu_core == "0"

def update_ini(path: Path, sections: dict[str, dict[str, str]]) -> None:
    parser = configparser.RawConfigParser(strict=False)
    parser.optionxform = str
    if path.exists():
        parser.read(path)
    for section, updates in sections.items():
        if not parser.has_section(section):
            parser.add_section(section)
        for key, value in updates.items():
            parser.set(section, key, value)
    with path.open("w") as handle:
        parser.write(handle, space_around_delimiters=True)

update_ini(user / "Config" / "Dolphin.ini", {
    "General": {
        "ISOPaths": "1",
        "ISOPath0": str(Path(iso).parent),
        "DumpPath": "",
    },
    "Core": {
        "DefaultISO": iso,
        "EmulationSpeed": "0.00000000",
        "CPUCore": cpu_core,
        "CPUThread": "False" if interpreter_mode else sys.argv[14],
        "Fastmem": "False" if interpreter_mode else "True",
        "GFXBackend": video_backend,
        "Overclock": sys.argv[13],
        "OverclockEnable": "False" if float(sys.argv[13]) == 1.0 else "True",
    },
    "Display": {
        "Fullscreen": "False",
        "RenderToMain": sys.argv[15],
        "RenderWindowWidth": sys.argv[16],
        "RenderWindowHeight": sys.argv[17],
    },
    "Movie": {
        "DumpFrames": sys.argv[12],
        "DumpFramesSilent": "True",
    },
    "DSP": {
        "Backend": audio_backend,
        "DumpAudio": "False",
        "DumpAudioSilent": "True",
    },
    "Interface": {
        "ShowLogWindow": "False",
        "ShowLogConfigWindow": "False",
    },
})

update_ini(user / "Config" / "GFX.ini", {
    "Hardware": {
        # Playback builds default this to true. EmulationSpeed=0 does not override swap interval.
        "VSync": "False",
    },
    "Settings": {
        "ShowFPS": "False",
        "DumpFramesAsImages": "False",
        "InternalResolutionFrameDumps": sys.argv[6],
        "EFBScale": sys.argv[7],
        "UseFFV1": sys.argv[8],
        "DumpCodec": sys.argv[9],
        "DumpFormat": sys.argv[10],
        "BitrateKbps": sys.argv[11],
        "DumpPath": "",
    },
})

(user / "Config" / "Logger.ini").write_text("""\
[Options]
Verbosity = 1
WriteToConsole = True
WriteToFile = True
WriteToWindow = False

[Logs]
MASTER = True
VIDEO = True
BOOT = True
OSREPORT = True
""")
PY

command=(
  "$DOLPHIN_BIN"
  -u "$USER_DIR"
  -i "$REPLAY_JSON"
  -e "$ISO"
  --hide-seekbar
  --cout
  --batch
  -v "$VIDEO_BACKEND"
)

END_WATCHER_PID=""
if [[ "$TARGET_STOP_FRAME" =~ ^-?[0-9]+$ || "$STALL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  (
    progress_frame=""
    progress_at="$SECONDS"
    while true; do
      if [[ -f "$RUN_LOG" ]]; then
        last_frame="$({ grep -a '\[CURRENT_FRAME\]' "$RUN_LOG" 2>/dev/null || true; } | tail -1 | awk '{print $2}')"
        if [[ "$TARGET_STOP_FRAME" =~ ^-?[0-9]+$ \
          && "$last_frame" =~ ^-?[0-9]+$ \
          && "$last_frame" -ge "$TARGET_STOP_FRAME" ]]; then
          echo "[STOP_FRAME_REACHED] $last_frame >= $TARGET_STOP_FRAME" >> "$RUN_LOG"
          sleep "${SLIPPI_END_FRAME_GRACE_SECONDS:-0.25}"
          ps -eo pid=,args= | while read -r pid args; do
            if [[ "$args" == "$DOLPHIN_BIN "* && "$args" == *"$USER_DIR"* && "$args" == *"$REPLAY_JSON"* ]]; then
              kill -TERM "$pid" 2>/dev/null || true
            fi
          done
          exit 0
        fi
        if [[ "$last_frame" =~ ^-?[0-9]+$ ]]; then
          if [[ "$last_frame" != "$progress_frame" ]]; then
            progress_frame="$last_frame"
            progress_at="$SECONDS"
          elif [[ "$STALL_SECONDS" =~ ^[1-9][0-9]*$ \
            && "$STALL_MIN_FRAME" =~ ^-?[0-9]+$ \
            && "$last_frame" -ge "$STALL_MIN_FRAME" \
            && $((SECONDS - progress_at)) -ge "$STALL_SECONDS" ]]; then
            echo "[STALL_FRAME_REACHED] $last_frame after ${STALL_SECONDS}s" >> "$RUN_LOG"
            printf '%s\n' "$last_frame" > "$STALL_MARKER"
            ps -eo pid=,args= | while read -r pid args; do
              if [[ "$args" == "$DOLPHIN_BIN "* && "$args" == *"$USER_DIR"* && "$args" == *"$REPLAY_JSON"* ]]; then
                kill -TERM "$pid" 2>/dev/null || true
              fi
            done
            exit 0
          fi
        fi
      fi
      sleep "${SLIPPI_END_FRAME_POLL_SECONDS:-0.25}"
    done
  ) &
  END_WATCHER_PID="$!"
fi

set +e
if [[ "$MAX_RAW_BYTES" =~ ^[0-9]+$ && "$MAX_RAW_BYTES" -gt 0 ]]; then
  # Linux bash expresses RLIMIT_FSIZE in 1 KiB blocks. Fail one malformed replay instead of
  # allowing an unreachable stop condition to exhaust the whole worker's 10 GB disk.
  ulimit -f "$(( (MAX_RAW_BYTES + 1023) / 1024 ))"
fi
if [[ "$USE_XVFB" == "1" ]]; then
  xvfb-run -a --server-args="-screen 0 960x720x24" timeout --preserve-status --kill-after="$TIMEOUT_KILL_AFTER_SECONDS" "$TIMEOUT_SECONDS" "${command[@]}" > "$RUN_LOG" 2>&1
else
  timeout --preserve-status --kill-after="$TIMEOUT_KILL_AFTER_SECONDS" "$TIMEOUT_SECONDS" "${command[@]}" > "$RUN_LOG" 2>&1
fi
status=$?
set -e

if [[ -n "$END_WATCHER_PID" ]]; then
  kill "$END_WATCHER_PID" 2>/dev/null || true
  wait "$END_WATCHER_PID" 2>/dev/null || true
fi

raw_dir="$USER_DIR/Dump/Frames"
video_path="$(find "$raw_dir" -maxdepth 1 -type f \( -name 'framedump*.avi' -o -name 'framedump*.mkv' -o -name 'framedump*.mp4' -o -name 'framedump*.mov' -o -name 'framedump*.nut' \) | sort | head -1 || true)"
current_frame_count="$(grep -a '\[CURRENT_FRAME\]' "$RUN_LOG" 2>/dev/null | wc -l | tr -d ' ')"
last_current_frame="$(grep -a '\[CURRENT_FRAME\]' "$RUN_LOG" 2>/dev/null | tail -1 | awk '{print $2}')"
last_stall_frame=""
if [[ -f "$STALL_MARKER" ]]; then
  read -r last_stall_frame < "$STALL_MARKER"
fi
if [[ -n "$video_path" ]]; then
  mv -f "$video_path" "$OUTPUT_DIR/"
fi
cp -p "$USER_DIR/Logs/dolphin.log" "$OUTPUT_DIR/dolphin.log" 2>/dev/null || true

if [[ "$status" -ne 0 && -n "$video_path" && "$TARGET_STOP_FRAME" =~ ^-?[0-9]+$ && "$last_current_frame" =~ ^-?[0-9]+$ && "$last_current_frame" -ge "$TARGET_STOP_FRAME" ]]; then
  status=0
fi
if [[ "$status" -ne 0 && -n "$video_path" && "$last_stall_frame" =~ ^-?[0-9]+$ && "$last_stall_frame" == "$last_current_frame" ]]; then
  status=0
fi

python3 - <<'PY' "$OUTPUT_DIR" "$REPLAY_JSON" "$status" "$current_frame_count" "$TIMEOUT_SECONDS" "$VIDEO_BACKEND" "$DOLPHIN_CPU_CORE" "$DOLPHIN_AUDIO_BACKEND" "$TARGET_END_FRAME" "$TARGET_STOP_FRAME" "$USE_FFV1" "$DUMP_CODEC" "$DUMP_FORMAT" "$BITRATE_KBPS" "$INTERNAL_RESOLUTION_FRAME_DUMPS" "$EFB_SCALE" "$DUMP_FRAMES" "$OVERCLOCK_FACTOR" "$CPU_THREAD" "$RENDER_TO_MAIN" "$RENDER_WIDTH" "$RENDER_HEIGHT"
from pathlib import Path
import json
import re
import sys

out_dir = Path(sys.argv[1])
run_log = out_dir / "render-ffv1.log"
current_frames = []
if run_log.exists():
    for line in run_log.read_text(errors="replace").splitlines():
        match = re.search(r"\[CURRENT_FRAME\]\s+(-?\d+)", line)
        if match:
            current_frames.append(int(match.group(1)))
stall_marker = out_dir / ".stall-frame"
try:
    stalled_frame = int(stall_marker.read_text().strip())
except (OSError, ValueError):
    stalled_frame = None
videos = sorted([
    *out_dir.glob("framedump*.avi"),
    *out_dir.glob("framedump*.mkv"),
    *out_dir.glob("framedump*.mp4"),
    *out_dir.glob("framedump*.mov"),
    *out_dir.glob("framedump*.nut"),
])
manifest = {
    "replayJson": sys.argv[2],
    "dolphinExitStatus": int(sys.argv[3]),
    "currentFrameLogEntries": int(sys.argv[4]),
    "currentFrameRange": {
        "first": current_frames[0] if current_frames else None,
        "last": current_frames[-1] if current_frames else None,
    },
    "stalledFrame": stalled_frame,
    "timeoutSeconds": int(sys.argv[5]),
    "videoBackend": sys.argv[6],
    "dolphinCpuCore": sys.argv[7],
    "audioBackend": sys.argv[8],
    "targetEndFrame": int(sys.argv[9]) if sys.argv[9] else None,
    "targetStopFrame": int(sys.argv[10]) if sys.argv[10] else None,
    "useFFV1": sys.argv[11],
    "dumpCodec": sys.argv[12],
    "dumpFormat": sys.argv[13],
    "bitrateKbps": sys.argv[14],
    "internalResolutionFrameDumps": sys.argv[15],
    "efbScale": sys.argv[16],
    "dumpFrames": sys.argv[17],
    "overclockFactor": sys.argv[18],
    "cpuThread": sys.argv[19],
    "renderToMain": sys.argv[20],
    "renderWindow": {"width": int(sys.argv[21]), "height": int(sys.argv[22])},
    "videos": [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in videos
    ],
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

if [[ "$DUMP_FRAMES" == "True" && -z "$video_path" ]]; then
  echo "No FFV1 video dump was written." >&2
  tail -n 80 "$RUN_LOG" >&2 || true
  exit 2
fi

exit "$status"
