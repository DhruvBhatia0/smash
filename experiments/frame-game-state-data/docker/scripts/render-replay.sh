#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  render-replay.sh --replay-json <path> --iso <path> --output-dir <path> [options]

Options:
  --user-dir <path>         Dolphin user dir. Defaults to <output-dir>/../dolphin-user.
  --timeout-seconds <n>     Stop Dolphin after n seconds. Defaults to 120.
  --video-backend <name>    Dolphin video backend. Defaults to OGL.
  --cpu-core <n>            Dolphin CPU core. Use 0 for interpreter under emulated Docker.
  --audio-backend <name>    Dolphin DSP audio backend. Defaults to Null.
  --dry-run                 Validate inputs and print the resolved command without rendering.
EOF
}

REPLAY_JSON=""
ISO=""
OUTPUT_DIR=""
USER_DIR=""
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-120}"
VIDEO_BACKEND="${VIDEO_BACKEND:-OGL}"
DOLPHIN_CPU_CORE="${DOLPHIN_CPU_CORE:-1}"
DOLPHIN_AUDIO_BACKEND="${DOLPHIN_AUDIO_BACKEND:-Null}"
DRY_RUN=0
DOLPHIN_BIN="${SLIPPI_DOLPHIN_BIN:-/opt/slippi/Slippi Dolphin}"
export LD_LIBRARY_PATH="/opt/slippi:${LD_LIBRARY_PATH:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --replay-json)
      REPLAY_JSON="$2"
      shift 2
      ;;
    --iso)
      ISO="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --user-dir)
      USER_DIR="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --video-backend)
      VIDEO_BACKEND="$2"
      shift 2
      ;;
    --cpu-core)
      DOLPHIN_CPU_CORE="$2"
      shift 2
      ;;
    --audio-backend)
      DOLPHIN_AUDIO_BACKEND="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$REPLAY_JSON" || -z "$ISO" || -z "$OUTPUT_DIR" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -x "$DOLPHIN_BIN" ]]; then
  echo "Missing Slippi Dolphin binary: $DOLPHIN_BIN" >&2
  exit 1
fi
if [[ ! -f "$REPLAY_JSON" ]]; then
  echo "Missing replay JSON: $REPLAY_JSON" >&2
  exit 1
fi
if [[ ! -f "$ISO" ]]; then
  echo "Missing Melee ISO: $ISO" >&2
  exit 1
fi
if ! grep -q '\$Required: Slippi Playback' /opt/slippi/Sys/GameSettings/GALE01r2.ini; then
  echo "Playback Gecko codes are missing from /opt/slippi/Sys/GameSettings/GALE01r2.ini" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
USER_DIR="${USER_DIR:-$(dirname "$OUTPUT_DIR")/dolphin-user}"
RUN_LOG="$OUTPUT_DIR/render-replay.log"
mkdir -p "$USER_DIR/Config" "$USER_DIR/Dump/Frames" "$USER_DIR/Dump/Audio" "$USER_DIR/ScreenShots"

python3 - <<'PY' "$USER_DIR" "$ISO" "$DOLPHIN_CPU_CORE" "$VIDEO_BACKEND" "$DOLPHIN_AUDIO_BACKEND"
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
        "CPUThread": "False" if interpreter_mode else "True",
        "Fastmem": "False" if interpreter_mode else "True",
        "GFXBackend": video_backend,
    },
    "Display": {
        "Fullscreen": "False",
        "RenderToMain": "True",
        "RenderWindowWidth": "960",
        "RenderWindowHeight": "720",
    },
    "Movie": {
        "DumpFrames": "True",
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
    "Settings": {
        "ShowFPS": "False",
        "DumpFramesAsImages": "True",
        "InternalResolutionFrameDumps": "False",
        "DumpFormat": "avi",
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

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Renderer dry run\n'
  printf 'Dolphin: %s\n' "$DOLPHIN_BIN"
  printf 'Replay JSON: %s\n' "$REPLAY_JSON"
  printf 'ISO: %s\n' "$ISO"
  printf 'Output dir: %s\n' "$OUTPUT_DIR"
  printf 'User dir: %s\n' "$USER_DIR"
  printf 'CPU core: %s\n' "$DOLPHIN_CPU_CORE"
  printf 'Video backend: %s\n' "$VIDEO_BACKEND"
  printf 'Audio backend: %s\n' "$DOLPHIN_AUDIO_BACKEND"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

set +e
xvfb-run -a --server-args="-screen 0 960x720x24" timeout --preserve-status "$TIMEOUT_SECONDS" "${command[@]}" > "$RUN_LOG" 2>&1
status=$?
set -e

raw_dir="$USER_DIR/Dump/Frames"
frame_count="$(find "$raw_dir" -maxdepth 1 -type f -name 'framedump_*.png' 2>/dev/null | wc -l | tr -d ' ')"
current_frame_count="$(grep -a '\[CURRENT_FRAME\]' "$RUN_LOG" 2>/dev/null | wc -l | tr -d ' ')"

if [[ "$frame_count" != "0" ]]; then
  find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'framedump_*.png' -delete
  cp -p "$raw_dir"/framedump_*.png "$OUTPUT_DIR"/
fi
cp -p "$RUN_LOG" "$OUTPUT_DIR/render-replay.log" 2>/dev/null || true
cp -p "$USER_DIR/Logs/dolphin.log" "$OUTPUT_DIR/dolphin.log" 2>/dev/null || true

python3 - <<'PY' "$OUTPUT_DIR" "$REPLAY_JSON" "$ISO" "$status" "$frame_count" "$current_frame_count" "$TIMEOUT_SECONDS" "$VIDEO_BACKEND" "$DOLPHIN_CPU_CORE" "$DOLPHIN_AUDIO_BACKEND"
from pathlib import Path
import json
import re
import struct
import sys

out_dir = Path(sys.argv[1])
frames = sorted(out_dir.glob("framedump_*.png"), key=lambda p: int(p.stem.split("_")[-1]))

def png_size(path: Path):
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return {"width": width, "height": height}

run_log = out_dir / "render-replay.log"
current_frames = []
if run_log.exists():
    for line in run_log.read_text(errors="replace").splitlines():
        match = re.search(r"\[CURRENT_FRAME\]\s+(-?\d+)", line)
        if match:
            current_frames.append(int(match.group(1)))

try:
    replay_config = json.loads(Path(sys.argv[2]).read_text())
except Exception as error:
    replay_config = {"error": f"{type(error).__name__}: {error}"}
requested_end_frame = replay_config.get("endFrame")
completed_requested_range = (
    not isinstance(requested_end_frame, int)
    or bool(current_frames and current_frames[-1] >= requested_end_frame)
)

manifest = {
    "replayJson": sys.argv[2],
    "replayConfig": replay_config,
    "iso": sys.argv[3],
    "dolphinExitStatus": int(sys.argv[4]),
    "timeoutSeconds": int(sys.argv[7]),
    "videoBackend": sys.argv[8],
    "dolphinCpuCore": sys.argv[9],
    "audioBackend": sys.argv[10],
    "frameCount": len(frames),
    "currentFrameLogEntries": len(current_frames),
    "currentFrameRange": {
        "first": current_frames[0] if current_frames else None,
        "last": current_frames[-1] if current_frames else None,
    },
    "requestedEndFrame": requested_end_frame,
    "completedRequestedRange": completed_requested_range,
    "samples": [
        {
            "path": str(frames[index]),
            "dumpFrameNumber": int(frames[index].stem.split("_")[-1]),
            "png": png_size(frames[index]),
        }
        for index in sorted(set(i for i in (0, len(frames) // 2, len(frames) - 1) if 0 <= i < len(frames)))
    ],
    "logs": {
        "run": str(out_dir / "render-replay.log"),
        "dolphin": str(out_dir / "dolphin.log"),
    },
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

if [[ "$frame_count" == "0" ]]; then
  echo "No frame PNGs were written." >&2
  tail -n 80 "$RUN_LOG" >&2 || true
  exit 2
fi
if [[ "$current_frame_count" == "0" ]]; then
  echo "Frame PNGs exist, but no CURRENT_FRAME entries were logged." >&2
  tail -n 80 "$RUN_LOG" >&2 || true
  exit 3
fi
if ! python3 - <<'PY' "$OUTPUT_DIR/manifest.json"
from pathlib import Path
import json
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
if not manifest.get("completedRequestedRange", False):
    print(
        "Render stopped before requested end frame: "
        f"last={manifest.get('currentFrameRange', {}).get('last')} "
        f"requested={manifest.get('requestedEndFrame')}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
then
  tail -n 80 "$RUN_LOG" >&2 || true
  exit 4
fi

exit 0
