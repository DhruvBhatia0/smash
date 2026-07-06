#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/dhruv/code/smash/experiments/frame-game-state-data"
PATCHED_PLAYBACK_APP="$ROOT/tools/patched-playback/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin"
if [[ -z "${PLAYBACK_APP:-}" && -x "$PATCHED_PLAYBACK_APP" ]]; then
  PLAYBACK_APP="$PATCHED_PLAYBACK_APP"
else
  PLAYBACK_APP="${PLAYBACK_APP:-/Applications/Slippi Playback Dolphin.app/Contents/MacOS/Slippi Dolphin}"
fi
ISO="/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"
REPLAY_JSON="${1:-$ROOT/replay-playback.capture-test.json}"
REPLAY_NAME="$(basename "${REPLAY_JSON%.*}")"
USER_DIR="${USER_DIR:-$ROOT/playback-debug-user}"
FRAME_OUTPUT_DIR="${FRAME_OUTPUT_DIR:-$ROOT/frames/$REPLAY_NAME}"
LOG_DIR="$ROOT/logs"
RUN_LOG="$LOG_DIR/render-replay-debug.log"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-90}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PLAYBACK_APP" ]]; then
  echo "Missing Playback Dolphin app: $PLAYBACK_APP" >&2
  echo "Expected the official Ishiiruka-Playback build, not the Homebrew netplay build." >&2
  exit 1
fi

if [[ ! -f "$ISO" ]]; then
  echo "Missing Melee ISO: $ISO" >&2
  exit 1
fi

if [[ ! -f "$REPLAY_JSON" ]]; then
  echo "Missing replay JSON: $REPLAY_JSON" >&2
  exit 1
fi

APP_CONTENTS_DIR="$(cd "$(dirname "$PLAYBACK_APP")/.." && pwd)"
PLAYBACK_CODES_INI="$APP_CONTENTS_DIR/Resources/Sys/GameSettings/GALE01r2.ini"
if [[ ! -f "$PLAYBACK_CODES_INI" ]] || ! grep -q '\$Required: Slippi Playback' "$PLAYBACK_CODES_INI"; then
  echo "Playback app is missing playback Gecko codes: $PLAYBACK_CODES_INI" >&2
  echo "This would boot Online Play and dump menu frames instead of replay frames." >&2
  echo "For the local source build, run: $ROOT/scripts/prepare-playback-build.sh" >&2
  exit 1
fi

pkill -f "/Applications/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin" 2>/dev/null || true
pkill -f "/Applications/Slippi Playback Dolphin.app/Contents/MacOS/Slippi Dolphin" 2>/dev/null || true
pkill -f "$PLAYBACK_APP" 2>/dev/null || true
sleep 1

rm -rf "$USER_DIR"
mkdir -p "$USER_DIR/Config" "$USER_DIR/Dump/Frames" "$USER_DIR/Dump/Audio" "$USER_DIR/ScreenShots"

# Seed from the real playback profile if Slippi already created it; otherwise seed from
# the experiment profile that was created during earlier boot tests.
SOURCE_USER="$HOME/Library/Application Support/com.project-slippi.dolphin/playback/User"
if [[ -d "$SOURCE_USER/Config" ]]; then
  ditto "$SOURCE_USER/Config" "$USER_DIR/Config"
elif [[ -d "$ROOT/playback-user/Config" ]]; then
  ditto "$ROOT/playback-user/Config" "$USER_DIR/Config"
fi

cat > "$USER_DIR/Config/Logger.ini" <<'EOF'
[LogWindow]
x = 400
y = 652
pos = 2

[Options]
Font = 0
WrapLines = False
Verbosity = 1
WriteToConsole = True
WriteToFile = True
WriteToWindow = True

[Logs]
ActionReplay = True
AI = True
Audio = True
BOOT = True
COMMON = True
CONSOLE = True
CORE = True
CP = True
DIO = True
DSP = True
DSPHLE = True
DSPLLE = True
DSPMails = True
FileMon = True
IOS = True
MASTER = True
MEMMAP = True
OSREPORT = True
PAD = True
PROCESSORINTERFACE = True
VIDEO = True
VIDEOINTERFACE = True
WII_IPC = True
Wiimote = True
EOF

python3 - <<'PY' "$USER_DIR" "$ISO"
import configparser
from pathlib import Path
import sys

user = Path(sys.argv[1])
iso = sys.argv[2]

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
        "ISOPath0": "/Users/dhruv/Downloads",
        "DumpPath": "",
    },
    "Core": {
        "DefaultISO": iso,
        "EmulationSpeed": "0.00000000",
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
        "DumpAudio": "False",
        "DumpAudioSilent": "True",
    },
    "Interface": {
        "ShowLogWindow": "True",
        "ShowLogConfigWindow": "True",
    },
})

gfx = user / "Config" / "GFX.ini"
update_ini(gfx, {
    "Settings": {
        "ShowFPS": "False",
        "DumpFramesAsImages": "True",
        "InternalResolutionFrameDumps": "False",
        "UseFFV1": "True",
        "DumpFormat": "avi",
        "DumpPath": "",
    },
})
PY

echo "Playback app: $PLAYBACK_APP"
echo "Playback Gecko codes: $PLAYBACK_CODES_INI"
echo "User dir: $USER_DIR"
echo "Replay JSON: $REPLAY_JSON"
echo "Run log: $RUN_LOG"
echo "Expected video dump dir: $USER_DIR/Dump/Frames"
echo "Durable frame output dir: $FRAME_OUTPUT_DIR"
echo "Expected Dolphin log: $USER_DIR/Logs/dolphin.log"

set +e
"$PLAYBACK_APP" \
  -u "$USER_DIR" \
  -i "$REPLAY_JSON" \
  -e "$ISO" \
  --hide-seekbar \
  --cout \
  --batch \
  > "$RUN_LOG" 2>&1 &
pid=$!

echo "$pid" > "$LOG_DIR/render-replay-debug.pid"

deadline=$((SECONDS + TIMEOUT_SECONDS))
timed_out=false
while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do
  sleep 1
done

if kill -0 "$pid" 2>/dev/null; then
  echo "Timeout after ${TIMEOUT_SECONDS}s; stopping Dolphin." | tee -a "$RUN_LOG"
  timed_out=true
  kill "$pid" 2>/dev/null || true
  sleep 2
fi

wait "$pid" 2>/dev/null
status=$?
set -e

echo "Dolphin exit status: $status"
echo
echo "Media files:"
frame_count="$(find "$USER_DIR/Dump/Frames" -maxdepth 1 -type f -name 'framedump_*.png' 2>/dev/null | wc -l | tr -d ' ')"
current_frame_count="$(grep -a '\[CURRENT_FRAME\]' "$RUN_LOG" 2>/dev/null | wc -l | tr -d ' ')"
echo "Frame PNGs in raw dump: $frame_count"
echo "Playback current-frame log entries: $current_frame_count"
result_status=0

if [[ "$frame_count" != "0" ]]; then
  mkdir -p "$FRAME_OUTPUT_DIR"
  find "$FRAME_OUTPUT_DIR" -maxdepth 1 -type f -name 'framedump_*.png' -delete
  ditto "$USER_DIR/Dump/Frames" "$FRAME_OUTPUT_DIR"
  cp -p "$RUN_LOG" "$FRAME_OUTPUT_DIR/render-replay-debug.log" 2>/dev/null || true
  cp -p "$USER_DIR/Logs/dolphin.log" "$FRAME_OUTPUT_DIR/dolphin.log" 2>/dev/null || true

  python3 - <<'PY' "$FRAME_OUTPUT_DIR" "$REPLAY_JSON" "$PLAYBACK_APP" "$ISO" "$status" "$timed_out" "$USER_DIR" "$RUN_LOG" "$TIMEOUT_SECONDS" "$PLAYBACK_CODES_INI"
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

run_log = Path(sys.argv[8])
current_frames = []
if run_log.exists():
    for line in run_log.read_text(errors="replace").splitlines():
        match = re.search(r"\[CURRENT_FRAME\]\s+(-?\d+)", line)
        if match:
            current_frames.append(int(match.group(1)))

try:
    replay_config = json.loads(Path(sys.argv[2]).read_text())
except Exception:
    replay_config = None

sample_indices = sorted(set(index for index in (0, len(frames) // 2, len(frames) - 1) if 0 <= index < len(frames)))
manifest = {
    "replayJson": sys.argv[2],
    "replayConfig": replay_config,
    "playbackApp": sys.argv[3],
    "playbackCodesIni": sys.argv[10],
    "iso": sys.argv[4],
    "dolphinExitStatus": int(sys.argv[5]),
    "timedOut": sys.argv[6] == "true",
    "timeoutSeconds": int(sys.argv[9]),
    "frameCount": len(frames),
    "frameDumpNumbers": {
        "first": int(frames[0].stem.split("_")[-1]) if frames else None,
        "last": int(frames[-1].stem.split("_")[-1]) if frames else None,
    },
    "currentFrameLogEntries": len(current_frames),
    "currentFrameRange": {
        "first": current_frames[0] if current_frames else None,
        "last": current_frames[-1] if current_frames else None,
    },
    "samples": [
        {
            "path": str(frames[index]),
            "dumpFrameNumber": int(frames[index].stem.split("_")[-1]),
            "png": png_size(frames[index]),
        }
        for index in sample_indices
    ],
    "firstFrame": str(frames[0]) if frames else None,
    "lastFrame": str(frames[-1]) if frames else None,
    "rawDumpDir": str(Path(sys.argv[7]) / "Dump" / "Frames"),
    "logs": {
        "run": str(out_dir / "render-replay-debug.log"),
        "dolphin": str(out_dir / "dolphin.log"),
    },
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

  echo "Copied frames to: $FRAME_OUTPUT_DIR"
  echo "Sample copied frames:"
  find "$FRAME_OUTPUT_DIR" -maxdepth 1 -type f -name 'framedump_*.png' | sort -V | sed -n '1p;300p;600p;$p'
  if [[ "$current_frame_count" == "0" ]]; then
    echo "Frame PNGs exist, but Playback Dolphin did not log CURRENT_FRAME. Treating this as a failed replay dump." >&2
    result_status=3
  fi
else
  echo "No frame PNGs were written."
  result_status=2
fi
echo
echo "Run log tail:"
tail -n 80 "$RUN_LOG" || true
echo
echo "Dolphin log tail:"
tail -n 80 "$USER_DIR/Logs/dolphin.log" 2>/dev/null || true

exit "$result_status"
