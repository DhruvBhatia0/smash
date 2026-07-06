#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/dhruv/code/smash/experiments/frame-game-state-data"
SOURCE_DIR="${SOURCE_DIR:-$ROOT/source/Ishiiruka}"
APP_BUNDLE="${APP_BUNDLE:-$SOURCE_DIR/build/Binaries/Slippi Dolphin.app}"
RUNTIME_APP_BUNDLE="${RUNTIME_APP_BUNDLE:-$ROOT/tools/patched-playback/Slippi Dolphin.app}"
RESOURCES_DIR="$APP_BUNDLE/Contents/Resources"
SYS_DIR="$RESOURCES_DIR/Sys"
PLAYBACK_CODES_DIR="$SOURCE_DIR/Data/PlaybackGeckoCodes"
NETPLAY_SYS_DIR="$SOURCE_DIR/Data/Sys"
PLAYBACK_CODES_INI="$SYS_DIR/GameSettings/GALE01r2.ini"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Missing Ishiiruka source dir: $SOURCE_DIR" >&2
  exit 1
fi

if [[ ! -d "$APP_BUNDLE" ]]; then
  echo "Missing built app bundle: $APP_BUNDLE" >&2
  echo "Build it first, then rerun this script." >&2
  exit 1
fi

if [[ ! -d "$NETPLAY_SYS_DIR" ]]; then
  echo "Missing base Sys resources: $NETPLAY_SYS_DIR" >&2
  exit 1
fi

if [[ ! -d "$PLAYBACK_CODES_DIR" ]]; then
  echo "Missing playback Gecko codes: $PLAYBACK_CODES_DIR" >&2
  exit 1
fi

mkdir -p "$RESOURCES_DIR"
ditto "$NETPLAY_SYS_DIR" "$SYS_DIR"
rm -rf "$SYS_DIR/GameSettings"
mkdir -p "$SYS_DIR/GameSettings"
ditto "$PLAYBACK_CODES_DIR" "$SYS_DIR/GameSettings"

if ! grep -q '\$Required: Slippi Playback' "$PLAYBACK_CODES_INI"; then
  echo "Playback Gecko code verification failed: $PLAYBACK_CODES_INI" >&2
  exit 1
fi

rm -rf "$RUNTIME_APP_BUNDLE"
mkdir -p "$(dirname "$RUNTIME_APP_BUNDLE")"
ditto "$APP_BUNDLE" "$RUNTIME_APP_BUNDLE"

echo "Prepared Playback app bundle: $APP_BUNDLE"
echo "Verified playback Gecko codes: $PLAYBACK_CODES_INI"
echo "Updated runtime Playback app bundle: $RUNTIME_APP_BUNDLE"
