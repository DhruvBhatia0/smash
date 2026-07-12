#!/usr/bin/env bash
set -euo pipefail

video="${1:?usage: extract-cfr-frames.sh VIDEO OUTPUT_DIR}"
output_dir="${2:?usage: extract-cfr-frames.sh VIDEO OUTPUT_DIR}"

mkdir -p "$output_dir"
find "$output_dir" -maxdepth 1 -type f -name 'frame_*.png' -delete

# Slippi records a variable-timestamp packet stream. The fps filter restores the
# exact 60 Hz game-frame timeline, duplicating held visuals at missing timestamps.
ffmpeg -hide_banner -loglevel error -y \
  -i "$video" \
  -vf fps=60 \
  -compression_level 1 \
  -start_number 0 \
  "$output_dir/frame_%06d.png"

actual="$(find "$output_dir" -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d ' ')"
expected="$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
  -of default=noprint_wrappers=1:nokey=1 "$video")"
printf 'frames=%s expected=%s\n' "$actual" "$expected"
[[ "$actual" == "$expected" ]]
