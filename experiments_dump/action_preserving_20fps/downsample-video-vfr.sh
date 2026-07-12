#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: downsample-video-vfr.sh INPUT OUTPUT FIRST_INDEX LAST_INDEX [STEP]

Normalize the source timeline to 60 FPS, select FIRST_INDEX, every third frame
thereafter, and the exact LAST_INDEX, then losslessly encode as FFV1. Aligned
output is conventional CFR 20 FPS. An unaligned exact endpoint uses a short VFR tail.
EOF
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

input=$1
output=$2
first_index=$3
last_index=$4
step=${5:-3}

for value in "$first_index" "$last_index" "$step"; do
  [[ $value =~ ^[0-9]+$ ]] || { echo "indices and step must be non-negative integers" >&2; exit 2; }
done
(( step > 0 )) || { echo "step must be positive" >&2; exit 2; }
(( step == 3 )) || { echo "v1 requires STEP=3 for 60 -> 20 FPS" >&2; exit 2; }
(( first_index <= last_index )) || { echo "FIRST_INDEX must be <= LAST_INDEX" >&2; exit 2; }

mkdir -p "$(dirname "$output")"
select_expr="between(n\\,${first_index}\\,${last_index})*not(mod(n-${first_index}\\,${step}))+eq(n\\,${last_index})"
remainder=$(( (last_index - first_index) % step ))
filter="fps=60,select='${select_expr}',setpts=PTS-STARTPTS"
fps_mode=vfr
if (( remainder == 0 )); then
  filter="${filter},fps=20:eof_action=pass"
  fps_mode=cfr
fi

ffmpeg -hide_banner -loglevel error -y \
  -i "$input" \
  -vf "$filter" \
  -fps_mode "$fps_mode" \
  -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -slicecrc 1 \
  -an "$output"

expected=$(( (last_index - first_index) / step + 1 ))
if (( (last_index - first_index) % step != 0 )); then
  expected=$(( expected + 1 ))
fi
actual=$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of default=nw=1:nk=1 "$output")
[[ $actual == "$expected" ]] || {
  echo "frame count mismatch: expected $expected, got $actual" >&2
  exit 1
}

first_pts=$(ffprobe -v error -select_streams v:0 -show_entries packet=pts_time \
  -of csv=p=0 "$output" | sed -n '1p')
[[ $first_pts == "0.000000" ]] || {
  echo "output PTS must start at zero, got $first_pts" >&2
  exit 1
}

if (( remainder == 0 )); then
  frame_rate=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate \
    -of default=nw=1:nk=1 "$output")
  [[ $frame_rate == "20/1" ]] || {
    echo "aligned output must advertise 20 FPS, got $frame_rate" >&2
    exit 1
  }
fi

echo "wrote $actual lossless observation frames ($fps_mode) to $output"
