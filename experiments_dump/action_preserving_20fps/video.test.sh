#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/action-preserving-20fps.XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

source_video="$tmp_dir/source.mkv"
source_md5="$tmp_dir/source.framemd5"
aligned_video="$tmp_dir/aligned.mkv"
partial_video="$tmp_dir/partial.mkv"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i testsrc2=size=64x64:rate=60 \
  -frames:v 12 -c:v ffv1 "$source_video"
ffmpeg -hide_banner -loglevel error -i "$source_video" -f framemd5 "$source_md5"

frame_hashes() {
  ffmpeg -hide_banner -loglevel error -i "$1" -f framemd5 - |
    awk -F, '!/^#/ { gsub(/ /, "", $6); print $6 }'
}

expected_hashes() {
  awk -F, -v indices=" $1 " '!/^#/ {
    gsub(/ /, "", $3);
    gsub(/ /, "", $6);
    if (index(indices, " " $3 " ") != 0) print $6;
  }' "$source_md5"
}

packet_pts() {
  ffprobe -v error -select_streams v:0 -show_entries packet=pts_time \
    -of csv=p=0 "$1" | paste -sd, -
}

# Non-zero phase, aligned endpoint: exact source indices 2,5,8,11 and true CFR 20 metadata.
"$script_dir/downsample-video-vfr.sh" "$source_video" "$aligned_video" 2 11 3
[[ $(frame_hashes "$aligned_video") == "$(expected_hashes "2 5 8 11")" ]] || {
  echo "aligned output pixels do not match manually enumerated source frames" >&2
  exit 1
}
[[ $(packet_pts "$aligned_video") == "0.000000,0.050000,0.100000,0.150000" ]] || {
  echo "aligned output PTS are not CFR 20 from zero" >&2
  exit 1
}
aligned_rate=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate \
  -of default=nw=1:nk=1 "$aligned_video")
[[ $aligned_rate == "20/1" ]] || {
  echo "aligned output rate is $aligned_rate, expected 20/1" >&2
  exit 1
}

# Unaligned endpoint: exact source indices 0,3,6,9,10 with a final 1/60 s interval.
"$script_dir/downsample-video-vfr.sh" "$source_video" "$partial_video" 0 10 3
[[ $(frame_hashes "$partial_video") == "$(expected_hashes "0 3 6 9 10")" ]] || {
  echo "partial output pixels do not match manually enumerated source frames" >&2
  exit 1
}
[[ $(packet_pts "$partial_video") == "0.000000,0.050000,0.100000,0.150000,0.167000" ]] || {
  echo "partial output PTS do not preserve the exact short tail" >&2
  exit 1
}

echo "video tests passed: exact phase, pixels, endpoint, PTS, and CFR metadata"
