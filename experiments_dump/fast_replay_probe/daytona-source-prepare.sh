#!/usr/bin/env bash
set -euo pipefail

root="/home/daytona/slippi-source-probe"
mkdir -p "$root"

cat /tmp/melee.part.?? > "$root/melee.iso"
cp /tmp/realtimeTest.slp "$root/realtimeTest.slp"
cp /tmp/daytona-source-playback.json "$root/playback.json"

test "$(stat -c %s "$root/melee.iso")" = "1459978240"
test "$(sha256sum "$root/melee.iso" | awk '{print $1}')" = \
  "0de05981a34156b9cedcef73c73d4244ac05cf6149ab3c9cfed917698819e464"
test "$(sha256sum "$root/realtimeTest.slp" | awk '{print $1}')" = \
  "387311207f966e8ca14f6ccd6b271974f8a0c0e846825fe587d75650f2e45a97"

chmod 755 \
  /tmp/render-ffv1-replay.sh \
  /tmp/daytona-source-benchmark.sh \
  /tmp/daytona-source-launch.sh

printf 'iso_bytes=%s\n' "$(stat -c %s "$root/melee.iso")"
printf 'iso_sha256=%s\n' "$(sha256sum "$root/melee.iso" | awk '{print $1}')"
printf 'slp_sha256=%s\n' "$(sha256sum "$root/realtimeTest.slp" | awk '{print $1}')"
