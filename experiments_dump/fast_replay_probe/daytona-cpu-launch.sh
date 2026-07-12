#!/usr/bin/env bash
set -euo pipefail

mode="${1:?mode is required}"
name="${2:?result name is required}"
threads="${3:-4}"
overclock="${4:-1.0}"
cpu_thread="${5:-True}"
nohup /tmp/daytona-cpu-benchmark.sh "$mode" "$name" "$threads" "$overclock" "$cpu_thread" \
  > "/tmp/${name}.benchmark.log" 2>&1 &
echo "$!"
