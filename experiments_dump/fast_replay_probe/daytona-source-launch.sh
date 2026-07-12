#!/usr/bin/env bash
set -euo pipefail

label="${1:?label is required}"
nohup /tmp/daytona-source-benchmark.sh "$label" \
  > "/home/daytona/slippi-source-probe/${label}.benchmark.log" 2>&1 &
echo "$!"
