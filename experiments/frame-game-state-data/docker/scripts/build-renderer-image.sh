#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-smash/slippi-renderer:local}"
PLATFORM="${PLATFORM:-linux/amd64}"
BUILD_JOBS="${BUILD_JOBS:-2}"

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

docker build \
  --progress=plain \
  --platform "$PLATFORM" \
  --build-arg BUILD_JOBS="$BUILD_JOBS" \
  -f "$ROOT/docker/slippi-renderer/Dockerfile" \
  -t "$IMAGE_TAG" \
  "$ROOT"

echo "$IMAGE_TAG"
