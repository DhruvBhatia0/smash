#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: daytona-source-switch.sh baseline|queue|dump-only}"
patch_root="${2:-/tmp}"
source_root="/home/daytona/Ishiiruka"
build_root="$source_root/build-headless"
base_commit="e7711b104b339a99385f2bb12b472d46140a7bc7"

git -C "$source_root" reset --hard "$base_commit"

case "$variant" in
  baseline)
    git -C "$source_root" apply "$patch_root/ishiiruka-existing.patch"
    ;;
  queue)
    git -C "$source_root" apply "$patch_root/ishiiruka-queue.patch"
    git -C "$source_root" apply "$patch_root/ishiiruka-queue-race.patch"
    ;;
  dump-only)
    git -C "$source_root" apply "$patch_root/ishiiruka-queue.patch"
    git -C "$source_root" apply "$patch_root/ishiiruka-queue-race.patch"
    git -C "$source_root" apply "$patch_root/ishiiruka-dump-only.patch"
    ;;
  *)
    echo "unknown source variant: $variant" >&2
    exit 2
    ;;
esac

git -C "$source_root" apply "$patch_root/ishiiruka-memfd.patch"
git -C "$source_root" diff --check
cmake --build "$build_root" --target dolphin-emu-nogui -- -j6

sudo install -m 0755 \
  "$build_root/Binaries/dolphin-emu-nogui" \
  "/opt/slippi/dolphin-emu-nogui-$variant"
sudo cp "/opt/slippi/dolphin-emu-nogui-$variant" /opt/slippi/dolphin-emu-nogui

git -C "$source_root" status --short
sha256sum "/opt/slippi/dolphin-emu-nogui-$variant"
