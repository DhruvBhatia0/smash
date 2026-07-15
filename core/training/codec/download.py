"""Copy committed codec shards from Google Drive with rclone."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


REMOTE = (
    "smash-drive:hal-fox-captain-falcon-battlefield/"
    "recordings-642x528-20fps-slippi-pts-v4/batches"
)


def _run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout or ""


def _filters(names: list[str]) -> list[str]:
    filters: list[str] = []
    for name in names:
        stem = name.removesuffix(".tar.zst")
        filters.extend(("--filter", f"+ /{stem}.tar.zst"))
        filters.extend(("--filter", f"+ /{stem}.manifest.jsonl"))
    return [*filters, "--filter", "- *"] if filters else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--remote", default=REMOTE)
    parser.add_argument("--transfers", type=int, default=8)
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="Copy only this archive and its manifest; repeat for smoke tests",
    )
    args = parser.parse_args()
    if args.transfers < 1:
        parser.error("--transfers must be positive")

    filters = _filters(args.archive)
    remote = json.loads(
        _run(["rclone", "size", args.remote, "--json", *filters], capture=True)
    )
    args.destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.destination).free
    local = sum(
        path.stat().st_size
        for suffix in ("*.tar.zst", "*.manifest.jsonl")
        for path in args.destination.glob(suffix)
    )
    if free + local < remote["bytes"]:
        raise OSError(
            f"Need {remote['bytes']:,} total bytes; destination has "
            f"{local:,} downloaded and {free:,} free"
        )

    _run(
        [
            "rclone",
            "copy",
            args.remote,
            str(args.destination),
            "--transfers",
            str(args.transfers),
            "--progress",
            *filters,
        ]
    )
    archives = sorted(args.destination.glob("*.tar.zst"))
    selected = {
        name.removesuffix(".tar.zst") for name in args.archive
    } or {path.name.removesuffix(".tar.zst") for path in archives}
    missing = [
        stem
        for stem in sorted(selected)
        if not (args.destination / f"{stem}.tar.zst").is_file()
        or not (args.destination / f"{stem}.manifest.jsonl").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing archive/manifest pairs: {missing}")
    if remote["count"] != 2 * len(selected):
        raise ValueError(
            f"Remote contains {remote['count']} files for {len(selected)} archive pairs"
        )
    copied = sum(
        (args.destination / f"{stem}.tar.zst").stat().st_size
        + (args.destination / f"{stem}.manifest.jsonl").stat().st_size
        for stem in selected
    )
    if copied != remote["bytes"]:
        raise ValueError(f"Copied {copied:,} bytes; remote contains {remote['bytes']:,}")
    print(f"verified {len(selected)} archive/manifest pairs ({copied:,} bytes)")


if __name__ == "__main__":
    main()
