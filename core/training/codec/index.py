"""Build fast per-archive frame indexes without decoding video pixels."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import zstandard

from .data import TarZstdClipDataset, archive_index_path, committed_archives


def _video_frames(video) -> int:
    with av.open(video, mode="r", format="mp4") as container:
        stream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        if stream.duration is not None and stream.average_rate is not None:
            return round(
                float(stream.duration * stream.time_base * stream.average_rate)
            )
        if container.duration is not None and stream.average_rate is not None:
            return round(container.duration / av.time_base * float(stream.average_rate))
        raise ValueError("MP4 does not expose its frame count")


def _scan_archive(path: Path, spool_bytes: int) -> tuple[Path, list[int], float]:
    started = time.monotonic()
    frames: list[int] = []
    with (
        path.open("rb", buffering=0) as compressed,
        zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            if not member.isfile() or not member.name.endswith(
                TarZstdClipDataset.VIDEO_SUFFIX
            ):
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {member.name} from {path}")
            with (
                source,
                tempfile.SpooledTemporaryFile(
                    max_size=spool_bytes, mode="w+b"
                ) as video,
            ):
                shutil.copyfileobj(source, video, length=4 * 1024**2)
                video.seek(0)
                frames.append(_video_frames(video))
    return path, frames, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--spool-mb", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.spool_mb < 1:
        parser.error("--workers and --spool-mb must be positive")

    archives = committed_archives(args.data_dir)
    pending = [
        archive
        for archive in archives
        if args.force or not archive_index_path(archive.path).is_file()
    ]
    print(
        f"indexing {len(pending)} of {len(archives)} archives with {args.workers} workers"
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_scan_archive, archive.path, args.spool_mb * 1024**2): archive
            for archive in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            archive = futures[future]
            path, frames, elapsed = future.result()
            if len(frames) != archive.samples:
                raise ValueError(
                    f"{path} contains {len(frames)} videos; manifest has {archive.samples}"
                )
            payload = {
                "schemaVersion": 1,
                "status": "complete",
                "archiveBytes": path.stat().st_size,
                "videoFrames": frames,
            }
            index_path = archive_index_path(path)
            temporary = index_path.with_suffix(index_path.suffix + ".partial")
            temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
            temporary.replace(index_path)
            print(
                f"[{completed}/{len(pending)}] {path.name}: {len(frames)} videos in {elapsed:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
