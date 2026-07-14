"""Streaming input pipeline for committed codec ``tar.zst`` batches."""

from __future__ import annotations

import json
import os
import queue
import random
import shutil
import tarfile
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, IO, Iterator

import av
import torch
import zstandard
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class ArchiveInfo:
    path: Path
    samples: int
    clips: int | None = None


def archive_index_path(path: Path) -> Path:
    return Path(f"{path}.codec-index.json")


def committed_archives(
    folder: Path,
    *,
    frames: int | None = None,
    stride_frames: int | None = None,
    require_index: bool = False,
) -> list[ArchiveInfo]:
    """Return positive-size archives whose manifest commit record is present."""

    if (frames is None) != (stride_frames is None):
        raise ValueError("frames and stride_frames must be supplied together")

    candidates = (
        [folder] if folder.name.endswith(".tar.zst") else folder.rglob("*.tar.zst")
    )
    archives: list[ArchiveInfo] = []
    for path in sorted(candidates):
        manifest = Path(str(path).removesuffix(".tar.zst") + ".manifest.jsonl")
        if not path.is_file() or not path.stat().st_size or not manifest.is_file():
            continue
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        if not rows or any(row.get("status") != "complete" for row in rows):
            continue
        samples = sum(row.get("artifact") != "skipped" for row in rows)
        if samples:
            clips = None
            index_path = archive_index_path(path)
            if frames is not None and index_path.is_file():
                index = json.loads(index_path.read_text())
                video_frames = index.get("videoFrames")
                if index.get("status") != "complete" or not isinstance(
                    video_frames, list
                ):
                    raise ValueError(f"Invalid codec index: {index_path}")
                if index.get("archiveBytes") != path.stat().st_size:
                    raise ValueError(f"Stale codec index: {index_path}")
                if len(video_frames) != samples:
                    raise ValueError(
                        f"{index_path} has {len(video_frames)} videos; manifest has {samples}"
                    )
                clips = sum(
                    max(0, 1 + (int(count) - frames) // stride_frames)
                    for count in video_frames
                )
            elif require_index:
                raise FileNotFoundError(
                    f"Missing {index_path}; run python -m core.training.codec.index {folder}"
                )
            archives.append(ArchiveInfo(path=path, samples=samples, clips=clips))
    if not archives:
        raise ValueError(
            f"No committed .tar.zst/.manifest.jsonl batch pairs found in {folder}"
        )
    return archives


def split_archives(
    archives: list[ArchiveInfo], train_fraction: float, seed: int, minimum: int = 1
) -> tuple[list[ArchiveInfo], list[ArchiveInfo]]:
    """Deterministically split whole archives so a replay cannot leak across splits."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    if len(archives) < 2 * minimum:
        raise ValueError(
            f"Need at least {2 * minimum} archives for {minimum} per train/eval split"
        )
    shuffled = archives.copy()
    random.Random(seed).shuffle(shuffled)
    split = round(len(shuffled) * train_fraction)
    split = min(len(shuffled) - minimum, max(minimum, split))
    return shuffled[:split], shuffled[split:]


class TarZstdClipDataset(IterableDataset[Tensor]):
    """Decompress each shard once and emit every complete contiguous video clip.

    Archives are divided first across DDP ranks and then across loader workers.
    Training workers cycle their private archive set forever, which prevents
    unequal archive sizes from deadlocking DDP at an epoch boundary.
    """

    VIDEO_SUFFIX = "/video.mp4"

    def __init__(
        self,
        archives: list[ArchiveInfo],
        *,
        frames: int,
        stride_frames: int,
        fps: int,
        size: tuple[int, int],
        training: bool,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 28,
        spool_bytes: int = 64 * 1024**2,
        decoder_threads: int = 1,
        video_prefetch: int = 4,
    ) -> None:
        super().__init__()
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        if len(archives) < world_size:
            raise ValueError(
                f"{len(archives)} archives cannot feed {world_size} distributed ranks"
            )
        self.archives = archives
        self.frames = frames
        self.stride_frames = stride_frames
        self.fps = fps
        self.height, self.width = size
        self.training = training
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.spool_bytes = spool_bytes
        self.decoder_threads = decoder_threads
        self.video_prefetch = video_prefetch
        if self.frames < 1 or self.stride_frames < 1:
            raise ValueError("frames and stride_frames must be positive")
        if self.video_prefetch < 1:
            raise ValueError("video_prefetch must be positive")

    def __iter__(self) -> Iterator[Tensor]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        rank_archives = self.archives[self.rank :: self.world_size]
        assigned = rank_archives[worker_id::workers]
        if not assigned:
            return

        cycle = 0
        while True:
            order = assigned.copy()
            if self.training:
                random.Random(
                    self.seed + 1_000_003 * cycle + 10_007 * self.rank + worker_id
                ).shuffle(order)
            for archive in order:
                yield from self._read_archive(archive.path)
            if not self.training:
                return
            cycle += 1

    def _read_archive(self, path: Path) -> Iterator[Tensor]:
        videos: queue.Queue[_Video | _ProducerFailure | object] = queue.Queue(
            maxsize=self.video_prefetch
        )
        stop = threading.Event()

        def put(item: _Video | _ProducerFailure | object) -> bool:
            while not stop.is_set():
                try:
                    videos.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def produce() -> None:
            try:
                with (
                    path.open("rb", buffering=0) as compressed,
                    zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
                    tarfile.open(fileobj=stream, mode="r|") as archive,
                ):
                    for member in archive:
                        if stop.is_set():
                            break
                        if not member.isfile() or not member.name.endswith(
                            self.VIDEO_SUFFIX
                        ):
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(f"Could not read {member.name}")
                        with source:
                            video = self._spool_video(source)
                        if not put(_Video(member.name, video)):
                            video.close()
                            return
            except BaseException as error:
                put(_ProducerFailure(error))
            finally:
                put(_END)

        producer = threading.Thread(
            target=produce,
            name=f"codec-video-producer-{path.name}",
            daemon=True,
        )
        producer.start()
        try:
            while True:
                item = videos.get()
                if item is _END:
                    break
                if isinstance(item, _ProducerFailure):
                    raise item.error
                assert isinstance(item, _Video)
                with item.file:
                    yield from self._read_video(item.file)
        except Exception as error:
            raise RuntimeError(f"Failed while streaming {path}") from error
        finally:
            stop.set()
            while True:
                try:
                    item = videos.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, _Video):
                    item.file.close()
            producer.join()

    def _spool_video(self, source: BinaryIO) -> IO[bytes]:
        spool_dir = os.environ.get("SMASH_CODEC_SPOOL_DIR")
        video = tempfile.SpooledTemporaryFile(
            max_size=self.spool_bytes, mode="w+b", dir=spool_dir
        )
        try:
            shutil.copyfileobj(source, video, length=4 * 1024**2)
            video.seek(0)
            return video
        except Exception:
            video.close()
            raise

    def _read_video(self, video: BinaryIO) -> Iterator[Tensor]:
        with av.open(video, mode="r", format="mp4") as container:
            stream = container.streams.video[0]
            if (
                stream.average_rate is None
                or abs(float(stream.average_rate) - self.fps) > 1e-6
            ):
                raise ValueError(
                    f"Expected {self.fps} FPS, got {stream.average_rate or 'unknown'}"
                )
            stream.thread_count = self.decoder_threads
            clip: deque[Tensor] = deque(maxlen=self.frames)
            decoded = 0
            for frame in container.decode(stream):
                decoded += 1
                if frame.width != self.width or frame.height != self.height:
                    frame = frame.reformat(
                        width=self.width, height=self.height, format="rgb24"
                    )
                    array = frame.to_ndarray()
                else:
                    array = frame.to_ndarray(format="rgb24")
                clip.append(torch.from_numpy(array).permute(2, 0, 1))
                if (
                    len(clip) == self.frames
                    and (decoded - self.frames) % self.stride_frames == 0
                ):
                    yield torch.stack(tuple(clip))
            if decoded < self.frames:
                raise ValueError(
                    f"Video contains {decoded} frames; expected at least {self.frames}"
                )


@dataclass
class _Video:
    name: str
    file: IO[bytes]


@dataclass
class _ProducerFailure:
    error: BaseException


_END = object()
