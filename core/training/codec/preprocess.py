"""Build the 20 FPS, 252x208 codec dataset directly from source AVIs.

Run one deterministic shard per CPU worker machine.  Shards are balanced by
source byte size and only include videos absent from the target dataset:

    python -m core.training.codec.preprocess --shards 4 --shard 0 --workers 12
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import av
import httpx
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

SOURCE_REPO = "DhruvBhatia0/smash-battlefield-fox"
SOURCE_REVISION = "2c94351b82c2c65a31fb39fe52a34ff905b6abcf"
TARGET_REPO = "DhruvBhatia0/smash-battlefield-fox-codec-20fps"
TARGET_FPS = 20
WIDTH, HEIGHT = 252, 208


@dataclass(frozen=True)
class RemoteVideo:
    path: str
    size: int

    @property
    def video_id(self) -> str:
        return PurePosixPath(self.path).parent.name

    @property
    def output_path(self) -> str:
        return f"slp_with_video/{self.video_id}/video.nut"


@dataclass(frozen=True)
class Config:
    shard: int
    shards: int
    workers: int
    work_dir: Path
    upload: bool


class HTTPRangeReader:
    """Seekable HTTP reader that keeps only a small LRU cache in memory."""

    chunk_size = 8 * 1024 * 1024
    cache_size = 4

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.client = httpx.Client(follow_redirects=True, timeout=120)

    def __enter__(self) -> HTTPRangeReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.client.close()

    def read(self, count: int = -1) -> bytes:
        count = self.size - self.position if count < 0 else count
        count = min(count, self.size - self.position)
        data = bytearray()
        while count:
            chunk = self._chunk(self.position // self.chunk_size)
            offset = self.position % self.chunk_size
            piece = chunk[offset : offset + count]
            data.extend(piece)
            self.position += len(piece)
            count -= len(piece)
        return bytes(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 1:
            offset += self.position
        elif whence == 2:
            offset += self.size
        if not 0 <= offset <= self.size:
            raise ValueError(f"Invalid seek to {offset}")
        self.position = offset
        return offset

    def tell(self) -> int:
        return self.position

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def _chunk(self, index: int) -> bytes:
        if index in self.cache:
            data = self.cache.pop(index)
            self.cache[index] = data
            return data
        start = index * self.chunk_size
        end = min(start + self.chunk_size, self.size) - 1
        response = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
        response.raise_for_status()
        data = response.content
        if response.status_code != 206 or len(data) != end - start + 1:
            raise OSError(f"Unexpected range response for {self.url}")
        self.cache[index] = data
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return data


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/codec-data"))
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    return Config(
        shard=args.shard,
        shards=args.shards,
        workers=args.workers,
        work_dir=args.work_dir,
        upload=not args.skip_upload,
    )


def processed_ids() -> set[str]:
    index_path = hf_hub_download(
        repo_id=TARGET_REPO,
        repo_type="dataset",
        filename=".codec-video-index.json",
        token=True,
    )
    return {
        PurePosixPath(row["path"]).parent.name
        for row in json.loads(Path(index_path).read_text())
    }


def assigned_videos(config: Config) -> list[RemoteVideo]:
    api = HfApi()
    done = processed_ids()
    videos = sorted(
        (
            RemoteVideo(entry.path, entry.size)
            for entry in api.list_repo_tree(
                SOURCE_REPO,
                repo_type="dataset",
                revision=SOURCE_REVISION,
                recursive=True,
            )
            if entry.path.endswith("/video.avi")
            and PurePosixPath(entry.path).parent.name not in done
        ),
        key=lambda video: video.size,
        reverse=True,
    )
    shards: list[list[RemoteVideo]] = [[] for _ in range(config.shards)]
    sizes = [0] * config.shards
    for video in videos:
        shard = min(range(config.shards), key=sizes.__getitem__)
        shards[shard].append(video)
        sizes[shard] += video.size
    print(
        f"{len(videos):,} remaining videos / {sum(video.size for video in videos):,} bytes; "
        f"shard {config.shard}: {len(shards[config.shard]):,} videos / "
        f"{sizes[config.shard]:,} bytes"
    )
    return shards[config.shard]


def convert(video: RemoteVideo, config: Config) -> dict[str, int | str]:
    output = config.work_dir / "output" / video.output_path
    marker = config.work_dir / "progress" / f"{video.video_id}.json"
    if marker.exists():
        return json.loads(marker.read_text())

    source = hf_hub_url(
        SOURCE_REPO,
        video.path,
        repo_type="dataset",
        revision=SOURCE_REVISION,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    with (
        HTTPRangeReader(source, video.size) as reader,
        av.open(reader) as input_container,
        av.open(
            output, "w", format="nut"
        ) as output_container,
    ):
        input_stream = input_container.streams.video[0]
        source_fps = float(input_stream.average_rate)
        stride = round(source_fps / TARGET_FPS)
        if stride * TARGET_FPS != round(source_fps):
            raise ValueError(f"{video.path} is {source_fps:g} FPS, not divisible by 20")
        output_stream = output_container.add_stream("ffv1", rate=TARGET_FPS)
        output_stream.width, output_stream.height = WIDTH, HEIGHT
        output_stream.pix_fmt = "gbrp"
        for index, frame in enumerate(input_container.decode(input_stream)):
            if index % stride:
                continue
            resized = frame.reformat(width=WIDTH, height=HEIGHT, format="rgb24")
            output_container.mux(output_stream.encode(resized))
            frames += 1
        output_container.mux(output_stream.encode())
    row = {"path": video.output_path, "frames": frames, "fps": float(TARGET_FPS)}
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(row))
    return row


def upload(config: Config, rows: list[dict[str, int | str]]) -> None:
    output = config.work_dir / "output"
    manifest = output / "manifests" / f"shard-{config.shard:03d}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    if config.upload:
        HfApi().upload_large_folder(
            repo_id=TARGET_REPO, repo_type="dataset", folder_path=output
        )


def main() -> None:
    config = parse_args()
    videos = assigned_videos(config)
    rows: list[dict[str, int | str]] = []
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(convert, video, config): video for video in videos}
        for count, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"{count:,}/{len(videos):,}: {row['path']}", flush=True)
    rows.sort(key=lambda row: str(row["path"]))
    upload(config, rows)


if __name__ == "__main__":
    main()
