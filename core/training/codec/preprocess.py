"""Build the 20 FPS, 252x208 codec dataset directly from source AVIs.

Run one deterministic shard per CPU worker machine.  Shards are balanced by
source byte size and only include videos absent from the target dataset:

    python -m core.training.codec.preprocess --shards 4 --shard 0 --workers 12
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import av
from huggingface_hub import HfApi, hf_hub_download

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
    input_dir = config.work_dir / "input" / video.video_id
    output = config.work_dir / "output" / video.output_path
    if output.exists():
        with av.open(output) as container:
            stream = container.streams.video[0]
            return {
                "path": video.output_path,
                "frames": stream.frames,
                "fps": float(stream.average_rate),
            }

    source = hf_hub_download(
        repo_id=SOURCE_REPO,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        filename=video.path,
        local_dir=input_dir,
        token=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.nut")
    frames = 0
    try:
        with av.open(source) as input_container, av.open(
            temporary, "w", format="nut"
        ) as output_container:
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
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(input_dir, ignore_errors=True)
    return {"path": video.output_path, "frames": frames, "fps": float(TARGET_FPS)}


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
