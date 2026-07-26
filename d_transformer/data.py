"""Stream aligned 20 Hz video and lossless 60 Hz Slippi controller inputs."""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tarfile
import tempfile
from collections import deque
from pathlib import Path
from typing import BinaryIO, IO, Iterator, NamedTuple

import av
import torch
import zstandard
from av.video.reformatter import Interpolation, VideoReformatter
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from core.training.codec.data import ArchiveInfo


ACTION_FEATURES = (
    56  # 8 analog values + 32 processed-button bits + 16 physical-button bits.
)


class SlpVideoClip(NamedTuple):
    video: Tensor
    actions: Tensor
    sample: str
    archive: str


def parse_slp_actions(
    data: bytes, name: str = "input.slp"
) -> dict[int, dict[int, tuple]]:
    """Return the final leader pre-frame update for every ``(frame, player)``."""

    raw = 15 if data[:1] == b"{" else 0
    length = int.from_bytes(data[11:15], "big") if raw else len(data)
    end = min(len(data), raw + length)
    sizes = {0x36: 0x140, 0x37: 0x6, 0x38: 0x46, 0x39: 0x1}
    actions: dict[int, dict[int, tuple]] = {}
    current_game: dict[int, dict[int, tuple]] | None = None
    game_starts = 0
    position = raw
    while position < end:
        command = data[position]
        if command == 0x35:
            payload = data[position + 1]
            stop = position + payload + 1
            table = data[position + 2 : stop]
            sizes = {0x35: payload}
            for offset in range(0, len(table) - 2, 3):
                sizes[table[offset]] = int.from_bytes(
                    table[offset + 1 : offset + 3], "big"
                )
            position = stop
            continue

        size = sizes.get(command)
        stop = position + (size or 0) + 1
        if size is None or stop > end:
            break
        if command == 0x36:
            game_starts += 1
            current_game = {}
        elif command == 0x37 and size >= 0x3B and not data[position + 6]:
            frame = struct.unpack_from(">i", data, position + 1)[0]
            player = data[position + 5]
            values = (
                *(
                    struct.unpack_from(">f", data, position + offset)[0]
                    for offset in (0x19, 0x1D, 0x21, 0x25, 0x29, 0x33, 0x37)
                ),
                struct.unpack_from(">b", data, position + 0x3B)[0] / 128.0,
                struct.unpack_from(">I", data, position + 0x2D)[0],
                struct.unpack_from(">H", data, position + 0x31)[0],
            )
            if not all(math.isfinite(value) for value in values[:8]):
                raise ValueError(
                    f"{name} has non-finite controller input at frame {frame}"
                )
            (current_game if current_game is not None else actions).setdefault(
                frame, {}
            )[player] = values
        elif command == 0x39 and current_game:
            return current_game  # The renderer also selects the first complete concatenated game.
        position = stop
    if game_starts > 1:
        raise ValueError(f"{name} contains multiple games but no complete game")
    if current_game:
        return current_game
    return actions


def controller_windows(
    actions: dict[int, dict[int, tuple]],
    *,
    first_slp_frame: int,
    first_video_frame: int,
    video_frames: int,
) -> Tensor:
    """Build ``(video transitions, 3 microsteps, 2 players, 56 features)``."""

    first_transition = first_slp_frame + 3 * first_video_frame
    source_frames = range(first_transition + 1, first_transition + 3 * video_frames - 2)
    first = actions.get(source_frames.start)
    players = sorted(first or ())
    if len(players) != 2:
        raise ValueError(f"Expected two leader inputs, got player slots {players}")

    rows = []
    for source_frame in source_frames:
        frame = actions.get(source_frame)
        if frame is None or sorted(frame) != players:
            raise ValueError(f"Missing two-player input at SLP frame {source_frame}")
        player_rows = []
        for player in players:
            *analog, buttons, physical_buttons = frame[player]
            player_rows.append(
                analog
                + [(buttons >> bit) & 1 for bit in range(32)]
                + [(physical_buttons >> bit) & 1 for bit in range(16)]
            )
        rows.append(player_rows)
    return torch.tensor(rows, dtype=torch.float32).reshape(
        video_frames - 1, 3, 2, ACTION_FEATURES
    )


class SlpVideoDataset(IterableDataset[SlpVideoClip]):
    """Sequentially stream archive triplets without unpacking the corpus."""

    def __init__(
        self,
        archives: list[ArchiveInfo],
        *,
        frames: int = 40,
        stride_frames: int = 40,
        size: tuple[int, int] = (208, 252),
        rank: int = 0,
        world_size: int = 1,
        limit: int | None = None,
        clips_per_sample: int | None = None,
        first_clip_frame: int = 0,
        spool_bytes: int = 64 * 1024**2,
    ) -> None:
        super().__init__()
        self.archives = archives
        self.frames = frames
        self.stride_frames = stride_frames
        self.height, self.width = size
        self.rank = rank
        self.world_size = world_size
        self.limit = limit
        self.clips_per_sample = clips_per_sample
        self.first_clip_frame = first_clip_frame
        self.spool_bytes = spool_bytes

    def __iter__(self) -> Iterator[SlpVideoClip]:
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        assigned = self.archives[self.rank :: self.world_size][worker_id::workers]
        emitted = 0
        for archive in assigned:
            for sample in self._read_archive(archive.path):
                yield sample
                emitted += 1
                if self.limit is not None and emitted >= self.limit:
                    return

    def _read_archive(self, path: Path) -> Iterator[SlpVideoClip]:
        bundle: dict[str, object] = {}
        sample_name: str | None = None
        try:
            with (
                path.open("rb", buffering=0) as compressed,
                zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
                tarfile.open(fileobj=stream, mode="r|") as archive,
            ):
                for member in archive:
                    leaf = member.name.rsplit("/", 1)[-1]
                    if not member.isfile() or leaf not in {
                        "input.slp",
                        "video.mp4",
                        "metadata.json",
                    }:
                        continue
                    current = member.name.rsplit("/", 1)[0]
                    if sample_name is not None and current != sample_name:
                        for sample in self._read_sample(sample_name, bundle):
                            yield sample._replace(archive=path.name)
                        bundle = {}
                    sample_name = current
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Could not read {member.name}")
                    with source:
                        if leaf == "video.mp4":
                            bundle[leaf] = self._spool(source)
                        else:
                            bundle[leaf] = source.read()
                if sample_name is not None:
                    for sample in self._read_sample(sample_name, bundle):
                        yield sample._replace(archive=path.name)
        except Exception as error:
            video = bundle.get("video.mp4")
            if hasattr(video, "close"):
                video.close()  # type: ignore[union-attr]
            raise RuntimeError(f"Failed while streaming {path}") from error

    def _spool(self, source: BinaryIO) -> IO[bytes]:
        video = tempfile.SpooledTemporaryFile(
            max_size=self.spool_bytes,
            mode="w+b",
            dir=os.environ.get("SMASH_CODEC_SPOOL_DIR"),
        )
        shutil.copyfileobj(source, video, length=4 * 1024**2)
        video.seek(0)
        return video

    def _read_sample(
        self, name: str, bundle: dict[str, object]
    ) -> Iterator[SlpVideoClip]:
        video = bundle.get("video.mp4")
        if video is None:
            return
        try:
            metadata = json.loads(bundle["metadata.json"])  # type: ignore[arg-type]
            video_info = metadata["video"]
            if (
                video_info["sourceFps"] != 60
                or video_info["targetFps"] != 20
                or video_info["sourceFrameStep"] != 3
            ):
                raise ValueError(f"{name} is not a 60 Hz SLP / 20 FPS video sample")
            actions = parse_slp_actions(bundle["input.slp"], name)  # type: ignore[arg-type]
            yield from self._read_video(
                name, video, actions, int(video_info["firstSelectedSlpFrame"])
            )  # type: ignore[arg-type]
        finally:
            video.close()  # type: ignore[union-attr]

    def _read_video(
        self,
        name: str,
        video: BinaryIO,
        actions: dict[int, dict[int, tuple]],
        first_slp_frame: int,
    ) -> Iterator[SlpVideoClip]:
        with av.open(video, mode="r", format="mp4") as container:
            stream = container.streams.video[0]
            if (
                stream.average_rate is None
                or abs(float(stream.average_rate) - 20) > 1e-6
            ):
                raise ValueError(
                    f"Expected 20 FPS, got {stream.average_rate or 'unknown'}"
                )
            stream.thread_count = 1
            reformatter = VideoReformatter()
            clip: deque[Tensor] = deque(maxlen=self.frames)
            decoded = 0
            emitted = 0
            for frame in container.decode(stream):
                array = reformatter.reformat(
                    frame,
                    width=self.width,
                    height=self.height,
                    format="rgb24",
                    interpolation=Interpolation.AREA,
                ).to_ndarray()
                clip.append(torch.from_numpy(array).permute(2, 0, 1))
                decoded += 1
                first_video_frame = decoded - self.frames
                if (
                    len(clip) == self.frames
                    and first_video_frame >= self.first_clip_frame
                    and first_video_frame % self.stride_frames == 0
                ):
                    yield SlpVideoClip(
                        torch.stack(tuple(clip)),
                        controller_windows(
                            actions,
                            first_slp_frame=first_slp_frame,
                            first_video_frame=first_video_frame,
                            video_frames=self.frames,
                        ),
                        f"{name}@{first_video_frame}",
                        "",
                    )
                    emitted += 1
                    if (
                        self.clips_per_sample is not None
                        and emitted >= self.clips_per_sample
                    ):
                        return
