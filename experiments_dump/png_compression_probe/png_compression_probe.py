#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from PIL import Image


DEFAULT_FRAMES_DIR = (
    Path(__file__).resolve().parents[1]
    / "fast_replay_probe/runs/local-full-baseline-20260709T025734Z/dolphin-user/Dump/Frames"
)


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    number: int
    width: int
    height: int
    mode: str
    bytes_on_disk: int
    alpha_extrema: tuple[int, int] | None


@dataclass(frozen=True)
class Corpus:
    frames_dir: Path
    frames: list[FrameInfo]
    max_width: int
    max_height: int
    original_png_bytes: int
    source_rgb_hash: str

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def padded_rgb_frame_bytes(self) -> int:
        return self.max_width * self.max_height * 3

    @property
    def padded_rgb_bytes(self) -> int:
        return self.padded_rgb_frame_bytes * self.frame_count


@dataclass
class Result:
    name: str
    kind: str
    output_path: str
    output_bytes: int
    ratio_vs_png: float
    encode_seconds: float
    notes: str = ""
    verified: bool | None = None
    verify_seconds: float | None = None
    decoded_rgb_hash: str | None = None

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "outputPath": self.output_path,
            "outputBytes": self.output_bytes,
            "ratioVsPng": round(self.ratio_vs_png, 4),
            "encodeSeconds": round(self.encode_seconds, 3),
            "notes": self.notes,
            "verified": self.verified,
            "verifySeconds": None if self.verify_seconds is None else round(self.verify_seconds, 3),
            "decodedRgbHash": self.decoded_rgb_hash,
        }


def numeric_suffix(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def discover_frames(frames_dir: Path, limit: int | None) -> list[Path]:
    frames = sorted(frames_dir.glob("framedump_*.png"), key=numeric_suffix)
    if limit is not None:
        frames = frames[:limit]
    if not frames:
        raise SystemExit(f"No framedump_*.png files found in {frames_dir}")
    return frames


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_padded_rgb(path: Path, max_width: int, max_height: int) -> np.ndarray:
    source = read_rgb(path)
    frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    height, width = source.shape[:2]
    frame[:height, :width, :] = source
    return frame


def corpus_hash(frames: list[FrameInfo]) -> str:
    digest = hashlib.blake2b(digest_size=32)
    for frame in frames:
        source = read_rgb(frame.path)
        digest.update(frame.number.to_bytes(8, "little"))
        digest.update(frame.width.to_bytes(4, "little"))
        digest.update(frame.height.to_bytes(4, "little"))
        digest.update(source.tobytes(order="C"))
    return digest.hexdigest()


def inspect_corpus(frames_dir: Path, limit: int | None) -> Corpus:
    infos: list[FrameInfo] = []
    for path in discover_frames(frames_dir, limit):
        with Image.open(path) as image:
            alpha_extrema = None
            if "A" in image.getbands():
                alpha_extrema = image.getchannel("A").getextrema()
            infos.append(
                FrameInfo(
                    path=path,
                    number=numeric_suffix(path),
                    width=image.width,
                    height=image.height,
                    mode=image.mode,
                    bytes_on_disk=path.stat().st_size,
                    alpha_extrema=alpha_extrema,
                )
            )
    return Corpus(
        frames_dir=frames_dir,
        frames=infos,
        max_width=max(frame.width for frame in infos),
        max_height=max(frame.height for frame in infos),
        original_png_bytes=sum(frame.bytes_on_disk for frame in infos),
        source_rgb_hash=corpus_hash(infos),
    )


def write_manifest(path: Path, corpus: Corpus, strategy: str, extra: dict | None = None) -> None:
    payload = {
        "strategy": strategy,
        "framesDir": str(corpus.frames_dir),
        "frameCount": corpus.frame_count,
        "maxWidth": corpus.max_width,
        "maxHeight": corpus.max_height,
        "paddedRgbFrameBytes": corpus.padded_rgb_frame_bytes,
        "originalPngBytes": corpus.original_png_bytes,
        "sourceRgbHash": corpus.source_rgb_hash,
        "frames": [
            {
                "file": frame.path.name,
                "number": frame.number,
                "width": frame.width,
                "height": frame.height,
                "mode": frame.mode,
                "bytesOnDisk": frame.bytes_on_disk,
                "alphaExtrema": frame.alpha_extrema,
            }
            for frame in corpus.frames
        ],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_command(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    started = time.monotonic()
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    proc.elapsed_seconds = time.monotonic() - started  # type: ignore[attr-defined]
    if proc.returncode:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def stream_to_process(command: list[str], chunks: Iterable[bytes]) -> float:
    started = time.monotonic()
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdin is not None
        try:
            for chunk in chunks:
                proc.stdin.write(chunk)
            proc.stdin.close()
            stdout = proc.stdout.read() if proc.stdout is not None else b""
            stderr = proc.stderr.read() if proc.stderr is not None else b""
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
        returncode = proc.wait()
    if returncode:
        raise RuntimeError(
            f"Command failed ({returncode}): {' '.join(command)}\nSTDOUT:\n{stdout.decode(errors='replace')}\nSTDERR:\n{stderr.decode(errors='replace')}"
        )
    return time.monotonic() - started


def zstd_command(output: Path, level: int = 19, long_log: int = 27) -> list[str]:
    return ["zstd", f"-{level}", f"--long={long_log}", "-T0", "-q", "-f", "-o", str(output), "-"]


def xz_command(output: Path) -> list[str]:
    return ["xz", "-9e", "-T0", "-c"]


def compress_with_xz(output: Path, chunks: Iterable[bytes]) -> float:
    started = time.monotonic()
    with output.open("wb") as handle:
        with subprocess.Popen(["xz", "-9e", "-T0", "-c"], stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.PIPE) as proc:
            assert proc.stdin is not None
            try:
                for chunk in chunks:
                    proc.stdin.write(chunk)
                proc.stdin.close()
                stderr = proc.stderr.read() if proc.stderr is not None else b""
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            returncode = proc.wait()
    if returncode:
        raise RuntimeError(f"xz failed ({returncode}): {stderr.decode(errors='replace')}")
    return time.monotonic() - started


def padded_rgb_chunks(corpus: Corpus) -> Iterator[bytes]:
    for frame in corpus.frames:
        yield read_padded_rgb(frame.path, corpus.max_width, corpus.max_height).tobytes(order="C")


def interframe_chunks(corpus: Corpus, mode: str) -> Iterator[bytes]:
    previous = np.zeros((corpus.max_height, corpus.max_width, 3), dtype=np.uint8)
    for frame in corpus.frames:
        current = read_padded_rgb(frame.path, corpus.max_width, corpus.max_height)
        if mode == "xor":
            encoded = np.bitwise_xor(current, previous)
        elif mode == "sub8":
            encoded = np.subtract(current, previous, dtype=np.uint8)
        else:
            raise ValueError(mode)
        previous = current
        yield encoded.tobytes(order="C")


def zigzag_encode(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def zigzag_decode(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError(value)
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def encode_svarint(value: int) -> bytes:
    return encode_varint(zigzag_encode(value))


def shift_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = frame.shape[:2]
    shifted = np.zeros_like(frame)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(width, width - dx)
    src_y1 = min(height, height - dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1, :] = frame[src_y0:src_y1, src_x0:src_x1, :]
    return shifted


def shift_score(previous: np.ndarray, current: np.ndarray, dx: int, dy: int, stride: int) -> float:
    height, width = current.shape[:2]
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    src_x1 = min(width, width - dx)
    src_y1 = min(height, height - dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return float("inf")
    prev_sample = previous[src_y0:src_y1:stride, src_x0:src_x1:stride, :].astype(np.int16)
    curr_sample = current[dst_y0:dst_y1:stride, dst_x0:dst_x1:stride, :].astype(np.int16)
    return float(np.mean(np.abs(curr_sample - prev_sample)))


def best_global_shift(previous: np.ndarray, current: np.ndarray, radius: int, stride: int) -> tuple[int, int]:
    best = (0, 0)
    best_score = shift_score(previous, current, 0, 0, stride)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            score = shift_score(previous, current, dx, dy, stride)
            if score < best_score:
                best_score = score
                best = (dx, dy)
    return best


def motion_sub8_chunks(corpus: Corpus, radius: int, stride: int) -> Iterator[bytes]:
    previous = np.zeros((corpus.max_height, corpus.max_width, 3), dtype=np.uint8)
    for index, frame in enumerate(corpus.frames):
        current = read_padded_rgb(frame.path, corpus.max_width, corpus.max_height)
        dx, dy = (0, 0) if index == 0 else best_global_shift(previous, current, radius, stride)
        predicted = shift_frame(previous, dx, dy)
        residual = np.subtract(current, predicted, dtype=np.uint8)
        yield encode_svarint(dx) + encode_svarint(dy) + residual.tobytes(order="C")
        previous = current


def span_delta_chunks(corpus: Corpus) -> Iterator[bytes]:
    previous = np.zeros((corpus.max_height, corpus.max_width, 3), dtype=np.uint8)
    for frame in corpus.frames:
        current = read_padded_rgb(frame.path, corpus.max_width, corpus.max_height)
        mask = np.any(current != previous, axis=2)
        runs: list[tuple[int, int, int, bytes]] = []
        for y in range(corpus.max_height):
            row = mask[y]
            if not row.any():
                continue
            padded = np.concatenate(([False], row, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            for start, end in zip(edges[0::2], edges[1::2]):
                run = current[y : y + 1, start:end, :].tobytes(order="C")
                runs.append((y, int(start), int(end - start), run))
        block = bytearray()
        block.extend(encode_varint(len(runs)))
        for y, x, length, run in runs:
            block.extend(encode_varint(y))
            block.extend(encode_varint(x))
            block.extend(encode_varint(length))
            block.extend(run)
        previous = current
        yield bytes(block)


def iter_tiles(width: int, height: int, tile_size: int) -> Iterator[tuple[int, int, int, int, int]]:
    tile_index = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            yield tile_index, x, y, min(tile_size, width - x), min(tile_size, height - y)
            tile_index += 1


def tile_delta_chunks(corpus: Corpus, tile_size: int) -> Iterator[bytes]:
    previous = np.zeros((corpus.max_height, corpus.max_width, 3), dtype=np.uint8)
    tiles = list(iter_tiles(corpus.max_width, corpus.max_height, tile_size))
    for frame in corpus.frames:
        current = read_padded_rgb(frame.path, corpus.max_width, corpus.max_height)
        changed: list[tuple[int, bytes]] = []
        for tile_index, x, y, tile_width, tile_height in tiles:
            current_tile = current[y : y + tile_height, x : x + tile_width, :]
            previous_tile = previous[y : y + tile_height, x : x + tile_width, :]
            if not np.array_equal(current_tile, previous_tile):
                changed.append((tile_index, current_tile.tobytes(order="C")))
        block = bytearray()
        block.extend(encode_varint(len(changed)))
        for tile_index, tile_bytes in changed:
            block.extend(encode_varint(tile_index))
            block.extend(tile_bytes)
        previous = current
        yield bytes(block)


class ByteReader:
    def __init__(self, handle):
        self.handle = handle
        self.buffer = bytearray()

    def read_exact(self, size: int) -> bytes:
        while len(self.buffer) < size:
            chunk = self.handle.read(max(65536, size - len(self.buffer)))
            if not chunk:
                raise EOFError(f"needed {size} bytes, had {len(self.buffer)}")
            self.buffer.extend(chunk)
        out = bytes(self.buffer[:size])
        del self.buffer[:size]
        return out

    def read_varint(self) -> int:
        shift = 0
        value = 0
        while True:
            byte = self.read_exact(1)[0]
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7

    def read_svarint(self) -> int:
        return zigzag_decode(self.read_varint())


def zstd_decompress_reader(path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["zstd", "-d", "-q", "--long=27", "-c", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def update_hash_from_cropped(digest, corpus: Corpus, frame: FrameInfo, padded: np.ndarray) -> None:
    cropped = padded[: frame.height, : frame.width, :]
    digest.update(frame.number.to_bytes(8, "little"))
    digest.update(frame.width.to_bytes(4, "little"))
    digest.update(frame.height.to_bytes(4, "little"))
    digest.update(cropped.tobytes(order="C"))


def verify_stream(
    corpus: Corpus,
    compressed_path: Path,
    mode: str,
    tile_size: int | None = None,
) -> tuple[bool, float, str]:
    started = time.monotonic()
    proc = zstd_decompress_reader(compressed_path)
    assert proc.stdout is not None
    reader = ByteReader(proc.stdout)
    digest = hashlib.blake2b(digest_size=32)
    previous = np.zeros((corpus.max_height, corpus.max_width, 3), dtype=np.uint8)
    tiles = list(iter_tiles(corpus.max_width, corpus.max_height, tile_size or 16))
    try:
        for frame in corpus.frames:
            if mode == "raw":
                chunk = reader.read_exact(corpus.padded_rgb_frame_bytes)
                current = np.frombuffer(chunk, dtype=np.uint8).reshape((corpus.max_height, corpus.max_width, 3))
            elif mode in {"xor", "sub8"}:
                chunk = reader.read_exact(corpus.padded_rgb_frame_bytes)
                encoded = np.frombuffer(chunk, dtype=np.uint8).reshape((corpus.max_height, corpus.max_width, 3))
                if mode == "xor":
                    current = np.bitwise_xor(encoded, previous)
                else:
                    current = np.add(encoded, previous, dtype=np.uint8)
            elif mode == "motion_sub8":
                dx = reader.read_svarint()
                dy = reader.read_svarint()
                predicted = shift_frame(previous, dx, dy)
                chunk = reader.read_exact(corpus.padded_rgb_frame_bytes)
                encoded = np.frombuffer(chunk, dtype=np.uint8).reshape((corpus.max_height, corpus.max_width, 3))
                current = np.add(encoded, predicted, dtype=np.uint8)
            elif mode == "spans":
                current = previous.copy()
                run_count = reader.read_varint()
                for _ in range(run_count):
                    y = reader.read_varint()
                    x = reader.read_varint()
                    length = reader.read_varint()
                    run = reader.read_exact(length * 3)
                    current[y : y + 1, x : x + length, :] = np.frombuffer(
                        run,
                        dtype=np.uint8,
                    ).reshape((1, length, 3))
            elif mode == "tiles":
                current = previous.copy()
                changed_count = reader.read_varint()
                for _ in range(changed_count):
                    tile_index = reader.read_varint()
                    _, x, y, tile_width, tile_height = tiles[tile_index]
                    tile_bytes = reader.read_exact(tile_width * tile_height * 3)
                    tile = np.frombuffer(tile_bytes, dtype=np.uint8).reshape((tile_height, tile_width, 3))
                    current[y : y + tile_height, x : x + tile_width, :] = tile
            else:
                raise ValueError(mode)
            update_hash_from_cropped(digest, corpus, frame, current)
            previous = current.copy()
        trailing = proc.stdout.read(1)
        if trailing:
            raise RuntimeError("decoded stream has trailing bytes")
    finally:
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
    if returncode:
        raise RuntimeError(f"zstd decode failed ({returncode}): {stderr.decode(errors='replace')}")
    decoded_hash = digest.hexdigest()
    return decoded_hash == corpus.source_rgb_hash, time.monotonic() - started, decoded_hash


def verify_video(corpus: Corpus, video_path: Path) -> tuple[bool, float, str]:
    started = time.monotonic()
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    reader = ByteReader(proc.stdout)
    digest = hashlib.blake2b(digest_size=32)
    try:
        for frame in corpus.frames:
            chunk = reader.read_exact(corpus.padded_rgb_frame_bytes)
            current = np.frombuffer(chunk, dtype=np.uint8).reshape((corpus.max_height, corpus.max_width, 3))
            update_hash_from_cropped(digest, corpus, frame, current)
        trailing = proc.stdout.read(1)
        if trailing:
            raise RuntimeError("decoded video has trailing bytes")
    finally:
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg decode failed ({returncode}): {stderr.decode(errors='replace')}")
    decoded_hash = digest.hexdigest()
    return decoded_hash == corpus.source_rgb_hash, time.monotonic() - started, decoded_hash


def make_result(
    *,
    name: str,
    kind: str,
    output_path: Path,
    corpus: Corpus,
    encode_seconds: float,
    notes: str = "",
) -> Result:
    output_bytes = output_path.stat().st_size
    return Result(
        name=name,
        kind=kind,
        output_path=str(output_path),
        output_bytes=output_bytes,
        ratio_vs_png=corpus.original_png_bytes / output_bytes,
        encode_seconds=encode_seconds,
        notes=notes,
    )


def bench_png_tar_zstd(corpus: Corpus, run_dir: Path) -> Result:
    output = run_dir / "png-files.tar.zst"
    tar_path = run_dir / "png-files.tar"
    manifest = run_dir / "png-files.manifest.json"
    write_manifest(manifest, corpus, "png_tar_zstd")
    started = time.monotonic()
    with tarfile.open(tar_path, "w") as tar:
        tar.add(manifest, arcname=manifest.name)
        for frame in corpus.frames:
            tar.add(frame.path, arcname=frame.path.name)
    run_command(["zstd", "-19", "--long=27", "-T0", "-q", "-f", str(tar_path), "-o", str(output)])
    tar_path.unlink()
    return make_result(
        name="png_tar_zstd19_long",
        kind="png-bytes",
        output_path=output,
        corpus=corpus,
        encode_seconds=time.monotonic() - started,
        notes="Original PNG files tarred with manifest, then zstd --long=27.",
    )


def bench_stream_zstd(corpus: Corpus, run_dir: Path, name: str, mode: str) -> Result:
    output = run_dir / f"{name}.rgb.zst"
    manifest = run_dir / f"{name}.manifest.json"
    write_manifest(manifest, corpus, name, {"encoding": mode})
    chunk_iter = padded_rgb_chunks(corpus) if mode == "raw" else interframe_chunks(corpus, mode)
    seconds = stream_to_process(zstd_command(output), chunk_iter)
    return make_result(
        name=f"{name}_zstd19_long",
        kind="rgb-stream",
        output_path=output,
        corpus=corpus,
        encode_seconds=seconds,
        notes=f"Padded RGB stream encoded as {mode}, compressed with zstd --long=27.",
    )


def bench_stream_xz(corpus: Corpus, run_dir: Path, name: str, mode: str) -> Result:
    output = run_dir / f"{name}.rgb.xz"
    manifest = run_dir / f"{name}.manifest.json"
    write_manifest(manifest, corpus, name, {"encoding": mode})
    chunk_iter = padded_rgb_chunks(corpus) if mode == "raw" else interframe_chunks(corpus, mode)
    seconds = compress_with_xz(output, chunk_iter)
    return make_result(
        name=f"{name}_xz9e",
        kind="rgb-stream",
        output_path=output,
        corpus=corpus,
        encode_seconds=seconds,
        notes=f"Padded RGB stream encoded as {mode}, compressed with xz -9e.",
    )


def bench_tiles_zstd(corpus: Corpus, run_dir: Path, tile_size: int) -> Result:
    output = run_dir / f"tiles_{tile_size}.rgb.zst"
    manifest = run_dir / f"tiles_{tile_size}.manifest.json"
    tile_count = math.ceil(corpus.max_width / tile_size) * math.ceil(corpus.max_height / tile_size)
    write_manifest(
        manifest,
        corpus,
        f"tiles_{tile_size}_zstd",
        {"encoding": "tiles", "tileSize": tile_size, "tileCount": tile_count},
    )
    seconds = stream_to_process(zstd_command(output), tile_delta_chunks(corpus, tile_size))
    return make_result(
        name=f"tiles_{tile_size}_zstd19_long",
        kind="tile-delta-stream",
        output_path=output,
        corpus=corpus,
        encode_seconds=seconds,
        notes=f"Only changed {tile_size}x{tile_size} RGB tiles are emitted, then zstd --long=27.",
    )


def bench_motion_sub8_zstd(corpus: Corpus, run_dir: Path, radius: int = 8, stride: int = 4) -> Result:
    output = run_dir / f"motion_sub8_r{radius}_s{stride}.rgb.zst"
    manifest = run_dir / f"motion_sub8_r{radius}_s{stride}.manifest.json"
    write_manifest(
        manifest,
        corpus,
        "motion_sub8_zstd",
        {"encoding": "motion_sub8", "searchRadius": radius, "searchStride": stride},
    )
    seconds = stream_to_process(zstd_command(output), motion_sub8_chunks(corpus, radius, stride))
    return make_result(
        name=f"motion_sub8_r{radius}_s{stride}_zstd19_long",
        kind="motion-residual-stream",
        output_path=output,
        corpus=corpus,
        encode_seconds=seconds,
        notes=f"Global +/-{radius}px motion search, modulo residual, then zstd --long=27.",
    )


def bench_spans_zstd(corpus: Corpus, run_dir: Path) -> Result:
    output = run_dir / "spans.rgb.zst"
    manifest = run_dir / "spans.manifest.json"
    write_manifest(manifest, corpus, "spans_zstd", {"encoding": "spans"})
    seconds = stream_to_process(zstd_command(output), span_delta_chunks(corpus))
    return make_result(
        name="spans_zstd19_long",
        kind="pixel-span-delta-stream",
        output_path=output,
        corpus=corpus,
        encode_seconds=seconds,
        notes="Only changed pixel runs are emitted row-by-row, then zstd --long=27.",
    )


def bench_video(corpus: Corpus, run_dir: Path, name: str, ffmpeg_args: list[str], notes: str) -> Result:
    output = run_dir / name
    manifest = run_dir / f"{Path(name).stem}.manifest.json"
    write_manifest(manifest, corpus, f"video_{Path(name).stem}", {"encoding": ffmpeg_args})
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{corpus.max_width}x{corpus.max_height}",
        "-r",
        "60",
        "-i",
        "-",
        *ffmpeg_args,
        str(output),
    ]
    seconds = stream_to_process(command, padded_rgb_chunks(corpus))
    return make_result(
        name=Path(name).stem,
        kind="lossless-video",
        output_path=output,
        corpus=corpus,
        encode_seconds=seconds,
        notes=notes,
    )


def run_strategy(name: str, corpus: Corpus, run_dir: Path, tile_size: int) -> Result:
    if name == "png_tar_zstd":
        return bench_png_tar_zstd(corpus, run_dir)
    if name == "raw_zstd":
        return bench_stream_zstd(corpus, run_dir, "raw", "raw")
    if name == "raw_xz":
        return bench_stream_xz(corpus, run_dir, "raw", "raw")
    if name == "xor_zstd":
        return bench_stream_zstd(corpus, run_dir, "xor", "xor")
    if name == "xor_xz":
        return bench_stream_xz(corpus, run_dir, "xor", "xor")
    if name == "sub8_zstd":
        return bench_stream_zstd(corpus, run_dir, "sub8", "sub8")
    if name == "motion_sub8_zstd":
        return bench_motion_sub8_zstd(corpus, run_dir)
    if name == "spans_zstd":
        return bench_spans_zstd(corpus, run_dir)
    if name == "tiles_zstd":
        return bench_tiles_zstd(corpus, run_dir, tile_size)
    if name == "ffv1":
        return bench_video(
            corpus,
            run_dir,
            "ffv1_level3.mkv",
            ["-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "rgb24"],
            "FFV1 level 3; lossless but intra-frame, so it does not exploit temporal motion much.",
        )
    if name == "x264rgb":
        return bench_video(
            corpus,
            run_dir,
            "x264rgb_qp0_veryslow.mkv",
            [
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                "-preset",
                "veryslow",
                "-pix_fmt",
                "rgb24",
                "-x264-params",
                "keyint=9999:min-keyint=9999:scenecut=0",
            ],
            "libx264rgb lossless QP 0 with a long GOP to exploit inter-frame repetition.",
        )
    if name == "x264rgb_animation":
        return bench_video(
            corpus,
            run_dir,
            "x264rgb_qp0_animation.mkv",
            [
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                "-preset",
                "veryslow",
                "-tune",
                "animation",
                "-pix_fmt",
                "rgb24",
                "-x264-params",
                "keyint=9999:min-keyint=1:scenecut=40",
            ],
            "libx264rgb lossless QP 0 with animation tune and normal scenecut.",
        )
    if name == "x264rgb_placebo":
        return bench_video(
            corpus,
            run_dir,
            "x264rgb_qp0_placebo.mkv",
            [
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                "-preset",
                "placebo",
                "-pix_fmt",
                "rgb24",
                "-x264-params",
                "keyint=9999:min-keyint=1:scenecut=40",
            ],
            "libx264rgb lossless QP 0 with placebo preset.",
        )
    if name == "x265_lossless":
        return bench_video(
            corpus,
            run_dir,
            "x265_gbrp_lossless.mkv",
            [
                "-c:v",
                "libx265",
                "-preset",
                "veryslow",
                "-pix_fmt",
                "gbrp",
                "-x265-params",
                "lossless=1:keyint=9999:min-keyint=1:scenecut=40",
            ],
            "libx265 lossless with GBR 4:4:4 input.",
        )
    if name == "vp9":
        return bench_video(
            corpus,
            run_dir,
            "vp9_lossless.webm",
            [
                "-c:v",
                "libvpx-vp9",
                "-lossless",
                "1",
                "-deadline",
                "best",
                "-cpu-used",
                "0",
                "-row-mt",
                "1",
                "-pix_fmt",
                "gbrp",
                "-tune-content",
                "screen",
            ],
            "libvpx-vp9 lossless, tuned for screen content.",
        )
    raise ValueError(name)


def verify_result(corpus: Corpus, result: Result, tile_size: int) -> Result:
    path = Path(result.output_path)
    if result.name.startswith("raw_") and path.suffix == ".zst":
        verified, seconds, decoded_hash = verify_stream(corpus, path, "raw")
    elif result.name.startswith("xor_") and path.suffix == ".zst":
        verified, seconds, decoded_hash = verify_stream(corpus, path, "xor")
    elif result.name.startswith("sub8_") and path.suffix == ".zst":
        verified, seconds, decoded_hash = verify_stream(corpus, path, "sub8")
    elif result.name.startswith("motion_sub8_") and path.suffix == ".zst":
        verified, seconds, decoded_hash = verify_stream(corpus, path, "motion_sub8")
    elif result.name.startswith("spans_") and path.suffix == ".zst":
        verified, seconds, decoded_hash = verify_stream(corpus, path, "spans")
    elif result.name.startswith("tiles_") and path.suffix == ".zst":
        verified, seconds, decoded_hash = verify_stream(corpus, path, "tiles", tile_size=tile_size)
    elif result.kind == "lossless-video":
        verified, seconds, decoded_hash = verify_video(corpus, path)
    else:
        result.verified = None
        return result
    result.verified = verified
    result.verify_seconds = seconds
    result.decoded_rgb_hash = decoded_hash
    if not verified:
        raise RuntimeError(f"Verification failed for {result.name}: {decoded_hash} != {corpus.source_rgb_hash}")
    return result


def write_summary(run_dir: Path, corpus: Corpus, results: list[Result]) -> None:
    payload = {
        "framesDir": str(corpus.frames_dir),
        "frameCount": corpus.frame_count,
        "maxWidth": corpus.max_width,
        "maxHeight": corpus.max_height,
        "originalPngBytes": corpus.original_png_bytes,
        "originalPngMiB": round(corpus.original_png_bytes / 1024 / 1024, 3),
        "paddedRgbBytes": corpus.padded_rgb_bytes,
        "paddedRgbMiB": round(corpus.padded_rgb_bytes / 1024 / 1024, 3),
        "sourceRgbHash": corpus.source_rgb_hash,
        "results": [result.to_json() for result in sorted(results, key=lambda item: item.output_bytes)],
    }
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PNG Compression Probe Results",
        "",
        f"- Frames: {corpus.frame_count}",
        f"- Canvas: {corpus.max_width}x{corpus.max_height}",
        f"- Original PNG bytes: {corpus.original_png_bytes:,} ({corpus.original_png_bytes / 1024 / 1024:.2f} MiB)",
        f"- Padded RGB bytes: {corpus.padded_rgb_bytes:,} ({corpus.padded_rgb_bytes / 1024 / 1024:.2f} MiB)",
        "",
        "| Rank | Strategy | Size MiB | Ratio vs PNG | Encode s | Verified | Notes |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for index, result in enumerate(sorted(results, key=lambda item: item.output_bytes), start=1):
        verified = "" if result.verified is None else str(result.verified)
        lines.append(
            f"| {index} | `{result.name}` | {result.output_bytes / 1024 / 1024:.2f} | "
            f"{result.ratio_vs_png:.2f}x | {result.encode_seconds:.1f} | {verified} | {result.notes} |"
        )
    lines.append("")
    (run_dir / "REPORT.md").write_text("\n".join(lines))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CPU-decodable compression for PNG frame dumps.")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "runs/latest")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N numeric framedump PNGs.")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "png_tar_zstd",
            "raw_zstd",
            "xor_zstd",
            "sub8_zstd",
            "motion_sub8_zstd",
            "spans_zstd",
            "tiles_zstd",
            "ffv1",
            "x264rgb",
            "vp9",
        ],
        choices=[
            "png_tar_zstd",
            "raw_zstd",
            "raw_xz",
            "xor_zstd",
            "xor_xz",
            "sub8_zstd",
            "motion_sub8_zstd",
            "spans_zstd",
            "tiles_zstd",
            "ffv1",
            "x264rgb",
            "x264rgb_animation",
            "x264rgb_placebo",
            "x265_lossless",
            "vp9",
        ],
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    corpus = inspect_corpus(args.frames_dir.resolve(), args.limit)
    write_manifest(run_dir / "corpus.manifest.json", corpus, "corpus")
    print(
        json.dumps(
            {
                "event": "corpus",
                "frames": corpus.frame_count,
                "maxWidth": corpus.max_width,
                "maxHeight": corpus.max_height,
                "pngMiB": round(corpus.original_png_bytes / 1024 / 1024, 3),
                "paddedRgbMiB": round(corpus.padded_rgb_bytes / 1024 / 1024, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results: list[Result] = []
    for strategy in args.strategies:
        print(json.dumps({"event": "strategy_started", "strategy": strategy}, sort_keys=True), flush=True)
        result = run_strategy(strategy, corpus, run_dir, args.tile_size)
        if args.verify:
            result = verify_result(corpus, result, args.tile_size)
        results.append(result)
        write_summary(run_dir, corpus, results)
        print(json.dumps({"event": "strategy_done", **result.to_json()}, sort_keys=True), flush=True)

    write_summary(run_dir, corpus, results)
    print(str(run_dir / "REPORT.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
