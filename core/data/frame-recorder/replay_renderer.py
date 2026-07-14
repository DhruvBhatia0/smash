"""Strict CPU-only Slippi replay to 20 FPS MP4 conversion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path


WIDTH, HEIGHT, SOURCE_FPS, TARGET_FPS = 642, 528, 60, 20


class NoPlayableFrames(ValueError):
    pass


def _run(command: list[str], *, timeout: int = 600, env: dict | None = None) -> str:
    process = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, env=env,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    return process.stdout


@dataclass(frozen=True)
class _Plan:
    raw_position: int
    first: int
    last: int
    segment: int = 1
    chunks: tuple[tuple[int, int], ...] = ()


def _plan(data: bytes, path: Path) -> _Plan:
    """Select the first complete game from occasionally concatenated SLPs."""
    if not data:
        raise ValueError(f"empty SLP: {path}")
    raw = 0 if data[0] != ord("{") else 15
    length = len(data) if not raw else int.from_bytes(data[raw - 4:raw], "big")
    end = min(len(data), raw + (length if length > 0 else len(data) - raw))
    sizes = {0x36: 0x140, 0x37: 0x6, 0x38: 0x46, 0x39: 0x1} if not raw else {}
    overall: list[int] = []
    started = False
    starts = 0
    table = segment_table = None
    segment_start = None
    segment_frames: list[int] = []
    complete = None
    later = False
    position = raw
    while position < end:
        command = data[position]
        if command == 0x35:
            if position + 2 > end:
                break
            payload = data[position + 1]
            stop = position + payload + 1
            if payload < 1 or stop > end:
                break
            size_data = data[position + 2:position + 1 + payload]
            sizes = {0x35: payload}
            for offset in range(0, len(size_data) - 2, 3):
                sizes[size_data[offset]] = int.from_bytes(size_data[offset + 1:offset + 3], "big")
            table = (position, stop)
            position = stop
            continue
        size = sizes.get(command)
        stop = position + (size or 0) + 1
        if size is None or stop > end:
            break
        if command == 0x36:
            starts += 1
            later |= complete is not None
            started, segment_start, segment_table, segment_frames = True, position, table, []
        elif command == 0x38 and stop - position >= 5:
            frame = struct.unpack(">i", data[position + 1:position + 5])[0]
            overall.append(frame)
            if started:
                segment_frames.append(frame)
        elif command == 0x39:
            if started and complete is None and segment_start is not None and segment_frames:
                complete = (starts, segment_table, segment_start, stop, min(segment_frames), max(segment_frames))
            started = False
        position = stop
    if not overall:
        raise ValueError(f"SLP has no post-frame updates: {path}")
    if complete is None:
        if starts > 1:
            raise ValueError(f"SLP has multiple games but no complete game: {path}")
        return _Plan(raw, min(overall), max(overall))
    segment, table, start, stop, first, last = complete
    if segment == 1 and not later and first == min(overall) and last == max(overall):
        return _Plan(raw, first, last)
    return _Plan(raw, first, last, segment, tuple(([table] if table else []) + [(start, stop)]))


def _normalize(source: Path, target: Path) -> tuple[Path, _Plan, dict | None]:
    data = source.read_bytes()
    plan = _plan(data, source)
    if not plan.chunks:
        return source, plan, None
    selected = b"".join(data[start:stop] for start, stop in plan.chunks)
    if plan.raw_position:
        if not selected or selected[0] != 0x35:
            raise ValueError("normalized SLP is missing its message-size table")
        prefix = bytearray(data[:plan.raw_position])
        prefix[plan.raw_position - 4:plan.raw_position] = len(selected).to_bytes(4, "big")
        selected = bytes(prefix) + selected + b"U\x08metadata{}}"
    target.write_bytes(selected)
    return target, plan, {
        "policy": "first-complete-game-v1", "selectedSegment": plan.segment,
        "firstFrame": plan.first, "lastFrame": plan.last,
        "renderInputBytes": len(selected), "renderInputSha256": hashlib.sha256(selected).hexdigest(),
    }


def _probe(path: Path) -> dict:
    fields = "codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames"
    streams = json.loads(_run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", f"stream={fields}", "-of", "json", str(path),
    ]))["streams"]
    if len(streams) != 1 or streams[0].get("nb_read_frames") in {None, "N/A"}:
        raise RuntimeError(f"could not probe one video stream in {path}")
    streams[0]["frames"] = int(streams[0]["nb_read_frames"])
    return streams[0]


def _pts(path: Path) -> tuple[float, float]:
    values = [float(line.rstrip(",")) for line in _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "packet=pts_time", "-of", "csv=p=0", str(path),
    ]).splitlines() if line.rstrip(",") not in {"", "N/A"}]
    if not values:
        raise RuntimeError(f"video has no packet timestamps: {path}")
    return values[0], max(values)


def _validate_output(path: Path, expected_frames: int) -> dict:
    video = _probe(path)
    expected = {
        "codec_name": "h264", "width": WIDTH, "height": HEIGHT,
        "pix_fmt": "yuv420p", "r_frame_rate": "20/1",
        "avg_frame_rate": "20/1", "frames": expected_frames,
    }
    for key, value in expected.items():
        if video.get(key) != value:
            raise RuntimeError(f"invalid output {key}: {video.get(key)!r} != {value!r}")
    stream_types = json.loads(_run([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(path),
    ]))["streams"]
    if [stream["codec_type"] for stream in stream_types] != ["video"] or abs(_pts(path)[0]) > 0.0001:
        raise RuntimeError("output must contain one zero-based video stream and no audio")
    return video


def _recover_tail(manifest: dict, first: int, requested_last: int) -> tuple[int, dict | None]:
    rendered = (manifest.get("currentFrameRange") or {}).get("last")
    if rendered is not None and int(rendered) >= requested_last:
        return requested_last, None
    minimum = max(first + 600, first + math.ceil((requested_last - first) * 0.75))
    if manifest.get("stalledFrame") is None or int(manifest["stalledFrame"]) != int(rendered or -1) or int(rendered or -1) < minimum:
        raise RuntimeError(f"render stopped before replay end: {rendered} < {requested_last}")
    rendered = int(rendered)
    return rendered, {
        "policy": "stable-render-prefix-v1", "reason": "dolphin_frame_progress_stalled",
        "lastRenderedFrame": rendered, "originalLastFrame": requested_last,
        "discardedTailFrames": requested_last - rendered,
        "stallSeconds": int(os.environ.get("SMASH_RENDER_STALL_SECONDS", "60")),
    }


def render(job: dict, source: Path, work_root: Path, threads: int) -> Path:
    """Render one replay and return a tar containing video.mp4 and metadata.json."""
    work = work_root / f"{int(job['id']):06d}"
    shutil.rmtree(work, ignore_errors=True)
    render_root, output_dir = work / "render", work / "recording"
    render_root.mkdir(parents=True)
    output_dir.mkdir()
    render_input, plan, normalization = _normalize(source, work / "normalized.slp")
    metadata_path = output_dir / "metadata.json"
    _run([
        "/opt/slippi-renderer/extract-slp-metadata.mjs", str(render_input),
        str(metadata_path), job["reference"], "gdrive",
    ], timeout=120)
    metadata = json.loads(metadata_path.read_text())
    if normalization:
        metadata["file"] = {"name": "input.slp", "bytes": job["sourceBytes"], "sha256": job["sourceSha256"]}
        metadata["replayNormalization"] = normalization
    match = metadata.get("match") or {}
    frames = match.get("frames") or {}
    first, last = int(frames.get("firstPlayable", -39)), int(frames.get("last", plan.last))
    raw_first, raw_last = plan.first + 1, plan.last
    if last < first:
        raise NoPlayableFrames(f"no playable frames: {first}..{last}")
    if not raw_first <= first <= last <= raw_last:
        raise RuntimeError(f"invalid playable/render frame mapping: raw={raw_first}..{raw_last}, playable={first}..{last}")
    requested_last = last
    chunk_frames = int(os.environ.get("SMASH_RAW_CHUNK_SOURCE_FRAMES", "3600"))
    if chunk_frames < 3:
        raise ValueError("raw chunk must contain at least three 60 Hz source frames")
    chunk_frames -= chunk_frames % 3
    minimum_stall = max(first + 600, first + math.ceil((last - first) * 0.75))
    timeout = int(os.environ.get("SMASH_RENDER_TIMEOUT_SECONDS", "1200"))
    started = time.monotonic()
    parts: list[Path] = []
    part_frames: list[int] = []
    expected_frames = raw_frames = raw_bytes = 0
    capture_first = first
    segment = 0
    while capture_first <= last:
        capture_stop = min(requested_last, capture_first + chunk_frames - 1)
        render_dir = render_root / f"{segment:04d}"
        playback = work / f"playback-{segment:04d}.json"
        playback.write_text(json.dumps({
            "replay": str(render_input), "commandId": f"daytona-{job['id']}-{segment}",
            "startFrame": capture_first - 1, "endFrame": plan.last + 1, "stopFrame": capture_stop,
        }))
        raw_timeline = capture_stop - capture_first + 1
        environment = os.environ.copy()
        environment.update({
            "LP_NUM_THREADS": str(threads), "LIBGL_ALWAYS_SOFTWARE": "1",
            "EGL_PLATFORM": "surfaceless", "MESA_GLTHREAD": "false",
            "SLIPPI_DOLPHIN_BIN": "/opt/slippi/dolphin-emu-nogui", "SLIPPI_DUMP_ONLY": "1",
            "SLIPPI_DUMP_FRAMES": "True", "SLIPPI_USE_FFV1": "False",
            "SLIPPI_DUMP_CODEC": "rawvideo", "SLIPPI_DUMP_FORMAT": "avi",
            "SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS": "True", "SLIPPI_EFB_SCALE": "2",
            "SLIPPI_CPU_THREAD": "False", "SLIPPI_RENDER_TO_MAIN": "False",
            "SLIPPI_RENDER_WIDTH": "204", "SLIPPI_RENDER_HEIGHT": "168",
            "SLIPPI_END_FRAME_POLL_SECONDS": "0.05",
            "SLIPPI_STALL_SECONDS": os.environ.get("SMASH_RENDER_STALL_SECONDS", "60"),
            "SLIPPI_STALL_MIN_FRAME": str(minimum_stall),
            "SLIPPI_MAX_RAW_BYTES": str(int(raw_timeline * WIDTH * HEIGHT * 1.65) + 64 * 1024**2),
        })
        _run([
            "/tmp/render-ffv1-replay.sh", "--replay-json", str(playback), "--iso",
            os.environ.get("SMASH_REMOTE_ISO", "/mnt/smash-assets/melee.iso"),
            "--output-dir", str(render_dir), "--user-dir", str(work / f"dolphin-user-{segment:04d}"),
            "--timeout-seconds", str(timeout), "--video-backend", "OGL", "--cpu-core", "1",
            "--audio-backend", "Null", "--no-xvfb",
        ], timeout=timeout + 30, env=environment)
        manifest = json.loads((render_dir / "manifest.json").read_text())
        if manifest.get("targetEndFrame") != plan.last + 1 or manifest.get("targetStopFrame") != capture_stop:
            raise RuntimeError("renderer changed an authoritative replay segment endpoint")
        observed_value = (manifest.get("currentFrameRange") or {}).get("last")
        observed_last = int(observed_value) if observed_value is not None else -10**9
        if capture_stop == requested_last:
            last, recovery = _recover_tail(manifest, first, requested_last)
            if recovery:
                metadata["renderTailRecovery"] = recovery
        elif observed_last < capture_stop:
            raise RuntimeError(f"render stopped before segment end: {observed_last} < {capture_stop}")
        segment_last = last if capture_stop == requested_last else capture_stop
        raw_files = [path for path in render_dir.iterdir()
                     if path.suffix.lower() in {".avi", ".mkv", ".mp4", ".mov", ".nut"}]
        if len(raw_files) != 1:
            raise RuntimeError(f"expected one raw capture, found {len(raw_files)}")
        raw = raw_files[0]
        raw_video = _probe(raw)
        if (int(raw_video["width"]), int(raw_video["height"])) != (WIDTH, HEIGHT):
            raise RuntimeError(f"raw capture is not native {WIDTH}x{HEIGHT}")
        first_pts, last_pts = _pts(raw)
        last_offset = round(last_pts * SOURCE_FPS)
        raw_last_frame = capture_first + last_offset
        if (first_pts < -0.0001 or first_pts > 1 / SOURCE_FPS + 0.001
                or abs(last_pts - last_offset / SOURCE_FPS) > 0.001
                or raw_video["frames"] > last_offset + 1 or raw_last_frame < segment_last):
            raise RuntimeError(
                "raw capture does not cover the authoritative Slippi segment timeline: "
                f"capture={capture_first}..{capture_stop}, observedLast={observed_last}, "
                f"rawLast={raw_last_frame}, rawFrames={raw_video['frames']}, pts={first_pts}..{last_pts}"
            )
        chunk_selected_last = capture_first + ((segment_last - capture_first) // 3) * 3
        chunk_expected = (chunk_selected_last - capture_first) // 3 + 1
        part = output_dir / f"part-{segment:04d}.mp4"
        filter_graph = (
            f"fps=60:start_time=0,select='between(n\\,0\\,{chunk_selected_last - capture_first})*"
            "not(mod(n\\,3))',setpts=N/(20*TB),format=yuv420p"
        )
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
            "-vf", filter_graph, "-vsync", "cfr", "-r", str(TARGET_FPS), "-an",
            "-c:v", "libx264", "-preset", os.environ.get("SMASH_H264_PRESET", "veryfast"),
            "-crf", os.environ.get("SMASH_H264_CRF", "18"), "-g", "40", "-bf", "2",
            "-threads", str(threads), "-pix_fmt", "yuv420p", "-video_track_timescale", "20",
            "-movflags", "+faststart", str(part),
        ])
        raw_frames += int(raw_video["frames"])
        raw_bytes += raw.stat().st_size
        expected_frames += chunk_expected
        parts.append(part)
        part_frames.append(chunk_expected)
        shutil.rmtree(render_dir)
        shutil.rmtree(work / f"dolphin-user-{segment:04d}")
        if segment_last < capture_stop or capture_stop == requested_last:
            break
        capture_first += chunk_frames
        segment += 1
    selected_last = first + (expected_frames - 1) * 3
    target, temporary = output_dir / "video.mp4", output_dir / "video.partial.mp4"
    concat_reencoded = False
    if len(parts) == 1:
        parts[0].replace(temporary)
    else:
        concat = work / "parts.txt"
        concat.write_text("".join(
            f"file '{part}'\nduration {frames / TARGET_FPS:.9f}\n"
            for part, frames in zip(parts, part_frames)
        ))
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat), "-map", "0:v:0", "-c", "copy", "-an", "-movflags", "+faststart",
            "-video_track_timescale", "20", str(temporary),
        ])
        try:
            video = _validate_output(temporary, expected_frames)
        except RuntimeError:
            # A minority of MP4 chunk combinations retain one timestamp tick at
            # a boundary when stream-copied.  Preserve the fast remux path for
            # normal jobs, but normalize packet timestamps when that happens.
            temporary.unlink(missing_ok=True)
            _run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat), "-map", "0:v:0", "-vf", "setpts=N/(20*TB),format=yuv420p",
                "-vsync", "cfr", "-r", str(TARGET_FPS), "-an", "-c:v", "libx264",
                "-preset", os.environ.get("SMASH_H264_PRESET", "veryfast"),
                "-crf", os.environ.get("SMASH_H264_CRF", "18"), "-g", "40", "-bf", "2",
                "-threads", str(threads), "-pix_fmt", "yuv420p", "-video_track_timescale", "20",
                "-movflags", "+faststart", str(temporary),
            ])
            concat_reencoded = True
            video = _validate_output(temporary, expected_frames)
        for part in parts:
            part.unlink()
    if len(parts) == 1:
        video = _validate_output(temporary, expected_frames)
    temporary.replace(target)
    elapsed = time.monotonic() - started
    metadata["video"] = {
        "file": target.name, "container": "mp4", "codec": "h264", "pixelFormat": "yuv420p",
        "frames": expected_frames, "frameRate": "20/1", "width": WIDTH, "height": HEIGHT,
        "inputBytes": raw_bytes, "outputBytes": target.stat().st_size,
        "rawFrames": raw_frames, "sourceFps": SOURCE_FPS, "targetFps": TARGET_FPS,
        "firstSelectedSlpFrame": first, "lastSourceSlpFrame": last,
        "lastSelectedSlpFrame": selected_last, "sourceFrameStep": 3,
        "unsampledTailSourceFrames": last - selected_last, "spatialTransform": "none",
        "audio": False, "renderSeconds": round(elapsed, 3),
        "rawChunkSourceFrames": chunk_frames, "rawChunks": len(parts),
        "concatFallbackReencoded": concat_reencoded,
        "gameplaySeconds": expected_frames / TARGET_FPS, "cpuOnly": True,
        "rendererSnapshot": os.environ.get("SMASH_RENDERER_SNAPSHOT", "unknown"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    bundle = work / "result.tar"
    with tarfile.open(bundle, "w") as archive:
        archive.add(target, arcname="video.mp4", recursive=False)
        archive.add(metadata_path, arcname="metadata.json", recursive=False)
    return bundle
