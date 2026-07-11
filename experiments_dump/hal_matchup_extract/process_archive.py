#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import melee
import peppi_py
from tqdm import tqdm

# HAL's top-level package imports its simulator and training stack. This worker
# only needs the data modules, so expose the checkout as a namespace package.
hal_package_dir = Path(os.environ.get("HAL_PACKAGE_DIR", "/workspace/hal/hal"))
if hal_package_dir.is_dir():
    hal_package = types.ModuleType("hal")
    hal_package.__path__ = [str(hal_package_dir)]
    sys.modules["hal"] = hal_package

from hal.data.archive import archive_member_path, iter_archive_members, list_archive_slps, parse_archive_member_path
from hal.data.index import ReplayIndexEntry, extract_index_entry, read_jsonl, write_jsonl
from hal.scripts.build_index import build_index
from hal.wire import slp_character_to_libmelee


BATTLEFIELD = 31
CHARACTERS = {int(melee.Character.FOX.value), int(melee.Character.CPTFALCON.value)}
_ZIP_ARCHIVE: zipfile.ZipFile | None = None
_ZIP_ARCHIVE_PATH: Path | None = None
_ZIP_TMPFS: Path | None = None


def is_target(entry: ReplayIndexEntry) -> bool:
    return (
        entry.stage == BATTLEFIELD
        and len(entry.players) == 2
        and {player.character for player in entry.players} == CHARACTERS
    )


def output_name(member: str) -> str:
    basename = Path(member).name.removesuffix(".gz")
    digest = hashlib.sha1(member.encode()).hexdigest()[:16]
    return f"{digest}__{basename}"


def init_zip_worker(archive: str, tmpfs: str) -> None:
    global _ZIP_ARCHIVE, _ZIP_ARCHIVE_PATH, _ZIP_TMPFS
    _ZIP_ARCHIVE_PATH = Path(archive)
    _ZIP_ARCHIVE = zipfile.ZipFile(archive)
    _ZIP_TMPFS = Path(tmpfs)


def index_zip_member(member: str) -> ReplayIndexEntry | None:
    assert _ZIP_ARCHIVE is not None and _ZIP_ARCHIVE_PATH is not None and _ZIP_TMPFS is not None
    fd, temporary_name = tempfile.mkstemp(dir=_ZIP_TMPFS, suffix=".slp")
    temporary = Path(temporary_name)
    try:
        payload = _ZIP_ARCHIVE.read(member)
        if member.endswith(".gz"):
            payload = gzip.decompress(payload)
        with os.fdopen(fd, "wb") as destination:
            destination.write(payload)
        fd = -1
        synthetic = archive_member_path(_ZIP_ARCHIVE_PATH, member)
        entry = extract_index_entry(temporary, compute_sha1=False, name_hint=synthetic, with_stats=False)
        return dataclasses.replace(entry, path=synthetic) if entry is not None else None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        if fd >= 0:
            os.close(fd)
        return None
    finally:
        temporary.unlink(missing_ok=True)


def index_zip_archive(archive: Path, index_path: Path, workers: int) -> None:
    members = list_archive_slps(archive)
    tmpfs = Path("/dev/shm/hal_fast_zip")
    shutil.rmtree(tmpfs, ignore_errors=True)
    tmpfs.mkdir(parents=True)
    written = failed = 0
    batch: list[ReplayIndexEntry] = []
    context = mp.get_context("fork")
    try:
        with context.Pool(workers, initializer=init_zip_worker, initargs=(str(archive), str(tmpfs))) as pool:
            for entry in tqdm(pool.imap_unordered(index_zip_member, members, chunksize=8), total=len(members), desc="indexing"):
                if entry is None:
                    failed += 1
                    continue
                batch.append(entry)
                if len(batch) == 256:
                    write_jsonl(index_path, batch, append=True)
                    written += len(batch)
                    batch.clear()
        if batch:
            write_jsonl(index_path, batch, append=True)
            written += len(batch)
    finally:
        shutil.rmtree(tmpfs, ignore_errors=True)
    print(json.dumps({"event": "zip_index_complete", "written": written, "failures": failed}), flush=True)


def index_archive(archive: Path, index_path: Path, workers: int) -> list[ReplayIndexEntry]:
    with archive.open("rb") as source:
        is_zip = source.read(4) == b"PK\x03\x04"
    if is_zip:
        index_zip_archive(archive, index_path, workers)
    else:
        build_index(
            output=index_path,
            archive=archive,
            compute_sha1=False,
            with_stats=False,
            workers=workers,
            tmpfs_root=Path("/dev/shm/hal_build_index"),
            queue_size=workers * 2,
        )
    return list(read_jsonl(index_path))


def materialize_slps(
    archive: Path,
    entries: list[ReplayIndexEntry],
    output_dir: Path,
    workers: int,
) -> list[dict]:
    files_dir = output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    by_member: dict[str, ReplayIndexEntry] = {}
    for entry in entries:
        parsed = parse_archive_member_path(entry.path)
        if parsed is None:
            raise ValueError(f"expected archive path, got {entry.path}")
        _, member = parsed
        by_member[member] = entry

    rows = []
    completed = 0
    for synthetic_path, temporary in iter_archive_members(
        archive,
        tmpfs_root=Path("/dev/shm/hal_selected_slps"),
        filter_paths=set(by_member),
        queue_size=workers * 2,
    ):
        try:
            parsed = parse_archive_member_path(synthetic_path)
            if parsed is None:
                raise ValueError(f"expected archive path, got {synthetic_path}")
            _, member = parsed
            entry = by_member[member]
            name = output_name(member)
            target = files_dir / name
            shutil.copyfile(temporary, target)
            rows.append(
                {
                    "source": entry.path,
                    "output": f"files/{name}",
                    "bytes": target.stat().st_size,
                    "stage": entry.stage,
                    "characters": sorted(player.character for player in entry.players),
                }
            )
            completed += 1
            if completed % 1000 == 0:
                print(json.dumps({"event": "materialize_progress", "completed": completed, "total": len(entries)}), flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    if completed != len(entries):
        raise RuntimeError(f"materialized {completed} of {len(entries)} selected SLPs")
    return sorted(rows, key=lambda row: row["source"])


def verify_one(path: str) -> str | None:
    try:
        game = peppi_py.read_slippi(path, skip_frames=True)
        players = [
            player
            for player in game.start.players
            if str(getattr(player.type, "name", player.type)).upper() != "EMPTY"
        ]
        characters = {int(slp_character_to_libmelee(int(player.character)).value) for player in players}
        if int(game.start.stage) != BATTLEFIELD or len(players) != 2 or characters != CHARACTERS:
            return f"wrong settings: stage={int(game.start.stage)} players={sorted(characters)}"
        return None
    except Exception as error:
        return repr(error)


def verify_slps(output_dir: Path, workers: int) -> None:
    paths = sorted(str(path) for path in (output_dir / "files").glob("*.slp"))
    context = mp.get_context("fork")
    with context.Pool(workers) as pool:
        results = pool.imap(verify_one, paths, chunksize=16)
        failures = [(path, error) for path, error in zip(paths, results, strict=True) if error is not None]
    if failures:
        raise RuntimeError(f"verification failed for {len(failures)} SLPs; first={failures[0]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.jsonl"
    started = time.time()

    archive_members = len(list_archive_slps(args.archive))
    entries = index_archive(args.archive, index_path, args.workers)
    selected = [entry for entry in entries if is_target(entry)]
    paths_path = args.output / "paths.txt"
    paths_path.write_text("\n".join(sorted(entry.path for entry in selected)) + ("\n" if selected else ""))
    print(json.dumps({"event": "filter_complete", "indexed": len(entries), "selected": len(selected)}), flush=True)

    rows = materialize_slps(args.archive, selected, args.output, args.workers)
    verify_slps(args.output, args.workers)
    with (args.output / "manifest.jsonl").open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "status": "complete",
        "archive": args.archive.name,
        "workers": args.workers,
        "archiveMembers": archive_members,
        "indexed": len(entries),
        "parseFailures": archive_members - len(entries),
        "selected": len(selected),
        "selectedBytes": sum(row["bytes"] for row in rows),
        "elapsedSeconds": round(time.time() - started, 3),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
