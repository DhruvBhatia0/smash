#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frame_queue.hf_storage import (
    HfDatasetStore,
    HfStorageConfig,
    HfStorageError,
    LocalFileCollector,
    ProcessedFramePublisher,
    join_repo_path,
    normalize_repo_path,
)
from frame_queue.jobs import safe_slug, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_NODE = (
    "/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
)


@dataclass(frozen=True)
class ManifestReplay:
    replay_id: str
    url: str
    file_name: str
    output_dir: str
    expected_sha256: str | None
    source: str | None

    @classmethod
    def from_entry(cls, entry: dict[str, Any], defaults: dict[str, Any]) -> "ManifestReplay":
        url = entry.get("url")
        if not url:
            raise ValueError("Replay manifest entry is missing url")
        inferred_name = Path(urllib.parse.urlparse(url).path).name or "replay.slp"
        file_name = entry.get("fileName") or inferred_name
        if not file_name.endswith(".slp"):
            file_name = f"{file_name}.slp"
        replay_id = safe_slug(entry.get("id") or Path(file_name).stem)
        return cls(
            replay_id=replay_id,
            url=url,
            file_name=file_name,
            output_dir=entry.get("outputDir") or defaults.get("outputDir") or "replays/downloaded",
            expected_sha256=entry.get("sha256") or entry.get("expectedSha256"),
            source=entry.get("source") or defaults.get("source"),
        )

    def repo_slp_path(self, prefix: str) -> str:
        return join_repo_path(prefix, self.replay_id, self.file_name)

    def repo_metadata_path(self, prefix: str) -> str:
        return join_repo_path(prefix, f"{self.replay_id}.metadata.json")


class DownloadManifest:
    def __init__(self, *, path: Path, defaults: dict[str, Any], replays: list[ManifestReplay]):
        self.path = path
        self.defaults = defaults
        self.replays = replays

    @classmethod
    def load(cls, path: Path) -> "DownloadManifest":
        payload = json.loads(path.read_text())
        defaults = payload.get("defaults") or {}
        entries = payload.get("replays") or payload.get("downloads") or []
        if not entries:
            raise ValueError(f"Manifest has no replays: {path}")
        replays = [ManifestReplay.from_entry(entry, defaults) for entry in entries]
        seen: set[str] = set()
        for replay in replays:
            if replay.replay_id in seen:
                raise ValueError(f"Duplicate replay id in manifest: {replay.replay_id}")
            seen.add(replay.replay_id)
        return cls(path=path, defaults=defaults, replays=replays)


class HfRawSlpUploader:
    def __init__(self, *, store: HfDatasetStore, path_prefix: str, root_dir: Path):
        self.store = store
        self.path_prefix = normalize_repo_path(path_prefix)
        self.root_dir = root_dir.resolve()

    def upload_inputs(
        self,
        inputs: list[Path],
        *,
        recursive: bool,
        max_files: int | None,
    ) -> dict[str, Any]:
        files = LocalFileCollector.slp_files(
            inputs,
            root_dir=self.root_dir,
            recursive=recursive,
            max_files=max_files,
        )
        results = [self.upload_one(path) for path in files]
        return {
            "status": "planned" if self.store.dry_run else "uploaded",
            "repoId": self.store.config.repo_id,
            "fileCount": len(files),
            "uploads": results,
        }

    def upload_one(self, slp_path: Path) -> dict[str, Any]:
        try:
            relative = slp_path.resolve().relative_to(self.root_dir).as_posix()
        except ValueError:
            relative = slp_path.name
        repo_path = join_repo_path(self.path_prefix, relative)
        return self.store.upload_file(
            local_path=slp_path,
            repo_path=repo_path,
            commit_message=f"Upload raw SLP {slp_path.name}",
        )


class HfManifestMirror:
    def __init__(
        self,
        *,
        store: HfDatasetStore,
        slp_prefix: str,
        metadata_prefix: str,
        node_bin: str,
        root_dir: Path,
        concurrency: int,
        keep_temp: bool,
    ):
        self.store = store
        self.slp_prefix = normalize_repo_path(slp_prefix)
        self.metadata_prefix = normalize_repo_path(metadata_prefix)
        self.node_bin = node_bin
        self.root_dir = root_dir.resolve()
        self.concurrency = max(1, concurrency)
        self.keep_temp = keep_temp

    def mirror(self, manifest: DownloadManifest, *, max_jobs: int | None) -> dict[str, Any]:
        jobs = manifest.replays[:max_jobs] if max_jobs is not None else manifest.replays
        if not jobs:
            return {
                "status": "planned" if self.store.dry_run else "mirrored",
                "repoId": self.store.config.repo_id,
                "manifestPath": str(manifest.path),
                "count": 0,
                "results": [],
            }
        results: list[dict[str, Any]] = []
        worker_count = min(self.concurrency, len(jobs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_job = {pool.submit(self.mirror_one, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append(
                        {
                            "id": job.replay_id,
                            "status": "failed",
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        }
                    )

        results.sort(key=lambda item: item["id"])
        return {
            "status": "planned" if self.store.dry_run else "mirrored",
            "repoId": self.store.config.repo_id,
            "manifestPath": str(manifest.path),
            "count": len(results),
            "results": results,
        }

    def mirror_one(self, replay: ManifestReplay) -> dict[str, Any]:
        temp_parent = Path(tempfile.mkdtemp(prefix=f"hf-slp-{replay.replay_id}-"))
        slp_path = temp_parent / replay.file_name
        metadata_path = temp_parent / f"{replay.replay_id}.metadata.json"
        started = time.monotonic()
        try:
            download = self.download_replay(replay, slp_path)
            metadata = self.extract_metadata(replay, slp_path, metadata_path)
            uploads = [
                self.store.upload_file(
                    local_path=slp_path,
                    repo_path=replay.repo_slp_path(self.slp_prefix),
                    commit_message=f"Mirror raw SLP {replay.replay_id}",
                ),
                self.store.upload_file(
                    local_path=metadata_path,
                    repo_path=replay.repo_metadata_path(self.metadata_prefix),
                    commit_message=f"Mirror SLP metadata {replay.replay_id}",
                ),
            ]
            return {
                "id": replay.replay_id,
                "status": "planned" if self.store.dry_run else "mirrored",
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "download": download,
                "metadata": metadata,
                "uploads": uploads,
            }
        finally:
            if not self.keep_temp:
                shutil.rmtree(temp_parent, ignore_errors=True)

    def download_replay(self, replay: ManifestReplay, output_path: Path) -> dict[str, Any]:
        if self.store.dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"")
            return {
                "status": "planned",
                "url": replay.url,
                "outputPath": str(output_path),
                "expectedSha256": replay.expected_sha256,
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sha256 = hashlib.sha256()
        bytes_written = 0
        request = urllib.request.Request(
            replay.url,
            headers={"user-agent": "smash-frame-data-experiment"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with output_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    sha256.update(chunk)
                    bytes_written += len(chunk)
                    handle.write(chunk)

        digest = sha256.hexdigest()
        if replay.expected_sha256 and replay.expected_sha256 != digest:
            raise ValueError(
                f"SHA-256 mismatch for {replay.replay_id}: expected "
                f"{replay.expected_sha256}, got {digest}"
            )
        return {
            "status": "downloaded",
            "url": replay.url,
            "outputPath": str(output_path),
            "bytes": bytes_written,
            "sha256": digest,
        }

    def extract_metadata(
        self,
        replay: ManifestReplay,
        slp_path: Path,
        metadata_path: Path,
    ) -> dict[str, Any]:
        if self.store.dry_run:
            write_json(
                metadata_path,
                {
                    "schemaVersion": 1,
                    "id": replay.replay_id,
                    "dryRun": True,
                    "download": {
                        "url": replay.url,
                        "source": replay.source,
                    },
                },
            )
            return {"status": "planned", "outputPath": str(metadata_path)}

        completed = subprocess.run(
            [
                self.node_bin,
                "scripts/extract-slp-metadata.mjs",
                str(slp_path),
                str(metadata_path),
            ],
            cwd=str(self.root_dir),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Metadata extraction failed for {replay.replay_id}: "
                f"{completed.stderr[-2000:]}"
            )
        payload = json.loads(metadata_path.read_text())
        payload["download"] = {
            "id": replay.replay_id,
            "url": replay.url,
            "source": replay.source,
        }
        write_json(metadata_path, payload)
        return {
            "status": "extracted",
            "outputPath": str(metadata_path),
            "stage": payload.get("game", {}).get("stage"),
            "winner": payload.get("winner"),
        }

class HfStorageCli:
    def __init__(self, argv: list[str]):
        self.argv = argv

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Mirror raw SLPs and processed frame outputs to Hugging Face datasets.",
        )
        parser.add_argument("--repo", default=os.environ.get("SMASH_HF_REPO", ""))
        parser.add_argument(
            "--token",
            default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", ""),
        )
        parser.add_argument(
            "--expected-email",
            default=os.environ.get("SMASH_HF_EXPECTED_EMAIL", ""),
            help="Fail real HF operations unless the token belongs to this email.",
        )
        parser.add_argument("--public", action="store_true", help="Create repo as public.")
        parser.add_argument(
            "--no-create",
            action="store_true",
            help="Do not attempt to create the target dataset repo.",
        )
        parser.add_argument("--dry-run", action="store_true")

        subparsers = parser.add_subparsers(dest="command", required=True)

        subparsers.add_parser("whoami")

        upload_slp = subparsers.add_parser("upload-slp")
        upload_slp.add_argument("inputs", nargs="+")
        upload_slp.add_argument("--path-prefix", default="raw/slp")
        upload_slp.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
        upload_slp.add_argument("--max-files", type=int, default=None)

        mirror = subparsers.add_parser("mirror-manifest")
        mirror.add_argument("--manifest", default="download-manifests/slippi-js-samples.json")
        mirror.add_argument("--slp-prefix", default="raw/slp")
        mirror.add_argument("--metadata-prefix", default="raw/metadata")
        mirror.add_argument("--node-bin", default=DEFAULT_NODE)
        mirror.add_argument("--concurrency", type=int, default=1)
        mirror.add_argument("--max-jobs", type=int, default=None)
        mirror.add_argument("--keep-temp", action="store_true")

        processed = subparsers.add_parser("upload-processed-run")
        processed.add_argument("--run-dir", required=True)
        processed.add_argument("--path-prefix", default="processed/frame-queues")
        processed.add_argument(
            "--include-raw-frames",
            action=argparse.BooleanOptionalAction,
            default=True,
        )

        download = subparsers.add_parser("download-slp")
        download.add_argument("--repo-path", required=True)
        download.add_argument("--local-dir", default="hf-downloads")
        download.add_argument("--force", action="store_true")

        snapshot = subparsers.add_parser("snapshot-slp")
        snapshot.add_argument("--local-dir", default="hf-downloads")
        snapshot.add_argument("--allow-pattern", action="append", default=["raw/slp/**/*.slp"])

        return parser

    def run(self) -> int:
        args = self.parser().parse_args(self.argv)
        if not args.repo and args.command != "whoami":
            raise SystemExit("--repo or SMASH_HF_REPO is required")

        store = HfDatasetStore(
            HfStorageConfig(
                repo_id=args.repo or "identity-check",
                token=args.token or None,
                expected_email=args.expected_email or None,
                private=not args.public,
                allow_create=not args.no_create,
                dry_run=args.dry_run,
            )
        )

        if args.command == "whoami":
            result = store.verify_expected_identity() if args.expected_email else store.token_identity()
        elif args.command == "upload-slp":
            uploader = HfRawSlpUploader(
                store=store,
                path_prefix=args.path_prefix,
                root_dir=ROOT_DIR,
            )
            inputs = [Path(value) for value in args.inputs]
            result = uploader.upload_inputs(
                inputs,
                recursive=args.recursive,
                max_files=args.max_files,
            )
        elif args.command == "mirror-manifest":
            manifest_path = Path(args.manifest)
            if not manifest_path.is_absolute():
                manifest_path = ROOT_DIR / manifest_path
            manifest = DownloadManifest.load(manifest_path)
            mirror = HfManifestMirror(
                store=store,
                slp_prefix=args.slp_prefix,
                metadata_prefix=args.metadata_prefix,
                node_bin=args.node_bin,
                root_dir=ROOT_DIR,
                concurrency=args.concurrency,
                keep_temp=args.keep_temp,
            )
            result = mirror.mirror(manifest, max_jobs=args.max_jobs)
        elif args.command == "upload-processed-run":
            run_dir = Path(args.run_dir)
            if not run_dir.is_absolute():
                run_dir = ROOT_DIR / run_dir
            run_id = run_dir.resolve().name
            publisher = ProcessedFramePublisher(
                store=store,
                path_prefix=args.path_prefix,
                include_raw_frames=args.include_raw_frames,
            )
            job_uploads = []
            jobs_dir = run_dir / "jobs"
            if jobs_dir.exists():
                for job_dir in sorted(path for path in jobs_dir.iterdir() if path.is_dir()):
                    ignore = None if args.include_raw_frames else ["raw-frames/**"]
                    job_uploads.append(
                        store.upload_directory(
                            local_dir=job_dir,
                            repo_prefix=join_repo_path(
                                args.path_prefix,
                                run_id,
                                "jobs",
                                job_dir.name,
                            ),
                            ignore_patterns=ignore,
                            commit_message=f"Upload processed frame job {job_dir.name}",
                        )
                    )
            run_upload = publisher.publish_run_artifacts(run_id=run_id, run_dir=run_dir)
            result = {
                "status": "planned" if store.dry_run else "uploaded",
                "repoId": store.config.repo_id,
                "runId": run_id,
                "jobUploads": job_uploads,
                "runUpload": run_upload,
            }
        elif args.command == "download-slp":
            local_dir = Path(args.local_dir)
            if not local_dir.is_absolute():
                local_dir = ROOT_DIR / local_dir
            result = store.download_file(
                repo_path=args.repo_path,
                local_dir=local_dir,
                force_download=args.force,
            )
        elif args.command == "snapshot-slp":
            local_dir = Path(args.local_dir)
            if not local_dir.is_absolute():
                local_dir = ROOT_DIR / local_dir
            result = store.snapshot(local_dir=local_dir, allow_patterns=args.allow_pattern)
        else:
            raise SystemExit(f"Unknown command: {args.command}")

        print(json.dumps(result, indent=2, sort_keys=True))
        return 0


if __name__ == "__main__":
    try:
        argv = sys.argv[1:]
        if argv and argv[0] == "--":
            argv = argv[1:]
        raise SystemExit(HfStorageCli(argv).run())
    except HfStorageError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
