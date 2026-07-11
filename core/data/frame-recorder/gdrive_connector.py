from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


class GDriveConnector:
    """Rclone-backed Google Drive storage with compact SLP archive support."""

    kind = "gdrive"
    index_suffix = ".slp-index.jsonl"

    def __init__(self, config_path: str | None = None, rclone: str | None = None):
        self.config_path = Path(
            config_path or os.environ.get("SMASH_GDRIVE_CONFIG", "~/.config/rclone/rclone.conf")
        ).expanduser()
        self.rclone = rclone or os.environ.get("SMASH_RCLONE_BIN") or shutil.which("rclone")
        self.zstd = os.environ.get("SMASH_ZSTD_BIN") or shutil.which("zstd")
        self.index_workers = max(1, int(os.environ.get("SMASH_GDRIVE_INDEX_WORKERS", "6")))
        if not self.rclone:
            raise FileNotFoundError("rclone is required for Google Drive storage")
        if not self.config_path.exists():
            raise FileNotFoundError(self.config_path)

    def prepare(self, namespace: str) -> None:
        """Verify that the configured rclone Drive remote is accessible."""
        self._run("lsd", self._remote(namespace))

    def list_files(self, namespace: str, folder: str = "") -> list[str]:
        """List files recursively under one Drive folder."""
        prefix = folder.strip("/")
        completed = self._run(
            "lsf",
            self._remote(namespace, prefix),
            "--recursive",
            "--files-only",
            "--format",
            "p",
            check=False,
        )
        if completed.returncode:
            message = completed.stderr.lower()
            if "directory not found" in message or "object not found" in message:
                return []
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return sorted(self._join(prefix, line) for line in completed.stdout.splitlines() if line)

    def list_slp_references(self, namespace: str, folder: str = "") -> list[str]:
        """Return loose SLP paths and indexed tar.zst member references."""
        files = self.list_files(namespace, folder)
        references = [path for path in files if path.lower().endswith(".slp")]
        archives = [path for path in files if path.lower().endswith(".tar.zst")]
        known = set(files)

        def load(archive: str) -> list[str]:
            index_path = archive + self.index_suffix
            if index_path not in known:
                self._build_archive_index(namespace, archive, index_path)
            return self._read_archive_index(namespace, archive, index_path)

        if archives:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.index_workers, len(archives))
            ) as executor:
                for archive_references in executor.map(load, archives):
                    references.extend(archive_references)
        return sorted(references)

    def worker_config(self, namespace: str) -> dict:
        """Return pod-side rclone settings; the config itself is copied separately."""
        return {
            "kind": self.kind,
            "remote": namespace.rstrip(":"),
            "config": "/root/.config/rclone/rclone.conf",
        }

    def _read_archive_index(self, namespace: str, archive: str, index_path: str) -> list[str]:
        completed = self._run("cat", self._remote(namespace, index_path))
        references = []
        for line in completed.stdout.splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("archive") != archive or not str(row.get("member", "")).endswith(".slp"):
                raise ValueError(f"invalid Drive SLP index row in {index_path}: {row}")
            references.append(f"{archive}::{row['member']}")
        return references

    def _build_archive_index(self, namespace: str, archive: str, index_path: str) -> None:
        if not self.zstd:
            raise FileNotFoundError("zstd is required to index Drive tar.zst sources")
        rclone = subprocess.Popen(
            [
                self.rclone,
                "cat",
                self._remote(namespace, archive),
                "--config",
                str(self.config_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert rclone.stdout is not None
        zstd = subprocess.Popen(
            [self.zstd, "-dc"],
            stdin=rclone.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        rclone.stdout.close()
        assert zstd.stdout is not None
        rows = []
        try:
            with tarfile.open(fileobj=zstd.stdout, mode="r|") as source:
                for member in source:
                    if member.isfile() and member.name.lower().endswith(".slp"):
                        rows.append({"archive": archive, "member": member.name})
            zstd_error = zstd.stderr.read().decode(errors="replace") if zstd.stderr else ""
            rclone_error = rclone.stderr.read().decode(errors="replace") if rclone.stderr else ""
            zstd_status = zstd.wait()
            rclone_status = rclone.wait()
            if zstd_status or rclone_status:
                raise RuntimeError(zstd_error.strip() or rclone_error.strip() or f"failed to index {archive}")
        except BaseException:
            zstd.terminate()
            rclone.terminate()
            zstd.wait()
            rclone.wait()
            raise
        if not rows:
            raise RuntimeError(f"Drive source archive contains no SLP files: {archive}")

        with tempfile.NamedTemporaryFile("w", suffix=self.index_suffix, delete=False) as output:
            local_index = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
        try:
            self._run("copyto", str(local_index), self._remote(namespace, index_path))
        finally:
            local_index.unlink(missing_ok=True)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [self.rclone, *args, "--config", str(self.config_path)],
            text=True,
            capture_output=True,
        )
        if check and completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed

    def _remote(self, namespace: str, path: str = "") -> str:
        return f"{namespace.rstrip(':')}:{path.strip('/')}"

    def _join(self, *parts: str) -> str:
        return "/".join(part.strip("/") for part in parts if part.strip("/"))
