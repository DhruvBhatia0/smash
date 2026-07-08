from __future__ import annotations

import fnmatch
import json
import os
import shutil
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .jobs import FrameRenderJob


class HfStorageError(RuntimeError):
    pass


def _patterns(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def normalize_repo_path(value: str | os.PathLike[str] | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\\", "/").strip("/")
    if not text:
        return ""
    path = PurePosixPath(text)
    parts = [part for part in path.parts if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise HfStorageError(f"Repository paths cannot escape upward: {value}")
    return "/".join(parts)


def join_repo_path(*parts: str | os.PathLike[str] | None) -> str:
    normalized: list[str] = []
    for part in parts:
        value = normalize_repo_path(part)
        if value:
            normalized.extend(value.split("/"))
    return "/".join(normalized)


def _commit_info_to_json(value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"repr": repr(value)}
    for attr in ("commit_url", "commit_message", "oid", "pr_url"):
        attr_value = getattr(value, attr, None)
        if attr_value is not None:
            payload[attr] = attr_value
    return payload


@dataclass(frozen=True)
class TreeStats:
    file_count: int
    byte_count: int

    def to_json(self) -> dict[str, int]:
        return {
            "fileCount": self.file_count,
            "byteCount": self.byte_count,
        }


class LocalFileCollector:
    @staticmethod
    def slp_files(
        inputs: list[Path],
        *,
        root_dir: Path,
        recursive: bool = True,
        max_files: int | None = None,
    ) -> list[Path]:
        files: list[Path] = []
        for input_path in inputs:
            resolved = input_path.expanduser()
            if not resolved.is_absolute():
                resolved = root_dir / resolved
            resolved = resolved.resolve()
            if resolved.is_file() and resolved.suffix.lower() == ".slp":
                files.append(resolved)
            elif resolved.is_dir():
                pattern = "**/*.slp" if recursive else "*.slp"
                files.extend(path.resolve() for path in resolved.glob(pattern) if path.is_file())
            else:
                raise FileNotFoundError(
                    f"Input path is not an .slp file or directory: {resolved}"
                )

        unique = sorted(dict.fromkeys(files))
        if max_files is not None:
            return unique[:max_files]
        return unique

    @staticmethod
    def tree_stats(
        path: Path,
        *,
        allow_patterns: str | Iterable[str] | None = None,
        ignore_patterns: str | Iterable[str] | None = None,
    ) -> TreeStats:
        allow = _patterns(allow_patterns)
        ignore = _patterns(ignore_patterns)
        files = 0
        bytes_total = 0

        if path.is_file():
            if LocalFileCollector._included(path.name, allow=allow, ignore=ignore):
                return TreeStats(file_count=1, byte_count=path.stat().st_size)
            return TreeStats(file_count=0, byte_count=0)

        for child in path.rglob("*"):
            if not child.is_file():
                continue
            relative = child.relative_to(path).as_posix()
            if not LocalFileCollector._included(relative, allow=allow, ignore=ignore):
                continue
            files += 1
            bytes_total += child.stat().st_size
        return TreeStats(file_count=files, byte_count=bytes_total)

    @staticmethod
    def _included(relative_path: str, *, allow: list[str], ignore: list[str]) -> bool:
        if allow and not any(fnmatch.fnmatch(relative_path, pattern) for pattern in allow):
            return False
        if ignore and any(fnmatch.fnmatch(relative_path, pattern) for pattern in ignore):
            return False
        return True


@dataclass(frozen=True)
class HfStorageConfig:
    repo_id: str
    repo_type: str = "dataset"
    revision: str | None = None
    token: str | None = None
    expected_email: str | None = None
    private: bool = True
    allow_create: bool = True
    dry_run: bool = False


class HfDatasetStore:
    def __init__(self, config: HfStorageConfig):
        if not config.repo_id:
            raise HfStorageError("A Hugging Face repo id is required.")
        self.config = config
        self._api: Any | None = None
        self._ensure_lock = threading.Lock()
        self._ensure_result: dict[str, Any] | None = None
        self._identity_lock = threading.Lock()
        self._identity_result: dict[str, Any] | None = None

    @property
    def dry_run(self) -> bool:
        return self.config.dry_run

    @property
    def api(self) -> Any:
        if self._api is None:
            try:
                from huggingface_hub import HfApi
            except ModuleNotFoundError as error:
                raise HfStorageError(
                    "Missing dependency: install with `python3 -m pip install -r requirements.txt`."
                ) from error
            self._api = HfApi(token=self.config.token)
        return self._api

    def token_identity(self) -> dict[str, Any]:
        if self.dry_run:
            return {
                "status": "planned",
                "expectedEmail": self.config.expected_email,
            }
        with self._identity_lock:
            if self._identity_result is not None:
                return self._identity_result
            token = self.config.token
            if not token:
                raise HfStorageError(
                    "HF identity verification requires an explicit token. "
                    "Set HF_TOKEN or pass --token."
                )
            request = urllib.request.Request(
                "https://huggingface.co/api/whoami-v2",
                headers={"authorization": f"Bearer {token}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                raise HfStorageError(f"HF token verification failed: HTTP {error.code}") from error
            auth = payload.get("auth") or {}
            access_token = auth.get("accessToken") or {}
            self._identity_result = {
                "status": "verified",
                "name": payload.get("name"),
                "fullname": payload.get("fullname"),
                "email": payload.get("email"),
                "type": payload.get("type"),
                "tokenDisplayName": access_token.get("displayName"),
                "tokenRole": access_token.get("role"),
            }
            return self._identity_result

    def verify_expected_identity(self) -> dict[str, Any]:
        expected = (self.config.expected_email or "").strip().lower()
        if not expected:
            return {"status": "skipped", "reason": "no expected HF email configured"}
        identity = self.token_identity()
        if self.dry_run:
            return identity
        actual = str(identity.get("email") or "").strip().lower()
        if actual != expected:
            raise HfStorageError(
                f"HF token email mismatch: expected {self.config.expected_email}, got "
                f"{identity.get('email') or 'unknown'} for user {identity.get('name') or 'unknown'}."
            )
        return identity

    def ensure_repo(self) -> dict[str, Any]:
        if self.dry_run:
            return {
                "status": "planned",
                "repoId": self.config.repo_id,
                "repoType": self.config.repo_type,
                "private": self.config.private,
                "allowCreate": self.config.allow_create,
            }
        with self._ensure_lock:
            if self._ensure_result is not None:
                return self._ensure_result
            identity = self.verify_expected_identity()
            if not self.config.allow_create:
                self._ensure_result = {
                    "status": "skipped",
                    "reason": "repo creation disabled",
                    "repoId": self.config.repo_id,
                    "identity": identity,
                }
                return self._ensure_result
            result = self.api.create_repo(
                repo_id=self.config.repo_id,
                repo_type=self.config.repo_type,
                private=self.config.private,
                exist_ok=True,
                token=self.config.token,
            )
            self._ensure_result = {
                "status": "ensured",
                "repoId": self.config.repo_id,
                "repoType": self.config.repo_type,
                "url": str(result),
                "identity": identity,
            }
            return self._ensure_result

    def upload_file(
        self,
        *,
        local_path: Path,
        repo_path: str,
        commit_message: str | None = None,
    ) -> dict[str, Any]:
        local_path = local_path.expanduser().resolve()
        if not local_path.is_file():
            raise FileNotFoundError(f"Cannot upload missing file: {local_path}")
        normalized_repo_path = normalize_repo_path(repo_path)
        stats = LocalFileCollector.tree_stats(local_path)

        payload = {
            "status": "planned" if self.dry_run else "uploaded",
            "operation": "upload_file",
            "repoId": self.config.repo_id,
            "repoType": self.config.repo_type,
            "repoPath": normalized_repo_path,
            "localPath": str(local_path),
            **stats.to_json(),
        }
        if self.dry_run:
            return payload

        self.ensure_repo()
        result = self.api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=normalized_repo_path,
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            revision=self.config.revision,
            token=self.config.token,
            commit_message=commit_message,
        )
        payload["commit"] = _commit_info_to_json(result)
        return payload

    def upload_directory(
        self,
        *,
        local_dir: Path,
        repo_prefix: str,
        allow_patterns: str | Iterable[str] | None = None,
        ignore_patterns: str | Iterable[str] | None = None,
        commit_message: str | None = None,
    ) -> dict[str, Any]:
        local_dir = local_dir.expanduser().resolve()
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Cannot upload missing directory: {local_dir}")
        normalized_repo_prefix = normalize_repo_path(repo_prefix)
        stats = LocalFileCollector.tree_stats(
            local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        payload: dict[str, Any] = {
            "status": "planned" if self.dry_run else "uploaded",
            "operation": "upload_folder",
            "repoId": self.config.repo_id,
            "repoType": self.config.repo_type,
            "repoPath": normalized_repo_prefix,
            "localPath": str(local_dir),
            **stats.to_json(),
        }
        if allow_patterns:
            payload["allowPatterns"] = _patterns(allow_patterns)
        if ignore_patterns:
            payload["ignorePatterns"] = _patterns(ignore_patterns)
        if self.dry_run:
            return payload

        self.ensure_repo()
        result = self.api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=normalized_repo_prefix,
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            revision=self.config.revision,
            token=self.config.token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            commit_message=commit_message,
        )
        payload["commit"] = _commit_info_to_json(result)
        return payload

    def download_file(
        self,
        *,
        repo_path: str,
        local_dir: Path,
        force_download: bool = False,
    ) -> dict[str, Any]:
        normalized_repo_path = normalize_repo_path(repo_path)
        local_dir = local_dir.expanduser().resolve()
        payload = {
            "status": "planned" if self.dry_run else "downloaded",
            "operation": "download_file",
            "repoId": self.config.repo_id,
            "repoType": self.config.repo_type,
            "repoPath": normalized_repo_path,
            "localDir": str(local_dir),
        }
        if self.dry_run:
            return payload

        self.verify_expected_identity()
        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as error:
            raise HfStorageError(
                "Missing dependency: install with `python3 -m pip install -r requirements.txt`."
            ) from error

        local_dir.mkdir(parents=True, exist_ok=True)
        output = hf_hub_download(
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            revision=self.config.revision,
            token=self.config.token,
            filename=normalized_repo_path,
            local_dir=str(local_dir),
            force_download=force_download,
        )
        payload["outputPath"] = output
        return payload

    def snapshot(
        self,
        *,
        local_dir: Path,
        allow_patterns: str | Iterable[str] | None = None,
        ignore_patterns: str | Iterable[str] | None = None,
    ) -> dict[str, Any]:
        local_dir = local_dir.expanduser().resolve()
        payload = {
            "status": "planned" if self.dry_run else "downloaded",
            "operation": "snapshot_download",
            "repoId": self.config.repo_id,
            "repoType": self.config.repo_type,
            "localDir": str(local_dir),
        }
        if allow_patterns:
            payload["allowPatterns"] = _patterns(allow_patterns)
        if ignore_patterns:
            payload["ignorePatterns"] = _patterns(ignore_patterns)
        if self.dry_run:
            return payload

        self.verify_expected_identity()
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as error:
            raise HfStorageError(
                "Missing dependency: install with `python3 -m pip install -r requirements.txt`."
            ) from error

        local_dir.mkdir(parents=True, exist_ok=True)
        output = snapshot_download(
            repo_id=self.config.repo_id,
            repo_type=self.config.repo_type,
            revision=self.config.revision,
            token=self.config.token,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        payload["outputPath"] = output
        return payload


class ProcessedFramePublisher:
    def __init__(
        self,
        *,
        store: HfDatasetStore,
        path_prefix: str,
        delete_local_after_upload: bool = False,
        include_raw_frames: bool = True,
    ):
        self.store = store
        self.path_prefix = normalize_repo_path(path_prefix)
        self.delete_local_after_upload = delete_local_after_upload
        self.include_raw_frames = include_raw_frames

    @classmethod
    def from_args(cls, args: Any) -> "ProcessedFramePublisher | None":
        if not getattr(args, "upload_processed_to_hf", False):
            return None
        repo_id = getattr(args, "hf_processed_repo", "")
        if not repo_id:
            raise HfStorageError(
                "--hf-processed-repo is required when --upload-processed-to-hf is set"
            )
        token = getattr(args, "hf_token", "") or os.environ.get("HF_TOKEN")
        if not token:
            token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        store = HfDatasetStore(
            HfStorageConfig(
                repo_id=repo_id,
                token=token,
                expected_email=getattr(args, "hf_expected_email", ""),
                private=getattr(args, "hf_private", True),
                allow_create=getattr(args, "hf_create_repo", True),
                dry_run=getattr(args, "hf_dry_run", False),
            )
        )
        return cls(
            store=store,
            path_prefix=getattr(args, "hf_processed_prefix", "processed/frame-queues"),
            delete_local_after_upload=getattr(args, "delete_local_after_hf_upload", False),
            include_raw_frames=getattr(args, "hf_include_raw_frames", True),
        )

    def repo_job_prefix(self, job: FrameRenderJob) -> str:
        return join_repo_path(self.path_prefix, job.run_id, "jobs", job.job_id)

    def repo_run_prefix(self, run_id: str) -> str:
        return join_repo_path(self.path_prefix, run_id)

    def publish_job(self, job: FrameRenderJob) -> dict[str, Any]:
        ignore_patterns: list[str] = []
        if not self.include_raw_frames:
            ignore_patterns.extend(["raw-frames/**"])
        return self.store.upload_directory(
            local_dir=job.job_dir,
            repo_prefix=self.repo_job_prefix(job),
            ignore_patterns=ignore_patterns or None,
            commit_message=f"Upload processed frame job {job.job_id}",
        )

    def publish_job_result(self, job: FrameRenderJob) -> dict[str, Any]:
        return self.store.upload_file(
            local_path=job.result_path,
            repo_path=join_repo_path(self.repo_job_prefix(job), "result.json"),
            commit_message=f"Refresh processed frame result {job.job_id}",
        )

    def publish_run_artifacts(self, *, run_id: str, run_dir: Path) -> dict[str, Any]:
        run_dir = run_dir.resolve()
        uploads: list[dict[str, Any]] = []
        for file_name in ("manifest.json", "index.jsonl"):
            local_path = run_dir / file_name
            if local_path.exists():
                uploads.append(
                    self.store.upload_file(
                        local_path=local_path,
                        repo_path=join_repo_path(self.repo_run_prefix(run_id), file_name),
                        commit_message=f"Upload frame queue {run_id} {file_name}",
                    )
                )

        results_dir = run_dir / "results"
        if results_dir.exists():
            uploads.append(
                self.store.upload_directory(
                    local_dir=results_dir,
                    repo_prefix=join_repo_path(self.repo_run_prefix(run_id), "results"),
                    commit_message=f"Upload frame queue {run_id} result summaries",
                )
            )

        return {
            "status": "planned" if self.store.dry_run else "uploaded",
            "repoId": self.store.config.repo_id,
            "repoPath": self.repo_run_prefix(run_id),
            "uploads": uploads,
        }

    def cleanup_job(self, job: FrameRenderJob) -> dict[str, Any]:
        if not self.delete_local_after_upload:
            return {"status": "skipped", "reason": "delete-local disabled"}
        if not job.job_dir.exists():
            return {"status": "skipped", "reason": "job directory already absent"}
        children = [child for child in job.job_dir.iterdir() if child.name != "result.json"]
        if self.store.dry_run:
            return {
                "status": "planned",
                "localPath": str(job.job_dir),
                "deleteCount": len(children),
            }
        deleted: list[str] = []
        for child in children:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            deleted.append(str(child))
        return {
            "status": "pruned",
            "localPath": str(job.job_dir),
            "deleteCount": len(deleted),
            "deleted": deleted[:20],
        }
