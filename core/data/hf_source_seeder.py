from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_url

from .hf_connector import HfConnector
from .models import HfLocation, log_event


@dataclass(frozen=True)
class SourceSlp:
    """One source replay and its target HF path."""

    index: int
    source_path: str
    target_path: str
    size: int | None = None


class HfSourceSlpSeeder:
    def __init__(
        self,
        target: HfLocation,
        source_repo: str,
        desired_count: int,
        source_prefix: str = "",
        batch_size: int = 100,
        concurrency: int = 16,
        work_dir: str = "/workspace/slp-seed",
        download_timeout_seconds: int = 60,
        download_retries: int = 3,
    ):
        """Mirror source HF SLP files into the target HF raw folder in bounded batches."""
        self.target = target
        self.source_repo = source_repo
        self.source_prefix = source_prefix.strip("/")
        self.desired_count = desired_count
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)
        self.work_dir = Path(work_dir)
        self.download_timeout_seconds = download_timeout_seconds
        self.download_retries = download_retries
        self.source_api = HfApi(token=target.hf.token)

    def seed(self) -> dict:
        """Download source SLP files to temporary batches and upload each batch to HF."""
        started = time.monotonic()
        self.target.hf.create_repo(self.target.repo)
        existing = {
            path
            for path in self.target.hf.list_files(self.target.repo, self.target.raw_slp_path())
            if path.endswith(".slp")
        }
        if len(existing) >= self.desired_count:
            log_event("seed_done", existing=len(existing), seeded=0)
            return self._summary(started, existing=len(existing), seeded=0)

        jobs = self._plan_jobs(existing)
        log_event(
            "seed_planned",
            sourceRepo=self.source_repo,
            targetRepo=self.target.repo,
            targetPrefix=self.target.raw_slp_path(),
            existing=len(existing),
            planned=len(jobs),
        )

        seeded = 0
        for offset in range(0, len(jobs), self.batch_size):
            batch = jobs[offset : offset + self.batch_size]
            self._seed_batch(batch, offset // self.batch_size)
            seeded += len(batch)
            log_event("seeded_batch", batch=offset // self.batch_size, count=len(batch), seeded=seeded)

        summary = self._summary(started, existing=len(existing), seeded=seeded)
        log_event("seed_done", **summary)
        return summary

    def _plan_jobs(self, existing: set[str]) -> list[SourceSlp]:
        jobs: list[SourceSlp] = []
        filled = len(existing)
        for index, source_path, size in self._source_slps():
            target_path = self._target_path(index)
            if target_path in existing:
                continue
            jobs.append(SourceSlp(index=index, source_path=source_path, target_path=target_path, size=size))
            if filled + len(jobs) >= self.desired_count:
                return jobs
        raise RuntimeError(
            f"Source repo {self.source_repo} only had {filled + len(jobs)} usable SLP files"
        )

    def _seed_batch(self, batch: list[SourceSlp], batch_index: int):
        batch_dir = Path(tempfile.mkdtemp(prefix=f"slp-seed-{batch_index:06d}-", dir=self._work_parent()))
        upload_dir = batch_dir / "upload"
        raw_dir = upload_dir / self.target.raw_slp_dir
        manifest_path = upload_dir / "raw_slp_manifest" / f"batch-{batch_index:06d}.jsonl"
        raw_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.concurrency, len(batch))) as pool:
                rows = list(pool.map(lambda job: self._download_one(job, raw_dir), batch))
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.target.hf.upload_folder(self.target.repo, str(upload_dir), self.target.root)
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def _download_one(self, job: SourceSlp, upload_dir: Path) -> dict:
        local_path = upload_dir / Path(job.target_path).name
        url = hf_hub_url(self.source_repo, job.source_path, repo_type="dataset")
        self._stream_download(url, local_path)
        return {
            "index": job.index,
            "sourceRepo": self.source_repo,
            "sourcePath": job.source_path,
            "targetRepo": self.target.repo,
            "targetPath": job.target_path,
            "size": job.size,
        }

    def _stream_download(self, url: str, local_path: Path):
        headers = {"user-agent": "smash-frame-data-seeder"}
        if self.target.hf.token:
            headers["authorization"] = f"Bearer {self.target.hf.token}"
        for attempt in range(1, self.download_retries + 1):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=self.download_timeout_seconds) as response:
                    with local_path.open("wb") as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                return
                            handle.write(chunk)
            except Exception as error:
                local_path.unlink(missing_ok=True)
                if attempt == self.download_retries:
                    raise
                delay = self._download_retry_delay(attempt, error)
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    log_event("seed_download_rate_limited", attempt=attempt, waitSeconds=delay)
                time.sleep(delay)

    def _download_retry_delay(self, attempt: int, error: Exception) -> float:
        if isinstance(error, urllib.error.HTTPError) and error.code == 429:
            retry_after = error.headers.get("retry-after")
            if retry_after:
                try:
                    return min(600, float(retry_after))
                except ValueError:
                    pass
            return min(600, 60 * attempt)
        return min(30, 2**attempt)

    def _source_slps(self):
        index = 0
        for item in self.source_api.list_repo_tree(
            self.source_repo,
            repo_type="dataset",
            path_in_repo=self.source_prefix,
            recursive=True,
        ):
            path = getattr(item, "path", "")
            if path.endswith(".slp"):
                yield index, path, getattr(item, "size", None)
                index += 1

    def _target_path(self, index: int) -> str:
        return self.target._join(self.target.raw_slp_path(), f"{index:06d}.slp")

    def _work_parent(self) -> str:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return str(self.work_dir)

    def _summary(self, started: float, existing: int, seeded: int) -> dict:
        return {
            "sourceRepo": self.source_repo,
            "targetRepo": self.target.repo,
            "targetPrefix": self.target.raw_slp_path(),
            "desired": self.desired_count,
            "existing": existing,
            "seeded": seeded,
            "seconds": round(time.monotonic() - started, 3),
        }


def build_seeder() -> HfSourceSlpSeeder:
    """Build the HF source seeder from environment."""
    hf = HfConnector(expected_email=os.environ.get("SMASH_HF_EXPECTED_EMAIL"))
    target = HfLocation(
        repo=os.environ["SMASH_HF_REPO"],
        hf=hf,
        root=os.environ.get("SMASH_HF_ROOT", ""),
    )
    return HfSourceSlpSeeder(
        target=target,
        source_repo=os.environ["SMASH_SOURCE_HF_REPO"],
        source_prefix=os.environ.get("SMASH_SOURCE_HF_PREFIX", ""),
        desired_count=int(os.environ.get("SMASH_SOURCE_SAMPLE_LIMIT", os.environ.get("SMASH_SAMPLE_LIMIT", "1"))),
        batch_size=int(os.environ.get("SMASH_SOURCE_SEED_BATCH_SIZE", "100")),
        concurrency=int(os.environ.get("SMASH_SOURCE_SEED_CONCURRENCY", "16")),
        work_dir=os.environ.get("SMASH_SOURCE_SEED_WORK_DIR", "/workspace/slp-seed"),
        download_timeout_seconds=int(os.environ.get("SMASH_SOURCE_DOWNLOAD_TIMEOUT_SECONDS", "60")),
        download_retries=int(os.environ.get("SMASH_SOURCE_DOWNLOAD_RETRIES", "3")),
    )


def main():
    """CLI entrypoint for remote raw-SLP seeding."""
    print(json.dumps(build_seeder().seed(), indent=2))


if __name__ == "__main__":
    main()
