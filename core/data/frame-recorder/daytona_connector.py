from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import http.client
import io
import json
import math
import os
import re
import secrets
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


VIDEO_WIDTH = 252
VIDEO_HEIGHT = 208
VIDEO_FPS = 20
SOURCE_FPS = 60
UPLOAD_RETRY_SECONDS = 5.0
NO_PLAYABLE_FRAMES = "no_playable_frames"
SKIPPED_NO_PLAYABLE_FRAMES = f"skipped:{NO_PLAYABLE_FRAMES}"
DEFAULT_RENDERER_SNAPSHOT = "smash-cpu-renderer-e7711b1-v3"
DEFAULT_ASSET_VOLUME = "smash-frame-assets-v1"
DEFAULT_SOURCE_ROOT = "hal-fox-captain-falcon-battlefield"
DEFAULT_TARGET_ROOT = (
    "hal-fox-captain-falcon-battlefield/recordings-252x208-20fps-slippi-pts-v3"
)
VIDEO_SUFFIXES = (".avi", ".mkv", ".mp4", ".mov", ".nut")
DEFAULT_DRIVE_CHUNK_SIZE = "512M"
DEFAULT_SPOOL_MAX_BYTES = 9 * 1024**3

_INVALID_PLAYABLE_MAPPING = re.compile(
    r"(?:RuntimeError: )?invalid playable/render frame mapping: "
    r"raw=-?\d+\.\.-?\d+, playable=(?P<first>-?\d+)\.\.(?P<last>-?\d+)"
)


def _is_no_playable_frames_error(error: str) -> bool:
    match = _INVALID_PLAYABLE_MAPPING.fullmatch(error.strip())
    return bool(match and int(match.group("last")) < int(match.group("first")))


def _no_playable_frames_result_tar() -> bytes:
    metadata = (
        json.dumps({"skipReason": NO_PLAYABLE_FRAMES}, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w") as output:
        member = tarfile.TarInfo("metadata.json")
        member.mode = 0o644
        member.mtime = 0
        member.size = len(metadata)
        output.addfile(member, io.BytesIO(metadata))
    return bundle.getvalue()


class PostCommitCleanupError(RuntimeError):
    """The Drive manifest is durable, but local committed files could not be removed."""


def _json_line(event: str, **fields) -> None:
    print(
        json.dumps({"event": event, "time": round(time.time(), 3), **fields}, sort_keys=True),
        flush=True,
    )


def _run(
    command: list[str],
    *,
    check: bool = True,
    input_text: str | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
    )
    if check and completed.returncode:
        detail = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        raise RuntimeError(detail or f"command failed ({completed.returncode}): {command[0]}")
    return completed


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "-" for character in value.lower())
    return "-".join(part for part in cleaned.split("-") if part)[:48]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remote(remote: str, path: str = "") -> str:
    return f"{remote.rstrip(':')}:{path.strip('/')}"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Daytona volumes are S3/FUSE-backed and do not implement rename. Queue JSON is tiny and
    # every reader treats a partial/malformed observation as transient, so publish it directly
    # after its referenced immutable payload has been closed. Local files retain atomic replace.
    if str(path).startswith("/mnt/smash-assets/"):
        with path.open("w") as output:
            json.dump(value, output, separators=(",", ":"), sort_keys=True)
            output.write("\n")
            output.flush()
            with contextlib.suppress(OSError):
                os.fsync(output.fileno())
        return
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.partial")
    try:
        with temporary.open("w") as output:
            json.dump(value, output, separators=(",", ":"), sort_keys=True)
            output.write("\n")
            output.flush()
            with contextlib.suppress(OSError):
                os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class DaytonaSandbox:
    id: str
    name: str
    cpu: int
    memory: int
    disk: int
    gpu: int
    state: str

    @classmethod
    def from_json(cls, row: dict) -> "DaytonaSandbox":
        return cls(
            id=row["id"],
            name=row["name"],
            cpu=int(row["cpu"]),
            memory=int(row["memory"]),
            disk=int(row["disk"]),
            gpu=int(row.get("gpu", 0)),
            state=row["state"],
        )


@dataclass(frozen=True)
class FleetPlan:
    replay_count: int
    average_replay_seconds: float
    realtime_factor: float
    deadline_hours: float
    processes_per_sandbox: int
    recommended_processes: int
    recommended_sandboxes: int
    requested_sandboxes: int
    estimated_hours: float


class DaytonaConnector:
    """Launch one Daytona coordinator and a CPU-only renderer fleet.

    The local process carries control traffic only. SLP archives flow once from Drive to the
    coordinator, workers receive individual leases, and one coordinator uploader writes
    committed result batches back to Drive.
    """

    def __init__(self) -> None:
        self.daytona = os.environ.get("SMASH_DAYTONA_BIN", shutil.which("daytona") or "daytona")
        self.rclone_config = Path(
            os.environ.get("SMASH_GDRIVE_CONFIG", "~/.config/rclone/rclone.conf")
        ).expanduser()
        self.drive_remote = os.environ.get("SMASH_GDRIVE_REMOTE", "smash-drive")
        self.source_root = os.environ.get("SMASH_GDRIVE_ROOT", DEFAULT_SOURCE_ROOT).strip("/")
        self.target_root = os.environ.get("SMASH_GDRIVE_RECORDING_DIR", DEFAULT_TARGET_ROOT).strip(
            "/"
        )
        self.renderer_snapshot = os.environ.get(
            "SMASH_DAYTONA_RENDERER_SNAPSHOT", DEFAULT_RENDERER_SNAPSHOT
        )
        self.coordinator_snapshot = os.environ.get(
            "SMASH_DAYTONA_COORDINATOR_SNAPSHOT", "daytona-medium"
        )
        self.asset_volume = os.environ.get("SMASH_DAYTONA_ASSET_VOLUME", DEFAULT_ASSET_VOLUME)
        self.queue_mount = "/mnt/smash-assets"
        self.local_iso = Path(
            os.environ.get(
                "SMASH_MELEE_ISO",
                "/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso",
            )
        ).expanduser()
        self.region = os.environ.get("SMASH_DAYTONA_REGION", "us")
        self.worker_processes = max(
            1, int(os.environ.get("SMASH_PROCESSES_PER_SANDBOX", "4"))
        )
        self.result_batch_size = max(1, int(os.environ.get("SMASH_WORKER_RESULT_BATCH", "10")))
        self.upload_batch_size = max(1, int(os.environ.get("SMASH_UPLOAD_BATCH_SIZE", "100")))
        self.upload_concurrency = max(
            1, int(os.environ.get("SMASH_UPLOAD_CONCURRENCY", "1"))
        )
        self.upload_tps_limit = max(
            1, int(os.environ.get("SMASH_UPLOAD_TPS_LIMIT", "1"))
        )
        self.upload_min_batch = max(
            1,
            min(
                self.upload_batch_size,
                int(os.environ.get("SMASH_UPLOAD_MIN_BATCH", "64")),
            ),
        )
        self.upload_max_attempts = max(
            1, int(os.environ.get("SMASH_UPLOAD_MAX_ATTEMPTS", "3"))
        )
        self.upload_max_bytes = max(
            64 * 1024 * 1024,
            int(os.environ.get("SMASH_UPLOAD_BATCH_MAX_BYTES", str(2 * 1024**3))),
        )
        self.prefetch = max(1, int(os.environ.get("SMASH_COORDINATOR_PREFETCH", "512")))
        self.spool_max_bytes = max(
            1024 * 1024 * 1024,
            int(
                os.environ.get(
                    "SMASH_COORDINATOR_SPOOL_MAX_BYTES",
                    str(DEFAULT_SPOOL_MAX_BYTES),
                )
            ),
        )
        self.lease_seconds = max(120, int(os.environ.get("SMASH_JOB_LEASE_SECONDS", "300")))
        self.max_attempts = max(1, int(os.environ.get("SMASH_JOB_MAX_ATTEMPTS", "3")))
        self.deadline_hours = float(os.environ.get("SMASH_DEADLINE_HOURS", "5"))
        self.average_replay_seconds = float(
            os.environ.get("SMASH_AVERAGE_REPLAY_SECONDS", "148.32")
        )
        self.realtime_factor = float(os.environ.get("SMASH_RENDER_REALTIME_FACTOR", "0.9511"))
        self.poll_seconds = max(2, int(os.environ.get("SMASH_COORDINATOR_POLL_SECONDS", "15")))
        self.health_grace_seconds = max(
            120, int(os.environ.get("SMASH_COORDINATOR_HEALTH_GRACE_SECONDS", "900"))
        )
        self.launch_parallelism = max(
            1, int(os.environ.get("SMASH_DAYTONA_LAUNCH_PARALLELISM", "12"))
        )
        self.safety_minutes = max(
            360, int(os.environ.get("SMASH_DAYTONA_SAFETY_MINUTES", "720"))
        )
        self.keep_resources = os.environ.get("SMASH_KEEP_DAYTONA_RESOURCES", "") == "1"
        self.run_id = _safe_name(
            os.environ.get("SMASH_RUN_ID", time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
        )
        if not self.run_id:
            raise ValueError("SMASH_RUN_ID must contain at least one letter or digit")
        self.queue_root = f"{self.queue_mount}/runs/{self.run_id}"
        self._owned_sandboxes: list[str] = []
        self._owned_lock = threading.Lock()
        self._name_lock = threading.Lock()
        self._reserved_names: set[str] = set()
        self._existing_names: set[str] | None = None
        self._snapshot_rows: dict[str, dict] | None = None
        self._worker_recovery_generation: dict[int, int] = {}

        if not self.rclone_config.is_file():
            raise FileNotFoundError(self.rclone_config)
        if not self.local_iso.is_file():
            raise FileNotFoundError(self.local_iso)

    def plan(
        self,
        replay_count: int,
        requested_sandboxes: int = 0,
    ) -> FleetPlan:
        recommended_processes = (
            math.ceil(
                replay_count
                * self.average_replay_seconds
                / (self.realtime_factor * self.deadline_hours * 3600)
            )
            if replay_count
            else 0
        )
        recommended_sandboxes = math.ceil(recommended_processes / self.worker_processes)
        sandboxes = requested_sandboxes or recommended_sandboxes
        estimated_hours = 0.0
        if replay_count:
            if not sandboxes:
                raise ValueError("at least one worker sandbox is required for remaining work")
            estimated_hours = (
                replay_count
                * self.average_replay_seconds
                / (self.realtime_factor * sandboxes * self.worker_processes * 3600)
            )
        return FleetPlan(
            replay_count=replay_count,
            average_replay_seconds=self.average_replay_seconds,
            realtime_factor=self.realtime_factor,
            deadline_hours=self.deadline_hours,
            processes_per_sandbox=self.worker_processes,
            recommended_processes=recommended_processes,
            recommended_sandboxes=recommended_sandboxes,
            requested_sandboxes=sandboxes,
            estimated_hours=round(estimated_hours, 3),
        )

    def run(self, sample_limit: int = 0, worker_count: int = 0) -> dict:
        token = secrets.token_urlsafe(32)
        coordinator_name = f"smash-coord-{self.run_id}"
        coordinator: DaytonaSandbox | None = None
        workers: list[DaytonaSandbox] = []
        started = time.monotonic()
        try:
            worker_profile = self._cpu_only_snapshot(self.renderer_snapshot)
            self._cpu_only_snapshot(self.coordinator_snapshot)
            self._cpu_only_snapshot(
                os.environ.get(
                    "SMASH_DAYTONA_ASSET_SNAPSHOT", "smash-cpu-renderer-e7711b1-1cpu-v1"
                )
            )
            if self.worker_processes > int(worker_profile["cpu"]):
                raise ValueError(
                    f"{self.worker_processes} render processes exceed the "
                    f"{worker_profile['cpu']}-vCPU snapshot {self.renderer_snapshot}"
                )
            self._ensure_asset_volume()
            coordinator = self._create_sandbox(
                coordinator_name,
                self.coordinator_snapshot,
                labels={"smash-run-id": self.run_id, "smash-role": "coordinator"},
                env={"SMASH_COORDINATOR_TOKEN": token},
                volume=f"{self.asset_volume}:{self.queue_mount}",
            )
            self._prepare_coordinator(coordinator)
            self._start_coordinator(coordinator, sample_limit)
            base_url = self._preview_url(coordinator, port=8765, expires=43200)
            initial = self._wait_for_health(base_url, token, timeout_seconds=300)
            replay_count = sum(
                int(initial["counts"].get(status, 0))
                for status in ("pending", "queued", "leased", "complete", "uploading")
            )
            plan = self.plan(replay_count, worker_count)
            _json_line("fleet_plan", **asdict(plan))
            workers = self._launch_workers(plan.requested_sandboxes, base_url, token)

            last_counts: dict | None = None
            last_health_ok = time.monotonic()
            last_progress = time.monotonic()
            next_worker_check = time.monotonic() + 60
            while True:
                try:
                    health = self._http_json(base_url, token, "/health", timeout=60)
                    last_health_ok = time.monotonic()
                except Exception as error:
                    elapsed = time.monotonic() - last_health_ok
                    _json_line(
                        "coordinator_health_retry",
                        error=str(error),
                        unavailableSeconds=round(elapsed, 3),
                    )
                    if elapsed >= self.health_grace_seconds:
                        raise RuntimeError(
                            f"coordinator unavailable for {round(elapsed)} seconds: {error}"
                        ) from error
                    time.sleep(min(self.poll_seconds, 30))
                    continue
                counts = health["counts"]
                if counts != last_counts:
                    _json_line("pipeline_progress", state=health["state"], **counts)
                    last_counts = counts
                    last_progress = time.monotonic()
                if health["state"] in {"complete", "failed"}:
                    result = {
                        "runId": self.run_id,
                        "state": health["state"],
                        "seconds": round(time.monotonic() - started, 3),
                        "plan": asdict(plan),
                        "counts": counts,
                        "target": _remote(self.drive_remote, self.target_root),
                        "coordinator": asdict(coordinator),
                        "workers": [asdict(worker) for worker in workers],
                        "errors": health.get("errors", []),
                    }
                    if health["state"] != "complete":
                        raise RuntimeError(json.dumps(result, sort_keys=True))
                    return result
                if (
                    workers
                    and sum(counts.get(status, 0) for status in ("pending", "queued", "leased"))
                    > 0
                    and time.monotonic() >= next_worker_check
                ):
                    self._recover_workers(workers)
                    next_worker_check = time.monotonic() + 60
                    _json_line(
                        "worker_liveness_checked",
                        stalledSeconds=round(time.monotonic() - last_progress, 3),
                    )
                time.sleep(self.poll_seconds)
        finally:
            if not self.keep_resources:
                workers_quiesced = False
                try:
                    self._delete_run_workers()
                    workers_quiesced = True
                except Exception as error:
                    _json_line("worker_quiesce_failed", error=str(error))
                coordinator_stopped = False
                if coordinator is not None and workers_quiesced:
                    try:
                        self._stop_coordinator_process(coordinator)
                        coordinator_stopped = True
                    except Exception as error:
                        _json_line("coordinator_stop_failed", error=str(error))
                if coordinator is not None and workers_quiesced and coordinator_stopped:
                    with contextlib.suppress(Exception):
                        self._remove_queue_root(coordinator)
                self._cleanup()

    def _daytona(self, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        return _run([self.daytona, *args], timeout=timeout)

    def _exec(
        self,
        sandbox: DaytonaSandbox,
        command: list[str],
        *,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        # Daytona v0.196 flattens exec argv. Quote the complete remote command once so paths,
        # Python snippets, and shell operators survive that boundary exactly.
        remote = shlex.quote(shlex.join(command))
        return self._daytona(
            "exec",
            sandbox.name,
            "--timeout",
            str(timeout),
            "--",
            "bash",
            "-lc",
            remote,
            timeout=timeout + 30,
        )

    def _list_sandboxes(self) -> list[dict]:
        rows = json.loads(
            self._daytona("list", "--format", "json", "--limit", "100").stdout
        )["items"]
        return [
            row
            for row in rows
            if not str(row.get("name", "")).startswith("DESTROYED_")
            and str(row.get("state", "")).lower() not in {"deleted", "destroyed"}
        ]

    def _cpu_only_snapshot(self, name: str) -> dict:
        if self._snapshot_rows is None:
            rows = json.loads(self._daytona("snapshot", "list", "--format", "json").stdout)
            self._snapshot_rows = {str(row["name"]): row for row in rows}
        row = self._snapshot_rows.get(name)
        if row is None:
            raise RuntimeError(f"Daytona snapshot not found: {name}")
        if row.get("state") != "active":
            raise RuntimeError(f"Daytona snapshot is not active: {name}")
        if int(row.get("gpu", 0)) != 0:
            raise RuntimeError(f"GPU snapshots are forbidden for frame capture: {name}")
        return row

    def _create_sandbox(
        self,
        name: str,
        snapshot: str,
        *,
        labels: dict[str, str],
        env: dict[str, str] | None = None,
        volume: str | None = None,
    ) -> DaytonaSandbox:
        with self._name_lock:
            if self._existing_names is None:
                self._existing_names = {
                    str(row["name"]) for row in self._list_sandboxes()
                }
            if name in self._reserved_names or name in self._existing_names:
                raise RuntimeError(f"refusing to reuse existing Daytona sandbox: {name}")
            self._reserved_names.add(name)
        command = [
            "create",
            "--name",
            name,
            "--snapshot",
            snapshot,
            "--target",
            self.region,
            "--auto-stop",
            str(self.safety_minutes),
            "--auto-delete",
            str(self.safety_minutes * 2),
        ]
        for key, value in sorted(labels.items()):
            command += ["--label", f"{key}={value}"]
        for key, value in sorted((env or {}).items()):
            command += ["--env", f"{key}={value}"]
        if volume:
            command += ["--volume", volume]
        try:
            self._daytona(*command, timeout=600)
        except BaseException:
            with self._name_lock:
                self._reserved_names.discard(name)
            raise
        with self._owned_lock:
            self._owned_sandboxes.append(name)
        with self._name_lock:
            assert self._existing_names is not None
            self._existing_names.add(name)
        row = json.loads(self._daytona("info", name, "--format", "json").stdout)
        sandbox = DaytonaSandbox.from_json(row)
        if sandbox.gpu:
            raise RuntimeError(f"created forbidden GPU sandbox: {sandbox.name}")
        _json_line("sandbox_created", **asdict(sandbox))
        return sandbox

    def _create_worker(self, index: int, generation: int = 0) -> DaytonaSandbox:
        suffix = f"-{index:03d}" if generation == 0 else f"-{index:03d}-r{generation}"
        return self._create_sandbox(
            f"smash-worker-{self.run_id}{suffix}",
            self.renderer_snapshot,
            labels={
                "smash-run-id": self.run_id,
                "smash-role": "worker",
                "smash-worker-index": str(index),
            },
            env={
                "SMASH_QUEUE_ROOT": self.queue_root,
                "SMASH_RENDERER_SNAPSHOT": self.renderer_snapshot,
            },
            volume=f"{self.asset_volume}:/mnt/smash-assets",
        )

    def _launch_workers(
        self, count: int, base_url: str, token: str
    ) -> list[DaytonaSandbox]:
        if not count:
            return []

        def launch(index: int) -> DaytonaSandbox:
            worker = self._create_worker(index)
            self._start_worker(worker, index)
            return worker

        workers: list[DaytonaSandbox] = []
        with ThreadPoolExecutor(max_workers=min(self.launch_parallelism, count)) as executor:
            futures = {executor.submit(launch, index): index for index in range(count)}
            for future in as_completed(futures):
                workers.append(future.result())
        workers.sort(key=lambda worker: worker.name)
        return workers

    def _ensure_asset_volume(self) -> None:
        rows = json.loads(self._daytona("volume", "list", "--format", "json").stdout)
        matching = [row for row in rows if row["name"] == self.asset_volume]
        if not matching:
            self._daytona("volume", "create", self.asset_volume, "--size", "10", timeout=300)
            _json_line("asset_volume_created", name=self.asset_volume, sizeGiB=10)
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            row = json.loads(
                self._daytona("volume", "get", self.asset_volume, "--format", "json").stdout
            )
            if row["state"] == "ready":
                break
            if row["state"] == "error":
                raise RuntimeError(f"Daytona asset volume failed: {row.get('errorReason')}")
            time.sleep(2)
        else:
            raise TimeoutError(f"Daytona volume did not become ready: {self.asset_volume}")

        seed = self._create_sandbox(
            f"smash-assets-{self.run_id}",
            os.environ.get(
                "SMASH_DAYTONA_ASSET_SNAPSHOT", "smash-cpu-renderer-e7711b1-1cpu-v1"
            ),
            labels={"smash-run-id": self.run_id, "smash-role": "asset-seed"},
            volume=f"{self.asset_volume}:/mnt/smash-assets",
        )
        try:
            local_digest = _sha256(self.local_iso)
            try:
                digest = self._exec(
                    seed,
                    ["sha256sum", "/mnt/smash-assets/melee.iso"],
                    timeout=300,
                ).stdout.split()[0]
            except RuntimeError:
                digest = ""
            if digest != local_digest:
                with tempfile.TemporaryDirectory() as temporary:
                    upload_name = Path(temporary) / "melee.iso"
                    upload_name.symlink_to(self.local_iso)
                    self._upload_file(seed, upload_name, "/tmp/")
                self._exec(
                    seed,
                    [
                        "bash",
                        "-lc",
                        "cp /tmp/melee.iso /mnt/smash-assets/melee.iso",
                    ],
                    timeout=900,
                )
            digest = self._exec(
                seed,
                ["sha256sum", "/mnt/smash-assets/melee.iso"],
                timeout=300,
            ).stdout.split()[0]
            if digest != local_digest:
                raise RuntimeError(f"Daytona asset ISO hash mismatch: {digest} != {local_digest}")
            _json_line("asset_iso_ready", bytes=self.local_iso.stat().st_size, sha256=digest)
        finally:
            _run([self.daytona, "delete", seed.name], check=False, timeout=300)
            with contextlib.suppress(ValueError):
                with self._owned_lock:
                    self._owned_sandboxes.remove(seed.name)
            with self._name_lock:
                self._reserved_names.discard(seed.name)

    def _deploy_self(self, sandbox: DaytonaSandbox) -> None:
        import zlib

        payload = base64.b64encode(zlib.compress(Path(__file__).read_bytes(), 9)).decode("ascii")
        wrapper_path = (
            Path(__file__).parents[3]
            / "experiments_dump"
            / "fast_replay_probe"
            / "render-ffv1-replay.sh"
        )
        wrapper = base64.b64encode(zlib.compress(wrapper_path.read_bytes(), 9)).decode("ascii")
        script = (
            "import base64,os,pathlib,zlib; p=pathlib.Path('/tmp/daytona_connector.py'); "
            f"p.write_bytes(zlib.decompress(base64.b64decode({payload!r}))); "
            "w=pathlib.Path('/tmp/render-ffv1-replay.sh'); "
            f"w.write_bytes(zlib.decompress(base64.b64decode({wrapper!r}))); "
            "os.chmod(w,0o755)"
        )
        self._exec(sandbox, ["python3", "-c", script], timeout=120)

    def _prepare_coordinator(self, sandbox: DaytonaSandbox) -> None:
        self._deploy_self(sandbox)
        self._exec(
            sandbox,
            [
                "bash",
                "-lc",
                "sudo apt-get update >/dev/null && sudo apt-get install -y rclone zstd >/dev/null && "
                "mkdir -p /home/daytona/.config/rclone /home/daytona/smash-coordinator",
            ],
            timeout=300,
        )
        config_payload = base64.b64encode(self.rclone_config.read_bytes()).decode("ascii")
        config_script = (
            "import base64,pathlib; p=pathlib.Path('/tmp/rclone.conf'); "
            f"p.write_bytes(base64.b64decode({config_payload!r}))"
        )
        self._exec(sandbox, ["python3", "-c", config_script], timeout=60)
        self._exec(
            sandbox,
            [
                "install",
                "-m",
                "600",
                "/tmp/rclone.conf",
                "/home/daytona/.config/rclone/rclone.conf",
            ],
            timeout=60,
        )

    def _start_coordinator(self, sandbox: DaytonaSandbox, sample_limit: int) -> None:
        command = [
            "nohup",
            "python3",
            "/tmp/daytona_connector.py",
            "coordinator",
            "--remote",
            self.drive_remote,
            "--source-root",
            self.source_root,
            "--target-root",
            self.target_root,
            "--config",
            "/home/daytona/.config/rclone/rclone.conf",
            "--spool",
            "/home/daytona/smash-coordinator",
            "--sample-limit",
            str(sample_limit),
            "--prefetch",
            str(self.prefetch),
            "--lease-seconds",
            str(self.lease_seconds),
            "--max-attempts",
            str(self.max_attempts),
            "--upload-batch-size",
            str(self.upload_batch_size),
            "--upload-concurrency",
            str(self.upload_concurrency),
            "--upload-tps-limit",
            str(self.upload_tps_limit),
            "--upload-min-batch",
            str(self.upload_min_batch),
            "--upload-max-attempts",
            str(self.upload_max_attempts),
            "--upload-max-bytes",
            str(self.upload_max_bytes),
            "--spool-max-bytes",
            str(self.spool_max_bytes),
            "--queue-root",
            self.queue_root,
        ]
        shell = shlex.join(command)
        self._exec(
            sandbox,
            [
                "bash",
                "-lc",
                f"{shell} > /home/daytona/smash-coordinator/coordinator.log 2>&1 & "
                "echo $! > /home/daytona/smash-coordinator/coordinator.pid",
            ],
            timeout=30,
        )

    def _stop_coordinator_process(self, sandbox: DaytonaSandbox) -> None:
        script = """import contextlib,os,pathlib,signal,time
path=pathlib.Path('/home/daytona/smash-coordinator/coordinator.pid')
if path.is_file():
 pid=int(path.read_text().strip())
 with contextlib.suppress(ProcessLookupError): os.kill(pid,signal.SIGTERM)
 deadline=time.time()+15
 while time.time()<deadline:
  try: os.kill(pid,0)
  except ProcessLookupError: break
  time.sleep(.1)
 else:
  with contextlib.suppress(ProcessLookupError): os.kill(pid,signal.SIGKILL)
 path.unlink(missing_ok=True)
"""
        self._exec(sandbox, ["python3", "-c", script], timeout=30)

    def _remove_queue_root(self, sandbox: DaytonaSandbox) -> None:
        expected_prefix = f"{self.queue_mount}/runs/"
        if not self.queue_root.startswith(expected_prefix) or self.queue_root == expected_prefix:
            raise RuntimeError(f"refusing unsafe queue cleanup: {self.queue_root}")
        for attempt in range(1, 4):
            self._exec(sandbox, ["rm", "-rf", "--", self.queue_root], timeout=300)
            probe = self._exec(
                sandbox,
                [
                    "python3",
                    "-c",
                    "import pathlib; p=pathlib.Path(%r); "
                    "print(sum(1 for _ in p.rglob('*')) if p.exists() else 0)"
                    % self.queue_root,
                ],
                timeout=60,
            )
            if probe.stdout.strip() == "0":
                _json_line("queue_root_deleted", path=self.queue_root, attempt=attempt)
                return
            time.sleep(2**attempt)
        raise RuntimeError(f"queue cleanup did not converge: {self.queue_root}")

    def _start_worker(self, sandbox: DaytonaSandbox, index: int) -> None:
        self._deploy_self(sandbox)
        worker_command = (
            "python3 /tmp/daytona_connector.py worker "
            f"--worker-id worker-{index:03d} --processes {self.worker_processes} "
            f"--result-batch-size {self.result_batch_size}"
        )
        supervisor = (
            f"while true; do {worker_command}; status=$?; "
            "if [ $status -eq 0 ]; then exit 0; fi; sleep 5; done"
        )
        shell = (
            f"nohup bash -lc {shlex.quote(supervisor)} "
            "> /tmp/smash-worker.log 2>&1 &"
        )
        self._exec(sandbox, ["bash", "-lc", shell], timeout=30)
        self._exec(
            sandbox,
            ["bash", "-lc", "sleep 1; pgrep -f 'daytona_connector.py worker' >/dev/null"],
            timeout=30,
        )

    def _recover_workers(self, workers: list[DaytonaSandbox]) -> None:
        rows = {str(row["name"]): row for row in self._list_sandboxes()}

        def recover(item: tuple[int, DaytonaSandbox]) -> None:
            index, worker = item
            try:
                row = rows.get(worker.name)
                if row is None:
                    raise RuntimeError("sandbox is absent")
                if row.get("state") != "started":
                    self._daytona("start", worker.name, timeout=300)
                try:
                    process = self._exec(
                        worker,
                        ["pgrep", "-f", "daytona_connector.py worker"],
                        timeout=30,
                    )
                except RuntimeError:
                    process = None
                if process is None or not process.stdout.strip():
                    self._start_worker(worker, index)
                return
            except Exception as error:
                _json_line(
                    "worker_recovery_retry",
                    name=worker.name,
                    index=index,
                    error=str(error),
                )
            # A missing/broken sandbox must not abort the healthy fleet. Delete it if it still
            # exists, then replace just that capacity; lost leases requeue through heartbeats.
            _run([self.daytona, "delete", worker.name], check=False, timeout=300)
            replacement: DaytonaSandbox | None = None
            try:
                generation = self._worker_recovery_generation.get(index, 0) + 1
                self._worker_recovery_generation[index] = generation
                replacement = self._create_worker(index, generation)
                self._start_worker(replacement, index)
                workers[index] = replacement
                _json_line(
                    "worker_replaced",
                    oldName=worker.name,
                    newName=replacement.name,
                    index=index,
                )
            except Exception as error:
                if replacement is not None:
                    _run([self.daytona, "delete", replacement.name], check=False, timeout=300)
                _json_line(
                    "worker_replacement_retry",
                    oldName=worker.name,
                    index=index,
                    error=str(error),
                )

        with ThreadPoolExecutor(
            max_workers=min(self.launch_parallelism, len(workers))
        ) as executor:
            list(executor.map(recover, enumerate(workers)))

    def _upload_file(self, sandbox: DaytonaSandbox, local_path: Path, destination: str) -> None:
        shim = Path(__file__).parents[3] / "experiments_dump/fast_replay_probe/daytona_ssh_upload"
        env = os.environ.copy()
        env["PATH"] = str(shim) + os.pathsep + env.get("PATH", "")
        env["DAYTONA_UPLOAD_EXTRA"] = str(local_path)
        env["DAYTONA_UPLOAD_DEST"] = destination
        _run(
            [self.daytona, "ssh", sandbox.name, "--expires", "10"],
            timeout=900,
            env=env,
        )

    def _preview_url(self, sandbox: DaytonaSandbox, port: int, expires: int) -> str:
        output = self._daytona(
            "preview-url",
            sandbox.name,
            "--port",
            str(port),
            "--expires",
            str(expires),
        ).stdout
        for word in output.split():
            if word.startswith("https://"):
                return word.rstrip()
        raise RuntimeError(f"Daytona did not return a preview URL: {output.strip()}")

    def _wait_for_health(self, base_url: str, token: str, timeout_seconds: int) -> dict:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self._http_json(base_url, token, "/health", timeout=30)
            except Exception as error:
                last_error = error
                time.sleep(2)
        raise TimeoutError(f"coordinator health check timed out: {last_error}")

    def _http_json(
        self, base_url: str, token: str, path: str, *, timeout: float
    ) -> dict:
        parsed = urllib.parse.urlsplit(base_url)
        target = urllib.parse.urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path.rstrip("/") + path,
                parsed.query,
                parsed.fragment,
            )
        )
        request = urllib.request.Request(
            target,
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def _cleanup(self) -> None:
        with self._owned_lock:
            pending = set(self._owned_sandboxes)
        with contextlib.suppress(Exception):
            pending.update(
                str(row["name"])
                for row in self._list_sandboxes()
                if (row.get("labels") or {}).get("smash-run-id") == self.run_id
            )
        for attempt in range(1, 4):
            if not pending:
                break
            for name in sorted(pending, reverse=True):
                completed = _run([self.daytona, "delete", name], check=False, timeout=300)
                _json_line(
                    "sandbox_delete_attempt",
                    name=name,
                    attempt=attempt,
                    ok=completed.returncode == 0
                    or "not found" in completed.stderr.lower(),
                )
            try:
                existing = {str(row["name"]) for row in self._list_sandboxes()}
                pending.intersection_update(existing)
            except Exception as error:
                _json_line("sandbox_delete_verify_retry", attempt=attempt, error=str(error))
            if pending:
                time.sleep(2**attempt)
        if pending:
            _json_line("sandbox_cleanup_incomplete", names=sorted(pending))
        with self._owned_lock:
            self._owned_sandboxes[:] = [
                name for name in self._owned_sandboxes if name in pending
            ]

    def _delete_run_workers(self) -> None:
        pending = {
            str(row["name"])
            for row in self._list_sandboxes()
            if (row.get("labels") or {}).get("smash-run-id") == self.run_id
            and (row.get("labels") or {}).get("smash-role") == "worker"
        }

        def delete(name: str) -> tuple[str, bool]:
            completed = _run([self.daytona, "delete", name], check=False, timeout=300)
            return name, completed.returncode == 0 or "not found" in completed.stderr.lower()

        for attempt in range(1, 4):
            if not pending:
                break
            with ThreadPoolExecutor(
                max_workers=min(self.launch_parallelism, len(pending) or 1)
            ) as pool:
                results = list(pool.map(delete, sorted(pending)))
            for name, ok in results:
                _json_line("worker_quiesced", name=name, attempt=attempt, ok=ok)
            existing = {str(row["name"]) for row in self._list_sandboxes()}
            pending.intersection_update(existing)
            if pending:
                time.sleep(2**attempt)
        if pending:
            raise RuntimeError(f"could not quiesce workers: {sorted(pending)}")
        run_worker_names = {
            str(row["name"])
            for row in self._list_sandboxes()
            if (row.get("labels") or {}).get("smash-run-id") == self.run_id
            and (row.get("labels") or {}).get("smash-role") == "worker"
        }
        if run_worker_names:
            raise RuntimeError(f"workers reappeared during quiesce: {sorted(run_worker_names)}")
        with self._owned_lock:
            self._owned_sandboxes[:] = [
                name
                for name in self._owned_sandboxes
                if not name.startswith(f"smash-worker-{self.run_id}-")
            ]


class DriveClient:
    """One deliberately rate-limited rclone client owned by the coordinator."""

    def __init__(self, remote: str, config: Path) -> None:
        self.remote = remote.rstrip(":")
        self.config = config
        self.retries = 6

    def command(
        self,
        *args: str,
        check: bool = True,
        stdout: BinaryIO | int | None = subprocess.PIPE,
        stdin: BinaryIO | int | None = None,
    ) -> subprocess.Popen:
        command = [
            "rclone",
            *args,
            "--config",
            str(self.config),
            "--retries",
            "10",
            "--low-level-retries",
            "20",
            "--tpslimit",
            "8",
            "--tpslimit-burst",
            "8",
        ]
        process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE)
        if check:
            _, stderr = process.communicate()
            if process.returncode:
                raise RuntimeError(stderr.decode(errors="replace").strip())
        return process

    def text(self, *args: str) -> str:
        completed = _run(
            [
                "rclone",
                *args,
                "--config",
                str(self.config),
                "--retries",
                "10",
                "--low-level-retries",
                "20",
                "--tpslimit",
                "8",
                "--tpslimit-burst",
                "8",
            ]
        )
        return completed.stdout

    def list_files(self, root: str) -> list[str]:
        completed = _run(
            [
                "rclone",
                "lsf",
                _remote(self.remote, root),
                "--recursive",
                "--files-only",
                "--format",
                "p",
                "--config",
                str(self.config),
                "--retries",
                "10",
                "--low-level-retries",
                "20",
                "--tpslimit",
                "8",
            ],
            check=False,
        )
        if completed.returncode:
            message = completed.stderr.lower()
            if "directory not found" in message or "object not found" in message:
                return []
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return sorted(line for line in completed.stdout.splitlines() if line)

    def cat(self, path: str) -> bytes:
        completed = subprocess.run(
            [
                "rclone",
                "cat",
                _remote(self.remote, path),
                "--config",
                str(self.config),
                "--retries",
                "10",
                "--low-level-retries",
                "20",
                "--tpslimit",
                "8",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.decode(errors="replace").strip())
        return completed.stdout

    def upload_file(self, local: Path, remote_path: str, *, tps_limit: int = 8) -> None:
        limit = max(1, tps_limit)
        _run(
            [
                "rclone",
                "copyto",
                str(local),
                _remote(self.remote, remote_path),
                "--config",
                str(self.config),
                "--retries",
                "3",
                "--low-level-retries",
                "5",
                "--drive-chunk-size",
                os.environ.get("SMASH_DRIVE_CHUNK_SIZE", DEFAULT_DRIVE_CHUNK_SIZE),
                "--tpslimit",
                str(limit),
                "--tpslimit-burst",
                str(limit),
            ]
        )

    def download_manifests(self, remote_root: str, destination: Path) -> list[Path]:
        shutil.rmtree(destination, ignore_errors=True)
        destination.mkdir(parents=True, exist_ok=True)
        completed = _run(
            [
                "rclone",
                "copy",
                _remote(self.remote, remote_root),
                str(destination),
                "--include",
                "*.manifest.jsonl",
                "--transfers",
                "8",
                "--checkers",
                "8",
                "--config",
                str(self.config),
                "--retries",
                "3",
                "--low-level-retries",
                "5",
                "--tpslimit",
                "8",
                "--tpslimit-burst",
                "8",
            ],
            check=False,
        )
        if completed.returncode:
            message = completed.stderr.lower()
            if "directory not found" in message or "object not found" in message:
                return []
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return sorted(destination.rglob("*.manifest.jsonl"))


class CoordinatorState:
    def __init__(
        self,
        *,
        drive: DriveClient,
        source_root: str,
        target_root: str,
        spool: Path,
        sample_limit: int,
        prefetch: int,
        lease_seconds: int,
        max_attempts: int,
        upload_batch_size: int,
        upload_max_bytes: int,
        upload_concurrency: int = 1,
        upload_tps_limit: int = 1,
        upload_min_batch: int = 1,
        upload_max_attempts: int = 3,
        spool_max_bytes: int = DEFAULT_SPOOL_MAX_BYTES,
        queue_root: Path | None = None,
    ) -> None:
        self.drive = drive
        self.source_root = source_root.strip("/")
        self.target_root = target_root.strip("/")
        self.spool = spool
        self.incoming = spool / "incoming"
        self.results = spool / "results"
        self.manifests = spool / "manifests"
        for path in (spool, self.incoming, self.results, self.manifests):
            path.mkdir(parents=True, exist_ok=True)
        self.sample_limit = sample_limit
        self.prefetch = prefetch
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.upload_batch_size = upload_batch_size
        self.upload_max_bytes = upload_max_bytes
        self.upload_concurrency = max(1, upload_concurrency)
        self.upload_min_batch = max(1, min(upload_batch_size, upload_min_batch))
        self.upload_max_attempts = max(1, upload_max_attempts)
        self.upload_tps_limit = max(1, upload_tps_limit)
        self.spool_max_bytes = spool_max_bytes
        self.spool_headroom_bytes = min(512 * 1024**2, max(1, spool_max_bytes // 10))
        self._reserved_result_bytes = 0
        self.queue_sync_concurrency = max(
            1, int(os.environ.get("SMASH_QUEUE_SYNC_CONCURRENCY", "16"))
        )
        self._drive_backoff_until = 0.0
        self._drive_rate_limit_streak = 0
        self.queue_root = queue_root
        if self.queue_root is not None:
            for name in (
                "sources",
                "pending",
                "claims",
                "grants",
                "leased",
                "results",
                "failures",
                "heartbeats",
            ):
                (self.queue_root / name).mkdir(parents=True, exist_ok=True)
        self.db_path = spool / "pipeline.sqlite3"
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.producer_done = False
        self.fatal_error: str | None = None
        self.started_at = time.time()
        self._init_schema()

    def _init_schema(self) -> None:
        with self.lock, self.db:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY,
                    reference TEXT NOT NULL UNIQUE,
                    archive TEXT NOT NULL,
                    member TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_token TEXT,
                    lease_deadline REAL,
                    worker_id TEXT,
                    source_path TEXT,
                    source_bytes INTEGER,
                    source_sha256 TEXT,
                    result_path TEXT,
                    result_bytes INTEGER,
                    result_token TEXT,
                    result_sha256 TEXT,
                    upload_attempts INTEGER NOT NULL DEFAULT 0,
                    upload_not_before REAL,
                    error TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status, id);
                CREATE TABLE IF NOT EXISTS batches (
                    key TEXT PRIMARY KEY,
                    archive_path TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    job_count INTEGER NOT NULL,
                    bytes INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    error TEXT
                );
                """
            )
            columns = {str(row["name"]) for row in self.db.execute("PRAGMA table_info(jobs)")}
            if "result_token" not in columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN result_token TEXT")
            if "result_sha256" not in columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN result_sha256 TEXT")
            if "upload_attempts" not in columns:
                self.db.execute(
                    "ALTER TABLE jobs ADD COLUMN upload_attempts INTEGER NOT NULL DEFAULT 0"
                )
            if "upload_not_before" not in columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN upload_not_before REAL")

    def initialize(self) -> None:
        references = self._source_references()
        uploaded = self._uploaded_references()
        filesystem_leases: set[int] = set()
        if self.queue_root is not None:
            for path in (self.queue_root / "leased").glob("*.json"):
                try:
                    descriptor = json.loads(path.read_text())
                    if descriptor.get("leaseToken"):
                        filesystem_leases.add(int(descriptor["id"]))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    # The volume is eventually consistent. The queue sync loop will retry a
                    # transient read and reap a genuinely stale descriptor later.
                    continue
        now = time.time()
        with self.lock, self.db:
            for job_id, (reference, archive, member) in enumerate(references):
                status = "uploaded" if reference in uploaded else "pending"
                self.db.execute(
                    """
                    INSERT INTO jobs(id, reference, archive, member, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(reference) DO UPDATE SET
                        status = CASE WHEN excluded.status = 'uploaded' THEN 'uploaded' ELSE jobs.status END,
                        updated_at = excluded.updated_at
                    """,
                    (job_id, reference, archive, member, status, now),
                )
            valid = tuple(reference for reference, _, _ in references)
            if valid:
                placeholders = ",".join("?" for _ in valid)
                self.db.execute(
                    f"DELETE FROM jobs WHERE reference NOT IN ({placeholders})", valid
                )
            committed_rows = list(self.db.execute("SELECT * FROM jobs WHERE status='uploaded'"))
        self._delete_local_rows(committed_rows)
        with self.lock, self.db:
            if committed_rows:
                ids = [int(row["id"]) for row in committed_rows]
                placeholders = ",".join("?" for _ in ids)
                self.db.execute(
                    f"UPDATE jobs SET source_path=NULL, source_bytes=NULL, result_path=NULL, "
                    f"result_bytes=NULL, lease_token=NULL, lease_deadline=NULL, worker_id=NULL "
                    f"WHERE id IN ({placeholders})",
                    ids,
                )
            rows = list(self.db.execute("SELECT * FROM jobs WHERE status != 'uploaded'"))
            for row in rows:
                source_exists = bool(row["source_path"] and Path(row["source_path"]).is_file())
                result_exists = bool(row["result_path"] and Path(row["result_path"]).is_file())
                if row["status"] == "failed":
                    status = "failed"
                elif row["status"] in {"complete", "uploading"} and source_exists and result_exists:
                    status = "complete"
                elif row["status"] in {"queued", "leased"} and source_exists:
                    status = "queued"
                else:
                    status = "pending"
                self.db.execute(
                    """
                    UPDATE jobs SET status=?, lease_token=NULL, lease_deadline=NULL,
                        worker_id=NULL,
                        result_token=CASE WHEN ?='complete' THEN result_token ELSE NULL END,
                        result_sha256=CASE WHEN ?='complete' THEN result_sha256 ELSE NULL END,
                        updated_at=? WHERE id=?
                    """,
                    (status, status, status, now, row["id"]),
                )
        for partial in [*self.incoming.glob("*.partial"), *self.results.glob("*.partial")]:
            partial.unlink(missing_ok=True)
        if self.queue_root is not None:
            (self.queue_root / "producer.done").unlink(missing_ok=True)
            (self.queue_root / "rendering.done").unlink(missing_ok=True)
            for partial in self.queue_root.rglob("*.partial"):
                with contextlib.suppress(OSError):
                    if time.time() - partial.stat().st_mtime > self.lease_seconds * 2:
                        partial.unlink(missing_ok=True)
            self._sweep_stale_shared_results()
            with self.lock:
                queued_ids = [
                    int(row[0])
                    for row in self.db.execute("SELECT id FROM jobs WHERE status='queued'")
                ]
            for job_id in queued_ids:
                if job_id not in filesystem_leases:
                    self._publish_pending(job_id)
        _json_line(
            "coordinator_initialized",
            references=len(references),
            alreadyUploaded=len(uploaded),
        )

    def _sweep_stale_shared_results(self) -> None:
        assert self.queue_root is not None
        now = time.time()
        for payload in (self.queue_root / "results").glob("*.tar"):
            stem = payload.name.removesuffix(".tar")
            marker = self.queue_root / "results" / f"{stem}.result.json"
            lease = self.queue_root / "leased" / f"{stem}.json"
            try:
                stale = now - payload.stat().st_mtime > self.lease_seconds * 2
            except OSError:
                continue
            if stale and not marker.exists() and not lease.exists():
                if not self._unlink_shared_paths([payload], attempts=4):
                    raise RuntimeError(f"could not remove stale shared result: {payload}")
                _json_line("stale_shared_result_removed", payload=str(payload))

    def _delete_local_rows(self, rows: Iterable[sqlite3.Row]) -> None:
        allowed_roots = [self.spool.resolve()]
        if self.queue_root is not None:
            allowed_roots.append(self.queue_root.resolve())
        pending: set[Path] = set()
        for row in rows:
            for column in ("source_path", "result_path"):
                if not row[column]:
                    continue
                path = Path(str(row[column]))
                resolved = path.resolve()
                if not any(resolved.is_relative_to(root) for root in allowed_roots):
                    raise RuntimeError(f"refusing cleanup outside coordinator storage: {path}")
                pending.add(path)
        for attempt in range(1, 6):
            for path in list(pending):
                try:
                    path.unlink(missing_ok=True)
                    if not path.exists():
                        pending.remove(path)
                except OSError:
                    continue
            if not pending:
                return
            time.sleep(min(8, 2**attempt))
        raise RuntimeError(f"could not remove committed local files: {sorted(map(str, pending))}")

    def _source_references(self) -> list[tuple[str, str, str]]:
        files = self.drive.list_files(self.source_root)
        indexes = sorted(path for path in files if path.endswith(".tar.zst.slp-index.jsonl"))
        rows: list[tuple[str, str, str]] = []
        for relative in indexes:
            archive = relative[: -len(".slp-index.jsonl")]
            for raw_line in self.drive.cat(f"{self.source_root}/{relative}").splitlines():
                row = json.loads(raw_line)
                member = str(row["member"])
                indexed_archive = str(row["archive"])
                accepted_archives = {
                    Path(archive).name,
                    archive,
                    f"{self.source_root}/{archive}".strip("/"),
                }
                if indexed_archive.strip("/") not in accepted_archives:
                    raise ValueError(f"archive index mismatch in {relative}: {indexed_archive}")
                if not member.lower().endswith(".slp"):
                    continue
                reference = f"{archive}::{member}"
                rows.append((reference, archive, member))
        rows.sort(key=lambda row: row[0])
        if self.sample_limit > 0:
            rows = rows[: self.sample_limit]
        if not rows:
            raise RuntimeError(f"no indexed SLPs found under {_remote(self.drive.remote, self.source_root)}")
        return rows

    def _uploaded_references(self) -> set[str]:
        uploaded: set[str] = set()
        committed = self.manifests / "committed"
        for manifest in self.drive.download_manifests(
            f"{self.target_root}/batches", committed
        ):
            for raw_line in manifest.read_bytes().splitlines():
                row = json.loads(raw_line)
                if row.get("status") == "complete" and row.get("sourceReference"):
                    uploaded.add(str(row["sourceReference"]))
        return uploaded

    def start(self) -> None:
        self.initialize()
        threads = [
            threading.Thread(target=self._guarded, args=("producer", self.produce), daemon=True),
            *[
                threading.Thread(
                    target=self._guarded,
                    args=(f"uploader-{index:02d}", self.upload),
                    daemon=True,
                )
                for index in range(self.upload_concurrency)
            ],
            threading.Thread(target=self._guarded, args=("reaper", self.reap), daemon=True),
        ]
        if self.queue_root is not None:
            threads.append(
                threading.Thread(
                    target=self._guarded,
                    args=("filesystem_queue", self.sync_filesystem),
                    daemon=True,
                )
            )
        for thread in threads:
            thread.start()

    def _guarded(self, name: str, function) -> None:
        try:
            function()
        except BaseException as error:
            self.fatal_error = f"{name}: {type(error).__name__}: {error}"
            self.stop.set()
            with self.condition:
                self.condition.notify_all()
            _json_line("coordinator_component_failed", component=name, error=self.fatal_error)

    def produce(self) -> None:
        with self.lock:
            archives = [
                row[0]
                for row in self.db.execute(
                    "SELECT DISTINCT archive FROM jobs WHERE status != 'uploaded' ORDER BY archive"
                )
            ]
        for archive in archives:
            if self.stop.is_set():
                break
            self._stream_archive(archive)
        with self.condition:
            self.producer_done = True
            self.condition.notify_all()
        if self.queue_root is not None:
            _atomic_json(
                self.queue_root / "producer.done",
                {"completedAt": time.time()},
            )
        _json_line("producer_done")

    def _stream_archive(self, archive: str) -> None:
        with self.lock:
            wanted = {
                str(row["member"]).removeprefix("./"): int(row["id"])
                for row in self.db.execute(
                    "SELECT id, member FROM jobs WHERE archive = ? AND status = 'pending'",
                    (archive,),
                )
            }
        if not wanted:
            return
        remote_path = _remote(self.drive.remote, f"{self.source_root}/{archive}")
        rclone = subprocess.Popen(
            [
                "rclone",
                "cat",
                remote_path,
                "--config",
                str(self.drive.config),
                "--retries",
                "10",
                "--low-level-retries",
                "20",
                "--tpslimit",
                "8",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert rclone.stdout is not None
        zstd = subprocess.Popen(
            ["zstd", "-dc"], stdin=rclone.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        rclone.stdout.close()
        assert zstd.stdout is not None
        materialized = 0
        complete_early = False
        try:
            with tarfile.open(fileobj=zstd.stdout, mode="r|") as source:
                for member in source:
                    normalized = member.name.removeprefix("./")
                    job_id = wanted.get(normalized)
                    if job_id is None or not member.isfile():
                        continue
                    self._wait_for_capacity(max(0, int(member.size)))
                    extracted = source.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"could not extract {member.name} from {archive}")
                    source_root = (
                        self.queue_root / "sources"
                        if self.queue_root is not None
                        else self.incoming
                    )
                    target = source_root / f"{job_id:06d}.slp"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = (
                        target
                        if self.queue_root is not None
                        else target.with_suffix(".slp.partial")
                    )
                    digest = hashlib.sha256()
                    size = 0
                    with temporary.open("wb") as output:
                        while chunk := extracted.read(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if temporary != target:
                        temporary.replace(target)
                    with self.condition, self.db:
                        self.db.execute(
                            """
                            UPDATE jobs SET status='queued', source_path=?, source_bytes=?,
                                source_sha256=?, updated_at=? WHERE id=? AND status='pending'
                            """,
                            (str(target), size, digest.hexdigest(), time.time(), job_id),
                        )
                        self.condition.notify_all()
                    self._publish_pending(job_id)
                    materialized += 1
                    wanted.pop(normalized, None)
                    if not wanted:
                        complete_early = True
                        break
            if complete_early:
                with contextlib.suppress(Exception):
                    zstd.kill()
                    rclone.kill()
                zstd.wait()
                rclone.wait()
            else:
                zstd_stderr = zstd.stderr.read().decode(errors="replace") if zstd.stderr else ""
                rclone_stderr = rclone.stderr.read().decode(errors="replace") if rclone.stderr else ""
                zstd_status = zstd.wait()
                rclone_status = rclone.wait()
                if zstd_status or rclone_status:
                    raise RuntimeError(zstd_stderr.strip() or rclone_stderr.strip())
                if wanted:
                    raise RuntimeError(
                        f"archive {archive} is missing {len(wanted)} indexed SLP members: "
                        f"{sorted(wanted)[:10]}"
                    )
        except BaseException:
            with contextlib.suppress(Exception):
                zstd.kill()
                rclone.kill()
            with contextlib.suppress(Exception):
                zstd.wait()
                rclone.wait()
            raise
        _json_line("source_archive_streamed", archive=archive, materialized=materialized)

    def _publish_pending(self, job_id: int) -> None:
        if self.queue_root is None:
            return
        (self.queue_root / "rendering.done").unlink(missing_ok=True)
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "queued":
                return
            descriptor = {
                "id": int(row["id"]),
                "reference": row["reference"],
                "sourcePath": row["source_path"],
                "sourceBytes": int(row["source_bytes"]),
                "sourceSha256": row["source_sha256"],
                "attempts": int(row["attempts"]),
                "leaseSeconds": self.lease_seconds,
            }
        _atomic_json(self.queue_root / "pending" / f"{job_id:06d}.json", descriptor)

    def _wait_for_capacity(self, required_bytes: int = 0) -> None:
        with self.condition:
            while not self.stop.is_set():
                row = self.db.execute(
                    """
                    SELECT COUNT(*) AS count,
                           COALESCE(SUM(COALESCE(source_bytes, 0) +
                                        COALESCE(result_bytes, 0)), 0) AS bytes
                    FROM jobs
                    WHERE status IN ('queued', 'leased', 'complete', 'uploading')
                    """
                ).fetchone()
                if (
                    int(row["count"]) < self.prefetch
                    and int(row["bytes"])
                    + self._reserved_result_bytes
                    + required_bytes
                    <= self.spool_max_bytes
                ):
                    return
                self.condition.wait(timeout=2)
        raise RuntimeError("coordinator stopped while producer was waiting")

    def _reserve_result_capacity(self, required_bytes: int) -> None:
        if required_bytes + self.spool_headroom_bytes > self.spool_max_bytes:
            raise ValueError(
                f"result of {required_bytes} bytes cannot fit the coordinator spool safely"
            )
        with self.condition:
            while not self.stop.is_set():
                row = self.db.execute(
                    """
                    SELECT COALESCE(SUM(COALESCE(source_bytes, 0) +
                                        COALESCE(result_bytes, 0)), 0) AS bytes
                    FROM jobs
                    WHERE status IN ('queued', 'leased', 'complete', 'uploading')
                    """
                ).fetchone()
                reserved_after = self._reserved_result_bytes + required_bytes
                disk_free = shutil.disk_usage(self.spool).free
                if (
                    int(row["bytes"]) + reserved_after <= self.spool_max_bytes
                    and disk_free >= reserved_after + self.spool_headroom_bytes
                ):
                    self._reserved_result_bytes = reserved_after
                    return
                self.condition.wait(timeout=2)
        raise RuntimeError("coordinator stopped while reserving result capacity")

    def _release_result_capacity(self, reserved_bytes: int) -> None:
        with self.condition:
            self._reserved_result_bytes -= reserved_bytes
            if self._reserved_result_bytes < 0:
                self._reserved_result_bytes = 0
                raise RuntimeError("coordinator result reservation underflow")
            self.condition.notify_all()

    def lease(self, worker_id: str) -> dict | None:
        now = time.time()
        with self.condition, self.db:
            row = self.db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            token = secrets.token_urlsafe(24)
            self.db.execute(
                """
                UPDATE jobs SET status='leased', attempts=attempts+1, lease_token=?,
                    lease_deadline=?, worker_id=?, updated_at=? WHERE id=? AND status='queued'
                """,
                (token, now + self.lease_seconds, worker_id, now, row["id"]),
            )
            leased = self.db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
            return {
                "id": int(leased["id"]),
                "reference": leased["reference"],
                "leaseToken": token,
                "sourceBytes": int(leased["source_bytes"]),
                "sourceSha256": leased["source_sha256"],
            }

    def source(self, job_id: int, lease_token: str) -> Path:
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "leased" or row["lease_token"] != lease_token:
                raise PermissionError("invalid or expired lease")
            return Path(row["source_path"])

    def fail_job(self, job_id: int, lease_token: str, error: str) -> None:
        if _is_no_playable_frames_error(error):
            self._complete_no_playable_frames(job_id, lease_token)
            return
        retry = False
        with self.condition, self.db:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "leased" or row["lease_token"] != lease_token:
                return
            status = "failed" if int(row["attempts"]) >= self.max_attempts else "queued"
            retry = status == "queued"
            self.db.execute(
                """
                UPDATE jobs SET status=?, lease_token=NULL, lease_deadline=NULL,
                    worker_id=NULL, error=?, updated_at=? WHERE id=?
                """,
                (status, error[-4000:], time.time(), job_id),
            )
            self.condition.notify_all()
        if retry:
            self._publish_pending(job_id)

    def _complete_no_playable_frames(self, job_id: int, lease_token: str) -> None:
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] != "leased" or row["lease_token"] != lease_token:
                return
        payload = _no_playable_frames_result_tar()
        result_bytes = len(payload)
        self._reserve_result_capacity(result_bytes)
        target = self.results / f"{job_id:06d}.tar"
        token_key = hashlib.sha256(lease_token.encode()).hexdigest()[:12]
        temporary = self.results / f"{job_id:06d}.{token_key}.{secrets.token_hex(4)}.partial"
        try:
            temporary.write_bytes(payload)
            result_sha256 = hashlib.sha256(payload).hexdigest()
            with self.condition, self.db:
                cursor = self.db.execute(
                    """
                    UPDATE jobs SET status='complete', result_path=?, result_bytes=?,
                        result_token=?, result_sha256=?, lease_token=NULL,
                        lease_deadline=NULL, worker_id=NULL, error=?, updated_at=?
                    WHERE id=? AND status='leased' AND lease_token=?
                    """,
                    (
                        str(target),
                        result_bytes,
                        lease_token,
                        result_sha256,
                        SKIPPED_NO_PLAYABLE_FRAMES,
                        time.time(),
                        job_id,
                        lease_token,
                    ),
                )
                if cursor.rowcount == 1:
                    temporary.replace(target)
                    self.condition.notify_all()
        finally:
            temporary.unlink(missing_ok=True)
            self._release_result_capacity(result_bytes)

    def accept_result(
        self,
        job_id: int,
        lease_token: str,
        source: BinaryIO,
        length: int,
        *,
        expected_sha256: str | None = None,
    ) -> None:
        if length <= 0 or length > 2 * 1024 * 1024 * 1024:
            raise ValueError("invalid result length")
        with self.lock:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            current_lease = bool(
                row is not None
                and row["status"] == "leased"
                and row["lease_token"] == lease_token
            )
            committed_retry = bool(
                row is not None
                and row["status"] in {"complete", "uploading", "uploaded"}
                and row["result_token"] == lease_token
            )
            if not current_lease and not committed_retry:
                raise PermissionError("invalid or expired lease")
            if (
                committed_retry
                and expected_sha256 is not None
                and row["result_sha256"] == expected_sha256
            ):
                return
        self._reserve_result_capacity(length)
        target = self.results / f"{job_id:06d}.tar"
        token_key = hashlib.sha256(lease_token.encode()).hexdigest()[:12]
        temporary = self.results / f"{job_id:06d}.{token_key}.{secrets.token_hex(4)}.partial"
        remaining = length
        digest = hashlib.sha256()
        try:
            with temporary.open("wb") as output:
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise EOFError("worker result ended early")
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            result_sha256 = digest.hexdigest()
            if expected_sha256 is not None and result_sha256 != expected_sha256:
                raise ValueError(
                    f"result checksum mismatch: {result_sha256} != {expected_sha256}"
                )
            with self.lock:
                row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                if (
                    row is not None
                    and row["status"] in {"complete", "uploading", "uploaded"}
                    and row["result_token"] == lease_token
                    and row["result_sha256"] == result_sha256
                ):
                    return
                if (
                    row is None
                    or row["status"] != "leased"
                    or row["lease_token"] != lease_token
                ):
                    raise PermissionError("lease expired while result was uploading")
            self._validate_result(temporary, job_id)
            result_bytes = temporary.stat().st_size
            with self.condition, self.db:
                cursor = self.db.execute(
                    """
                    UPDATE jobs SET status='complete', result_path=?, result_bytes=?,
                        result_token=?, result_sha256=?, lease_token=NULL,
                        lease_deadline=NULL, updated_at=?
                    WHERE id=? AND status='leased' AND lease_token=?
                    """,
                    (
                        str(target),
                        result_bytes,
                        lease_token,
                        result_sha256,
                        time.time(),
                        job_id,
                        lease_token,
                    ),
                )
                if cursor.rowcount != 1:
                    row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                    if not (
                        row is not None
                        and row["status"] in {"complete", "uploading", "uploaded"}
                        and row["result_token"] == lease_token
                        and row["result_sha256"] == result_sha256
                    ):
                        raise PermissionError("lease expired while result was committing")
                else:
                    temporary.replace(target)
                    self.condition.notify_all()
        finally:
            temporary.unlink(missing_ok=True)
            self._release_result_capacity(length)

    def _validate_result(self, path: Path, job_id: int) -> None:
        with tarfile.open(path, "r") as result:
            names = {member.name for member in result.getmembers() if member.isfile()}
            if names != {"video.mp4", "metadata.json"}:
                raise ValueError(f"invalid result members for {job_id}: {sorted(names)}")
            metadata_file = result.extractfile("metadata.json")
            if metadata_file is None:
                raise ValueError("result is missing metadata.json")
            metadata = json.load(metadata_file)
            video = metadata.get("video") or {}
            expected = {
                "width": VIDEO_WIDTH,
                "height": VIDEO_HEIGHT,
                "frameRate": f"{VIDEO_FPS}/1",
                "container": "mp4",
                "codec": "h264",
                "pixelFormat": "yuv420p",
                "sourceFps": SOURCE_FPS,
                "targetFps": VIDEO_FPS,
                "firstSelectedSlpFrame": -39,
                "sourceFrameStep": 3,
                "cpuOnly": True,
            }
            for key, value in expected.items():
                if video.get(key) != value:
                    raise ValueError(f"result {job_id} has invalid video {key}: {video.get(key)}")
            cropped = video.get("croppedTailSourceFrames")
            last_source = video.get("lastSourceSlpFrame")
            last_selected = video.get("lastSelectedSlpFrame")
            if cropped not in {0, 1, 2}:
                raise ValueError(f"result {job_id} has invalid cropped video tail: {cropped}")
            if not isinstance(last_source, int) or not isinstance(last_selected, int):
                raise ValueError(f"result {job_id} is missing video frame endpoints")
            if last_source - last_selected != cropped:
                raise ValueError(f"result {job_id} has inconsistent video frame endpoints")

    @staticmethod
    def _lease_stem(job_id: int, lease_token: str) -> str:
        return f"{job_id:06d}-{lease_token}"

    def _cleanup_shared_lease(
        self,
        job_id: int,
        lease_token: str,
        *,
        include_result: bool,
    ) -> bool:
        if self.queue_root is None:
            return True
        stem = self._lease_stem(job_id, lease_token)
        result_payload = self.queue_root / "results" / f"{stem}.tar"
        result_marker = self.queue_root / "results" / f"{stem}.result.json"
        if include_result and not self._unlink_shared_paths([result_payload]):
            _json_line(
                "shared_cleanup_retry",
                sample=job_id,
                payload=str(result_payload),
            )
            return False
        paths = [
            self.queue_root / "claims" / f"{stem}.json",
            self.queue_root / "grants" / f"{stem}.json",
            self.queue_root / "leased" / f"{stem}.json",
            self.queue_root / "heartbeats" / f"{stem}.json",
            self.queue_root / "failures" / f"{stem}.failure.json",
        ]
        if include_result:
            # The marker is removed only after the payload is confirmed absent. A retained
            # marker makes an eventually-consistent unlink retryable by the next sync pass.
            paths.append(result_marker)
        removed = self._unlink_shared_paths(paths)
        if not removed:
            _json_line("shared_cleanup_retry", sample=job_id, metadata=True)
        return removed

    @staticmethod
    def _unlink_shared_paths(paths: Iterable[Path], *, attempts: int = 1) -> bool:
        pending = set(paths)
        for attempt in range(1, max(1, attempts) + 1):
            for path in list(pending):
                try:
                    path.unlink(missing_ok=True)
                    if not path.exists():
                        pending.remove(path)
                except OSError:
                    continue
            if not pending:
                return True
            if attempt < attempts:
                time.sleep(min(4, 2**attempt))
        return False

    def _sync_filesystem_claim(self, claim_path: Path) -> None:
        try:
            claim = json.loads(claim_path.read_text())
            job_id = int(claim["id"])
            lease_token = str(claim["leaseToken"])
            worker_id = str(claim["workerId"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            with contextlib.suppress(OSError):
                if time.time() - claim_path.stat().st_mtime > self.lease_seconds:
                    claim_path.unlink(missing_ok=True)
            return
        stem = self._lease_stem(job_id, lease_token)
        expected = self.queue_root / "claims" / f"{stem}.json"
        if claim_path != expected:
            claim_path.unlink(missing_ok=True)
            return
        granted = False
        row: sqlite3.Row | None = None
        now = time.time()
        with self.condition, self.db:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is not None and row["status"] == "queued":
                cursor = self.db.execute(
                    """
                    UPDATE jobs SET status='leased', attempts=attempts+1,
                        lease_token=?, lease_deadline=?, worker_id=?, updated_at=?
                    WHERE id=? AND status='queued' AND attempts < ?
                    """,
                    (
                        lease_token,
                        now + min(120, self.lease_seconds),
                        worker_id,
                        now,
                        job_id,
                        self.max_attempts,
                    ),
                )
                granted = cursor.rowcount == 1
                if granted:
                    row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                    self.condition.notify_all()
                elif int(row["attempts"]) >= self.max_attempts:
                    self.db.execute(
                        "UPDATE jobs SET status='failed', error='maximum attempts exceeded', "
                        "updated_at=? WHERE id=? AND status='queued'",
                        (now, job_id),
                    )
                    row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
                    self.condition.notify_all()
            elif (
                row is not None
                and row["status"] == "leased"
                and row["lease_token"] == lease_token
            ):
                granted = True
        if not granted or row is None:
            if row is None or row["status"] != "queued":
                (self.queue_root / "pending" / f"{job_id:06d}.json").unlink(
                    missing_ok=True
                )
            _atomic_json(
                self.queue_root / "grants" / f"{stem}.json",
                {
                    "id": job_id,
                    "leaseToken": lease_token,
                    "granted": False,
                    "decidedAt": time.time(),
                },
            )
            claim_path.unlink(missing_ok=True)
            return
        descriptor = {
            "id": job_id,
            "reference": row["reference"],
            "sourcePath": row["source_path"],
            "sourceBytes": int(row["source_bytes"]),
            "sourceSha256": row["source_sha256"],
            "attempts": int(row["attempts"]),
            "leaseToken": lease_token,
            "workerId": worker_id,
            "leaseDeadline": float(row["lease_deadline"]),
            "leaseSeconds": self.lease_seconds,
            "granted": True,
        }
        # The coordinator is the sole grant writer. A worker cannot render until this grant
        # exists, so the S3/FUSE volume's non-transactional rename semantics are irrelevant.
        leased_path = self.queue_root / "leased" / f"{stem}.json"
        grant_path = self.queue_root / "grants" / f"{stem}.json"
        if not leased_path.exists():
            _atomic_json(leased_path, descriptor)
        if not grant_path.exists():
            _atomic_json(grant_path, descriptor)
        (self.queue_root / "pending" / f"{job_id:06d}.json").unlink(missing_ok=True)

    def _sync_filesystem_lease(self, lease_path: Path) -> None:
        try:
            descriptor = json.loads(lease_path.read_text())
            job_id = int(descriptor["id"])
            lease_token = str(descriptor["leaseToken"])
            worker_id = str(descriptor["workerId"])
            attempts = int(descriptor["attempts"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            with contextlib.suppress(OSError):
                stale = time.time() - lease_path.stat().st_mtime > self.lease_seconds
                if stale:
                    lease_path.unlink(missing_ok=True)
            return
        expected = self.queue_root / "leased" / f"{self._lease_stem(job_id, lease_token)}.json"
        if lease_path != expected:
            lease_path.unlink(missing_ok=True)
            return
        heartbeat = self.queue_root / "heartbeats" / f"{self._lease_stem(job_id, lease_token)}.json"
        deadline = min(
            float(descriptor.get("leaseDeadline", 0)),
            time.time() + self.lease_seconds,
        )
        heartbeat_seen = False
        with contextlib.suppress(OSError):
            deadline = heartbeat.stat().st_mtime + self.lease_seconds
            heartbeat_seen = True
        cleanup = False
        adopted = False
        with self.condition, self.db:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["status"] in {"complete", "uploading", "uploaded", "failed"}:
                cleanup = True
            elif row["status"] == "queued":
                cursor = self.db.execute(
                    """
                    UPDATE jobs SET status='leased', attempts=?, lease_token=?,
                        lease_deadline=?, worker_id=?, updated_at=?
                    WHERE id=? AND status='queued'
                    """,
                    (max(int(row["attempts"]), attempts), lease_token, deadline,
                     worker_id, time.time(), job_id),
                )
                if cursor.rowcount:
                    adopted = True
                    self.condition.notify_all()
            elif row["status"] == "leased" and row["lease_token"] == lease_token:
                if heartbeat_seen and deadline > float(row["lease_deadline"] or 0):
                    self.db.execute(
                        "UPDATE jobs SET lease_deadline=?, updated_at=? WHERE id=?",
                        (deadline, time.time(), job_id),
                    )
                adopted = True
            else:
                cleanup = True
        if cleanup:
            self._cleanup_shared_lease(job_id, lease_token, include_result=True)
        elif adopted:
            descriptor["granted"] = True
            grant = self.queue_root / "grants" / f"{self._lease_stem(job_id, lease_token)}.json"
            if not grant.exists():
                _atomic_json(grant, descriptor)

    def _sync_filesystem_result(self, marker: Path) -> None:
        job_id: int | None = None
        lease_token: str | None = None
        cleanup = False
        try:
            descriptor = json.loads(marker.read_text())
            job_id = int(descriptor["id"])
            lease_token = str(descriptor["leaseToken"])
            stem = self._lease_stem(job_id, lease_token)
            expected_marker = self.queue_root / "results" / f"{stem}.result.json"
            expected_result = self.queue_root / "results" / f"{stem}.tar"
            result = Path(str(descriptor["resultPath"]))
            length = int(descriptor["resultBytes"])
            result_sha256 = str(descriptor["resultSha256"])
            if marker != expected_marker or result != expected_result:
                raise ValueError("result marker path does not match its lease")
            if not result.is_file():
                if time.time() - marker.stat().st_mtime <= 60:
                    return
                raise FileNotFoundError("shared result payload did not become visible")
            if result.stat().st_size != length:
                raise ValueError("shared result size does not match its marker")
            with result.open("rb") as source:
                self.accept_result(
                    job_id,
                    lease_token,
                    source,
                    length,
                    expected_sha256=result_sha256,
                )
            cleanup = True
        except PermissionError:
            cleanup = True
        except (KeyError, OSError, TypeError, ValueError, EOFError, tarfile.TarError,
                json.JSONDecodeError) as error:
            with contextlib.suppress(OSError):
                if time.time() - marker.stat().st_mtime <= 60:
                    return
            if job_id is not None and lease_token is not None:
                self.fail_job(job_id, lease_token, f"invalid shared result: {error}")
                cleanup = True
            else:
                marker.unlink(missing_ok=True)
            _json_line("filesystem_result_rejected", marker=str(marker), error=str(error))
        if cleanup and job_id is not None and lease_token is not None:
            self._cleanup_shared_lease(job_id, lease_token, include_result=True)

    def _sync_filesystem_failure(self, marker: Path) -> None:
        try:
            descriptor = json.loads(marker.read_text())
            job_id = int(descriptor["id"])
            lease_token = str(descriptor["leaseToken"])
            error = str(descriptor.get("error", "worker failed"))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as parse_error:
            with contextlib.suppress(OSError):
                if time.time() - marker.stat().st_mtime <= 60:
                    return
            _json_line("filesystem_failure_rejected", marker=str(marker), error=str(parse_error))
            marker.unlink(missing_ok=True)
            return
        stem = self._lease_stem(job_id, lease_token)
        if marker != self.queue_root / "failures" / f"{stem}.failure.json":
            marker.unlink(missing_ok=True)
            return
        if (self.queue_root / "results" / f"{stem}.result.json").exists():
            return
        self.fail_job(job_id, lease_token, error)
        self._cleanup_shared_lease(job_id, lease_token, include_result=True)

    def _update_rendering_done(self) -> None:
        with self.lock:
            active = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','queued','leased')"
                ).fetchone()[0]
            )
        marker = self.queue_root / "rendering.done"
        if self.producer_done and active == 0:
            if not marker.exists():
                _atomic_json(marker, {"completedAt": time.time()})
        else:
            marker.unlink(missing_ok=True)

    def sync_filesystem(self) -> None:
        assert self.queue_root is not None
        next_lease_reconcile = 0.0
        with ThreadPoolExecutor(max_workers=self.queue_sync_concurrency) as executor:
            def run_tasks(tasks) -> None:
                futures = {
                    executor.submit(function, path): (kind, path)
                    for kind, path, function in tasks
                }
                for future in as_completed(futures):
                    kind, path = futures[future]
                    try:
                        future.result()
                    except Exception as error:
                        _json_line(
                            "filesystem_queue_retry",
                            kind=kind,
                            path=str(path),
                            error=str(error),
                        )

            while not self.stop.is_set():
                if time.monotonic() >= next_lease_reconcile:
                    lease_tasks = [
                        ("lease", path, self._sync_filesystem_lease)
                        for path in sorted((self.queue_root / "leased").glob("*.json"))
                    ]
                    # A restarted coordinator normalizes durable leases to queued. Reconcile
                    # every lease before result/failure processing so valid completed work is
                    # not rejected and deleted merely because its token has not been adopted yet.
                    run_tasks(lease_tasks)
                    next_lease_reconcile = time.monotonic() + 30
                operations = (
                    ("claim", self.queue_root / "claims", "*.json", self._sync_filesystem_claim),
                    ("result", self.queue_root / "results", "*.result.json", self._sync_filesystem_result),
                    ("failure", self.queue_root / "failures", "*.failure.json", self._sync_filesystem_failure),
                )
                groups = [
                    [(kind, path, function) for path in sorted(directory.glob(pattern))]
                    for kind, directory, pattern, function in operations
                ]
                tasks = []
                for index in range(max((len(group) for group in groups), default=0)):
                    for group in groups:
                        if index < len(group):
                            tasks.append(group[index])
                run_tasks(tasks)
                try:
                    self._update_rendering_done()
                except Exception as error:
                    _json_line("filesystem_queue_retry", kind="done", error=str(error))
                self.stop.wait(0.5)

    def upload(self) -> None:
        while not self.stop.is_set():
            if not self._wait_for_drive_backoff():
                return
            rows = self._claim_upload_batch()
            if not rows:
                if self.is_terminal():
                    return
                with self.condition:
                    self.condition.wait(timeout=5)
                continue
            try:
                self._upload_rows(rows)
                with self.condition:
                    if time.monotonic() >= self._drive_backoff_until:
                        self._drive_rate_limit_streak = 0
            except PostCommitCleanupError:
                raise
            except Exception as error:
                if self._is_drive_rate_limit(error):
                    retry_after = self._record_drive_rate_limit(rows, error)
                    _json_line(
                        "drive_rate_limited",
                        count=len(rows),
                        retryAfterSeconds=round(retry_after, 3),
                        error=str(error),
                    )
                    continue
                exhausted = self._record_upload_failure(rows, error)
                attempts = max(int(row["upload_attempts"]) for row in rows)
                _json_line(
                    "upload_failed",
                    count=len(rows),
                    attempt=attempts,
                    maxAttempts=self.upload_max_attempts,
                    error=str(error),
                )
                if exhausted:
                    raise RuntimeError(
                        f"Drive upload exhausted {self.upload_max_attempts} attempts: {error}"
                    ) from error

    @staticmethod
    def _is_drive_rate_limit(error: Exception) -> bool:
        message = str(error).lower()
        return "ratelimitexceeded" in message or (
            "quota exceeded" in message and "queries" in message
        )

    def _wait_for_drive_backoff(self) -> bool:
        with self.condition:
            while not self.stop.is_set():
                remaining = self._upload_backoff_remaining()
                if remaining <= 0:
                    return True
                self.condition.wait(timeout=min(5, remaining))
            return False

    def _upload_backoff_remaining(self) -> float:
        persisted = self.db.execute(
            "SELECT MAX(upload_not_before) FROM jobs WHERE status='complete'"
        ).fetchone()[0]
        return max(
            0.0,
            self._drive_backoff_until - time.monotonic(),
            float(persisted or 0) - time.time(),
        )

    def _record_drive_rate_limit(
        self, rows: list[sqlite3.Row], error: Exception
    ) -> float:
        with self.condition, self.db:
            monotonic_now = time.monotonic()
            if monotonic_now >= self._drive_backoff_until:
                self._drive_rate_limit_streak += 1
                delay = min(300.0, 60.0 * (2 ** (self._drive_rate_limit_streak - 1)))
                self._drive_backoff_until = monotonic_now + delay
            remaining = max(0.0, self._drive_backoff_until - monotonic_now)
            wall_now = time.time()
            not_before = wall_now + remaining
            for row in rows:
                stored_error = (
                    SKIPPED_NO_PLAYABLE_FRAMES
                    if row["error"] == SKIPPED_NO_PLAYABLE_FRAMES
                    else str(error)[-4000:]
                )
                self.db.execute(
                    "UPDATE jobs SET status='complete', "
                    "upload_attempts=MAX(0, upload_attempts-1), "
                    "upload_not_before=MAX(COALESCE(upload_not_before, 0), ?), "
                    "error=?, updated_at=? "
                    "WHERE id=? AND status='uploading'",
                    (not_before, stored_error, wall_now, int(row["id"])),
                )
            self.condition.notify_all()
            return remaining

    def _record_upload_failure(
        self, rows: list[sqlite3.Row], error: Exception
    ) -> bool:
        exhausted = False
        with self.condition, self.db:
            now = time.time()
            not_before = now + UPLOAD_RETRY_SECONDS
            for row in rows:
                attempts = int(row["upload_attempts"])
                status = "failed" if attempts >= self.upload_max_attempts else "complete"
                exhausted = exhausted or status == "failed"
                stored_error = (
                    SKIPPED_NO_PLAYABLE_FRAMES
                    if status == "complete" and row["error"] == SKIPPED_NO_PLAYABLE_FRAMES
                    else str(error)[-4000:]
                )
                self.db.execute(
                    "UPDATE jobs SET status=?, "
                    "upload_not_before=CASE WHEN ?='complete' "
                    "THEN MAX(COALESCE(upload_not_before, 0), ?) ELSE NULL END, "
                    "error=?, updated_at=? "
                    "WHERE id=? AND status='uploading'",
                    (
                        status,
                        status,
                        not_before,
                        stored_error,
                        now,
                        int(row["id"]),
                    ),
                )
            self.condition.notify_all()
        return exhausted

    def _claim_upload_batch(self) -> list[sqlite3.Row]:
        with self.condition, self.db:
            # Recheck under the claim lock so a thread that passed the outer wait before another
            # uploader failed cannot immediately reclaim that uploader's rows.
            if self._upload_backoff_remaining() > 0:
                return []
            exhausted = self.db.execute(
                "SELECT id FROM jobs WHERE status='complete' AND upload_attempts>=? "
                "ORDER BY id LIMIT 1",
                (self.upload_max_attempts,),
            ).fetchone()
            if exhausted is not None:
                self.db.execute(
                    "UPDATE jobs SET status='failed', error='Drive upload attempts exhausted', "
                    "updated_at=? WHERE id=? AND status='complete'",
                    (time.time(), int(exhausted["id"])),
                )
                self.condition.notify_all()
                raise RuntimeError(
                    f"Drive upload attempts exhausted for job {int(exhausted['id'])}"
                )
            candidates = list(
                self.db.execute(
                    "SELECT * FROM jobs WHERE status='complete' AND upload_attempts<? "
                    "ORDER BY id LIMIT ?",
                    (self.upload_max_attempts, self.upload_batch_size),
                )
            )
            rows: list[sqlite3.Row] = []
            total_bytes = 0
            for row in candidates:
                row_bytes = int(row["source_bytes"] or 0) + int(row["result_bytes"] or 0)
                if rows and total_bytes + row_bytes > self.upload_max_bytes:
                    break
                rows.append(row)
                total_bytes += row_bytes
            if not rows:
                return []
            upstream = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN ('pending','queued','leased')"
                ).fetchone()[0]
            )
            oldest_age = time.time() - float(rows[0]["updated_at"])
            at_capacity = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN "
                    "('queued','leased','complete','uploading')"
                ).fetchone()[0]
            ) >= self.prefetch
            should_flush = (
                len(rows) >= self.upload_batch_size
                or total_bytes >= self.upload_max_bytes
                or oldest_age >= 300
                or (at_capacity and len(rows) >= self.upload_min_batch)
                or (self.producer_done and upstream == 0)
            )
            if not should_flush:
                return []
            ids = [int(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in ids)
            self.db.execute(
                f"UPDATE jobs SET status='uploading', upload_attempts=upload_attempts+1, "
                f"updated_at=? WHERE id IN ({placeholders})",
                (time.time(), *ids),
            )
            return list(
                self.db.execute(
                    f"SELECT * FROM jobs WHERE id IN ({placeholders}) ORDER BY id", ids
                )
            )

    def _upload_rows(self, rows: list[sqlite3.Row]) -> None:
        identity = "\n".join(str(row["reference"]) for row in rows).encode()
        key = hashlib.sha256(identity).hexdigest()[:20]
        archive_relative = f"{self.target_root}/batches/batch-{key}.tar.zst"
        manifest_relative = f"{self.target_root}/batches/batch-{key}.manifest.jsonl"
        manifest = self.manifests / f"batch-{key}.manifest.jsonl"
        with manifest.open("w") as output:
            for row in rows:
                entry = {
                    "schemaVersion": 1,
                    "status": "complete",
                    "sample": int(row["id"]),
                    "sourceReference": row["reference"],
                    "sourceSha256": row["source_sha256"],
                    "archive": archive_relative,
                }
                if row["error"] == SKIPPED_NO_PLAYABLE_FRAMES:
                    entry.update(
                        {
                            "artifact": "skipped",
                            "skipReason": NO_PLAYABLE_FRAMES,
                        }
                    )
                output.write(
                    json.dumps(
                        entry,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        self._stream_result_archive(rows, archive_relative)
        self.drive.upload_file(
            manifest,
            manifest_relative,
            tps_limit=self.upload_tps_limit,
        )
        try:
            self._delete_local_rows(rows)
            with self.condition, self.db:
                ids = [int(row["id"]) for row in rows]
                placeholders = ",".join("?" for _ in ids)
                self.db.execute(
                    f"UPDATE jobs SET status='uploaded', source_path=NULL, source_bytes=NULL, "
                    f"result_path=NULL, result_bytes=NULL, lease_token=NULL, "
                    f"lease_deadline=NULL, worker_id=NULL, updated_at=? "
                    f"WHERE id IN ({placeholders})",
                    (time.time(), *ids),
                )
                self.condition.notify_all()
        except Exception as error:
            raise PostCommitCleanupError(
                f"Drive manifest {manifest_relative} committed but local cleanup failed: {error}"
            ) from error
        _json_line("uploaded_batch", key=key, count=len(rows), target=archive_relative)

    def _stream_result_archive(self, rows: list[sqlite3.Row], remote_path: str) -> None:
        rclone = subprocess.Popen(
            [
                "rclone",
                "rcat",
                _remote(self.drive.remote, remote_path),
                "--config",
                str(self.drive.config),
                "--retries",
                "3",
                "--low-level-retries",
                "5",
                "--drive-chunk-size",
                os.environ.get("SMASH_DRIVE_CHUNK_SIZE", DEFAULT_DRIVE_CHUNK_SIZE),
                "--tpslimit",
                str(self.upload_tps_limit),
                "--tpslimit-burst",
                str(self.upload_tps_limit),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert rclone.stdin is not None
        zstd = subprocess.Popen(
            ["zstd", "-T1", "-1", "-c"],
            stdin=subprocess.PIPE,
            stdout=rclone.stdin,
            stderr=subprocess.PIPE,
        )
        rclone.stdin.close()
        assert zstd.stdin is not None
        try:
            with tarfile.open(fileobj=zstd.stdin, mode="w|") as output:
                for row in rows:
                    prefix = f"slp_with_video/{int(row['id'])}"
                    output.add(row["source_path"], arcname=f"{prefix}/input.slp", recursive=False)
                    with tarfile.open(row["result_path"], "r") as result:
                        result_members = (
                            ("metadata.json",)
                            if row["error"] == SKIPPED_NO_PLAYABLE_FRAMES
                            else ("video.mp4", "metadata.json")
                        )
                        for name in result_members:
                            member = result.getmember(name)
                            extracted = result.extractfile(member)
                            if extracted is None:
                                raise RuntimeError(f"missing {name} for job {row['id']}")
                            member.name = f"{prefix}/{name}"
                            output.addfile(member, extracted)
            zstd.stdin.close()
            zstd_stderr = zstd.stderr.read().decode(errors="replace") if zstd.stderr else ""
            zstd_status = zstd.wait()
            rclone_stderr = rclone.stderr.read().decode(errors="replace") if rclone.stderr else ""
            rclone_status = rclone.wait()
            if zstd_status or rclone_status:
                detail = "\n".join(
                    part.strip() for part in (rclone_stderr, zstd_stderr) if part.strip()
                )
                raise RuntimeError(detail)
        except BaseException as error:
            for process in (zstd, rclone):
                with contextlib.suppress(Exception):
                    process.kill()
            zstd.wait()
            rclone.wait()
            zstd_stderr = zstd.stderr.read().decode(errors="replace") if zstd.stderr else ""
            rclone_stderr = rclone.stderr.read().decode(errors="replace") if rclone.stderr else ""
            detail = "\n".join(
                part.strip() for part in (rclone_stderr, zstd_stderr) if part.strip()
            )
            if isinstance(error, Exception) and detail:
                raise RuntimeError(detail) from error
            raise

    def reap(self) -> None:
        while not self.stop.wait(5):
            now = time.time()
            with self.lock:
                candidates = list(self.db.execute("SELECT * FROM jobs WHERE status='leased'"))
            for row in candidates:
                job_id = int(row["id"])
                lease_token = str(row["lease_token"])
                deadline = float(row["lease_deadline"] or 0)
                if self.queue_root is not None:
                    stem = self._lease_stem(job_id, lease_token)
                    for path in (
                        self.queue_root / "heartbeats" / f"{stem}.json",
                        self.queue_root / "results" / f"{stem}.tar",
                        self.queue_root / "results" / f"{stem}.result.json",
                    ):
                        with contextlib.suppress(OSError):
                            deadline = max(deadline, path.stat().st_mtime + self.lease_seconds)
                if deadline >= now:
                    continue
                retry = int(row["attempts"]) < self.max_attempts
                status = "queued" if retry else "failed"
                with self.condition, self.db:
                    cursor = self.db.execute(
                        """
                        UPDATE jobs SET status=?, lease_token=NULL, lease_deadline=NULL,
                            worker_id=NULL, error='lease expired', updated_at=?
                        WHERE id=? AND status='leased' AND lease_token=?
                        """,
                        (status, now, job_id, lease_token),
                    )
                    if cursor.rowcount:
                        self.condition.notify_all()
                if not cursor.rowcount:
                    continue
                self._cleanup_shared_lease(job_id, lease_token, include_result=True)
                if retry:
                    self._publish_pending(job_id)
                _json_line("lease_expired", sample=job_id, retry=retry)

    def is_terminal(self) -> bool:
        if not self.producer_done:
            return False
        with self.lock:
            active = int(
                self.db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status IN "
                    "('pending','queued','leased','complete','uploading')"
                ).fetchone()[0]
            )
        return active == 0

    def health(self) -> dict:
        with self.lock:
            counts = {
                row["status"]: int(row["count"])
                for row in self.db.execute(
                    "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
                )
            }
            total = sum(counts.values())
            errors = [
                {"sample": int(row["id"]), "source": row["reference"], "error": row["error"]}
                for row in self.db.execute(
                    "SELECT id, reference, error FROM jobs WHERE status='failed' ORDER BY id LIMIT 100"
                )
            ]
        if self.fatal_error:
            state = "failed"
            errors.insert(0, {"component": "coordinator", "error": self.fatal_error})
        elif self.is_terminal():
            state = "failed" if counts.get("failed", 0) else "complete"
        else:
            state = "running"
        for name in ("pending", "queued", "leased", "complete", "uploading", "uploaded", "failed"):
            counts.setdefault(name, 0)
        counts["total"] = total
        return {
            "state": state,
            "counts": counts,
            "producerDone": self.producer_done,
            "uptimeSeconds": round(time.time() - self.started_at, 3),
            "errors": errors,
        }


class CoordinatorHandler(BaseHTTPRequestHandler):
    server: "CoordinatorServer"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        _json_line("coordinator_http", client=self.client_address[0], message=format % args)

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.token}"
        return secrets.compare_digest(self.headers.get("Authorization", ""), expected)

    def _json(self, status: int, value: dict) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _reject(self, status: int, message: str) -> None:
        self.close_connection = True
        self._json(status, {"error": message})

    def do_GET(self) -> None:
        if not self._authorized():
            self._reject(HTTPStatus.UNAUTHORIZED, "unauthorized")
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self._json(HTTPStatus.OK, self.server.state.health())
            return
        pieces = parsed.path.strip("/").split("/")
        if len(pieces) == 2 and pieces[0] == "source" and pieces[1].isdigit():
            query = urllib.parse.parse_qs(parsed.query)
            lease = query.get("lease", [""])[0]
            try:
                path = self.server.state.source(int(pieces[1]), lease)
            except PermissionError as error:
                self._reject(HTTPStatus.CONFLICT, str(error))
                return
            size = path.stat().st_size
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with path.open("rb") as source:
                shutil.copyfileobj(source, self.wfile, 1024 * 1024)
            return
        self._reject(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if not self._authorized():
            self._reject(HTTPStatus.UNAUTHORIZED, "unauthorized")
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/lease":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            job = self.server.state.lease(str(body.get("workerId", "unknown")))
            if job is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header(
                    "X-Pipeline-Done", "1" if self.server.state.is_terminal() else "0"
                )
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self._json(HTTPStatus.OK, job)
            return
        pieces = parsed.path.strip("/").split("/")
        if len(pieces) != 2 or not pieces[1].isdigit():
            self._reject(HTTPStatus.NOT_FOUND, "not found")
            return
        query = urllib.parse.parse_qs(parsed.query)
        lease = query.get("lease", [""])[0]
        job_id = int(pieces[1])
        try:
            if pieces[0] == "result":
                length = int(self.headers.get("Content-Length", "0"))
                self.server.state.accept_result(job_id, lease, self.rfile, length)
                self._json(HTTPStatus.OK, {"accepted": job_id})
                return
            if pieces[0] == "failed":
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                self.server.state.fail_job(job_id, lease, str(body.get("error", "worker failed")))
                self._json(HTTPStatus.OK, {"failed": job_id})
                return
        except PermissionError as error:
            self._reject(HTTPStatus.CONFLICT, str(error))
            return
        except (ValueError, EOFError, tarfile.TarError, json.JSONDecodeError) as error:
            self._reject(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._reject(HTTPStatus.NOT_FOUND, "not found")


class CoordinatorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, state: CoordinatorState, token: str) -> None:
        super().__init__(address, CoordinatorHandler)
        self.state = state
        self.token = token


class CoordinatorClient:
    def __init__(self, base_url: str, token: str, worker_id: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.worker_id = worker_id

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | BinaryIO | None = None,
        length: int | None = None,
        timeout: float = 120,
    ) -> tuple[int, dict[str, str], bytes]:
        parsed = urllib.parse.urlparse(self.base_url)
        connection_class = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        connection = connection_class(parsed.netloc, timeout=timeout)
        base_path = parsed.path.rstrip("/")
        target = base_path + path
        if parsed.query:
            separator = "&" if "?" in target else "?"
            target += separator + parsed.query
        headers = {"Authorization": f"Bearer {self.token}"}
        if body is None:
            payload = None
        elif isinstance(body, bytes):
            payload = body
            headers["Content-Length"] = str(len(body))
        else:
            if length is None:
                raise ValueError("streaming request length is required")
            payload = body
            headers["Content-Length"] = str(length)
        connection.request(method, target, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        status = response.status
        connection.close()
        if status >= 400:
            raise RuntimeError(f"coordinator HTTP {status}: {data[:1000].decode(errors='replace')}")
        return status, response_headers, data

    def lease(self) -> tuple[dict | None, bool]:
        body = json.dumps({"workerId": self.worker_id}).encode()
        status, headers, data = self.request("POST", "/lease", body=body, timeout=60)
        if status == HTTPStatus.NO_CONTENT:
            return None, headers.get("x-pipeline-done") == "1"
        return json.loads(data), False

    def download(self, job: dict, target: Path) -> None:
        path = f"/source/{job['id']}?lease={urllib.parse.quote(job['leaseToken'])}"
        status, _, data = self.request("GET", path, timeout=300)
        if status != HTTPStatus.OK:
            raise RuntimeError(f"unexpected source response: {status}")
        target.write_bytes(data)
        if target.stat().st_size != int(job["sourceBytes"]) or _sha256(target) != job["sourceSha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError("downloaded SLP failed size/hash validation")

    def upload(self, job: dict, result: Path) -> None:
        path = f"/result/{job['id']}?lease={urllib.parse.quote(job['leaseToken'])}"
        with result.open("rb") as source:
            self.request(
                "POST",
                path,
                body=source,
                length=result.stat().st_size,
                timeout=900,
            )

    def fail(self, job: dict, error: str) -> None:
        path = f"/failed/{job['id']}?lease={urllib.parse.quote(job['leaseToken'])}"
        body = json.dumps({"error": error[-4000:]}).encode()
        self.request("POST", path, body=body, timeout=60)

    def heartbeat(self, job: dict) -> None:
        # HTTP leases predate the shared-volume transport and retain their fixed deadline.
        return


class SharedQueueClient:
    """Worker-side client for the coordinator-arbitrated Daytona volume queue.

    The volume is S3/FUSE-backed, so workers only publish unique claim intents. The
    coordinator's SQLite compare-and-swap is the exclusive claim and a worker renders only
    after observing its token-specific grant.
    """

    def __init__(self, root: Path, worker_id: str, worker_slot: int = 0) -> None:
        self.root = root
        self.worker_id = worker_id
        self.worker_slot = max(0, worker_slot)
        for name in (
            "sources",
            "pending",
            "claims",
            "grants",
            "leased",
            "results",
            "failures",
            "heartbeats",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stem(job_id: int, lease_token: str) -> str:
        return f"{job_id:06d}-{lease_token}"

    def lease(self) -> tuple[dict | None, bool]:
        pending = sorted((self.root / "pending").glob("*.json"))
        if pending:
            offset = self.worker_slot % len(pending)
            pending = pending[offset:] + pending[:offset]
        for pending_path in pending:
            try:
                advertised = json.loads(pending_path.read_text())
                job_id = int(advertised["id"])
                if pending_path != self.root / "pending" / f"{job_id:06d}.json":
                    continue
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            lease_token = secrets.token_urlsafe(24)
            stem = self._stem(job_id, lease_token)
            claim_path = self.root / "claims" / f"{stem}.json"
            decision_path = self.root / "grants" / f"{stem}.json"
            _atomic_json(
                claim_path,
                {
                    "id": job_id,
                    "leaseToken": lease_token,
                    "workerId": self.worker_id,
                    "claimedAt": time.time(),
                },
            )
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                try:
                    decision = json.loads(decision_path.read_text())
                except (OSError, ValueError, json.JSONDecodeError):
                    time.sleep(0.1)
                    continue
                if (
                    int(decision.get("id", -1)) != job_id
                    or decision.get("leaseToken") != lease_token
                ):
                    raise RuntimeError("coordinator returned a mismatched lease decision")
                if not decision.get("granted"):
                    claim_path.unlink(missing_ok=True)
                    decision_path.unlink(missing_ok=True)
                    break
                claim_path.unlink(missing_ok=True)
                self.heartbeat(decision)
                return decision, False
            else:
                raise TimeoutError(f"coordinator did not decide claim {stem}")
        return None, (self.root / "rendering.done").is_file()

    def download(self, job: dict, target: Path) -> None:
        expected = self.root / "sources" / f"{int(job['id']):06d}.slp"
        source = Path(str(job["sourcePath"]))
        if source != expected:
            raise ValueError("source path is outside the run-scoped queue")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{secrets.token_hex(4)}.partial")
        deadline = time.monotonic() + 120
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with source.open("rb") as input_file, temporary.open("wb") as output:
                    shutil.copyfileobj(input_file, output, 1024 * 1024)
                if (
                    temporary.stat().st_size != int(job["sourceBytes"])
                    or _sha256(temporary) != job["sourceSha256"]
                ):
                    raise RuntimeError("shared SLP failed size/hash validation")
                os.replace(temporary, target)
                return
            except (OSError, RuntimeError) as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                with contextlib.suppress(Exception):
                    self.heartbeat(job)
                time.sleep(1)
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"shared SLP did not become readable: {last_error}")

    def upload(self, job: dict, result: Path) -> None:
        job_id = int(job["id"])
        lease_token = str(job["leaseToken"])
        stem = self._stem(job_id, lease_token)
        target = self.root / "results" / f"{stem}.tar"
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        # This path is token-unique. Close the immutable payload before publishing its marker;
        # Daytona's volume does not implement rename.
        with result.open("rb") as source, target.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            with contextlib.suppress(OSError):
                os.fsync(output.fileno())
        _atomic_json(
            self.root / "results" / f"{stem}.result.json",
            {
                "id": job_id,
                "leaseToken": lease_token,
                "resultPath": str(target),
                "resultBytes": size,
                "resultSha256": digest.hexdigest(),
                "completedAt": time.time(),
            },
        )

    def fail(self, job: dict, error: str) -> None:
        job_id = int(job["id"])
        lease_token = str(job["leaseToken"])
        stem = self._stem(job_id, lease_token)
        _atomic_json(
            self.root / "failures" / f"{stem}.failure.json",
            {
                "id": job_id,
                "leaseToken": lease_token,
                "error": error[-4000:],
                "failedAt": time.time(),
            },
        )

    def heartbeat(self, job: dict) -> None:
        job_id = int(job["id"])
        lease_token = str(job["leaseToken"])
        stem = self._stem(job_id, lease_token)
        _atomic_json(
            self.root / "heartbeats" / f"{stem}.json",
            {
                "id": job_id,
                "leaseToken": lease_token,
                "workerId": self.worker_id,
                "heartbeatAt": time.time(),
            },
        )


def _worker_client(
    worker_id: str, worker_slot: int = 0
) -> CoordinatorClient | SharedQueueClient:
    queue_root = os.environ.get("SMASH_QUEUE_ROOT")
    if queue_root:
        return SharedQueueClient(Path(queue_root), worker_id, worker_slot)
    return CoordinatorClient(
        os.environ["SMASH_COORDINATOR_URL"],
        os.environ["SMASH_COORDINATOR_TOKEN"],
        worker_id,
    )


@dataclass(frozen=True)
class _SlpRenderPlan:
    raw_pos: int
    first_frame: int
    last_frame: int
    selected_segment: int = 1
    selected_chunks: tuple[tuple[int, int], ...] = ()


def _slp_render_plan(data: bytes, path: Path) -> _SlpRenderPlan:
    import struct

    if not data:
        raise ValueError(f"empty SLP: {path}")
    raw_pos = 0 if data[0] != ord("{") else 15
    raw_len = len(data) if raw_pos == 0 else int.from_bytes(data[raw_pos - 4 : raw_pos], "big")
    if raw_len <= 0 or raw_pos + raw_len > len(data):
        raw_len = len(data) - raw_pos
    raw_end = raw_pos + raw_len
    if raw_pos != 0 and (raw_pos + 2 > len(data) or data[raw_pos] != 0x35):
        raise ValueError("SLP is missing message-size table")
    sizes = {0x36: 0x140, 0x37: 0x6, 0x38: 0x46, 0x39: 0x1} if raw_pos == 0 else {}
    overall_first: int | None = None
    overall_last: int | None = None
    segment_first: int | None = None
    segment_last: int | None = None
    segment_start: int | None = None
    segment_table: tuple[int, int] | None = None
    last_table: tuple[int, int] | None = None
    game_started = False
    game_starts = 0
    first_complete: tuple[
        int, tuple[int, int] | None, int, int, int, int
    ] | None = None
    later_game_started = False
    position = raw_pos
    while position < raw_end:
        command = data[position]
        if command == 0x35:
            if position + 2 > raw_end:
                break
            payload_len = data[position + 1]
            if payload_len < 1:
                break
            stop = position + payload_len + 1
            if stop > raw_end:
                break
            refreshed = {0x35: payload_len}
            size_bytes = data[position + 2 : position + 1 + payload_len]
            for offset in range(0, len(size_bytes), 3):
                if offset + 2 >= len(size_bytes):
                    break
                refreshed[size_bytes[offset]] = int.from_bytes(
                    size_bytes[offset + 1 : offset + 3], "big"
                )
            sizes = refreshed
            last_table = (position, stop)
            position = stop
            continue
        size = sizes.get(command)
        if size is None:
            break
        stop = position + size + 1
        if stop > raw_end:
            break
        if command == 0x36:
            game_starts += 1
            if first_complete is not None:
                later_game_started = True
            game_started = True
            segment_start = position
            segment_table = last_table
            segment_first = None
            segment_last = None
        elif command == 0x38 and stop - position >= 5:
            frame = struct.unpack(">i", data[position + 1 : position + 5])[0]
            overall_first = frame if overall_first is None else min(overall_first, frame)
            overall_last = frame if overall_last is None else max(overall_last, frame)
            if game_started:
                segment_first = frame if segment_first is None else min(segment_first, frame)
                segment_last = frame if segment_last is None else max(segment_last, frame)
        elif command == 0x39:
            if (
                game_started
                and first_complete is None
                and segment_start is not None
                and segment_first is not None
                and segment_last is not None
            ):
                first_complete = (
                    game_starts,
                    segment_table,
                    segment_start,
                    stop,
                    segment_first,
                    segment_last,
                )
            game_started = False
        position = stop
    if overall_first is None or overall_last is None:
        raise ValueError(f"SLP has no post-frame updates: {path}")
    if first_complete is None:
        if game_starts > 1:
            raise ValueError(f"SLP has multiple games but no complete game: {path}")
        return _SlpRenderPlan(raw_pos, overall_first, overall_last)
    segment_number, table, game_start, game_end, first, last = first_complete
    if segment_number == 1 and not later_game_started:
        return _SlpRenderPlan(raw_pos, overall_first, overall_last)
    chunks: list[tuple[int, int]] = []
    if table is not None:
        chunks.append(table)
    chunks.append((game_start, game_end))
    return _SlpRenderPlan(raw_pos, first, last, segment_number, tuple(chunks))


def slp_frame_range(path: Path) -> tuple[int, int]:
    plan = _slp_render_plan(path.read_bytes(), path)
    return plan.first_frame, plan.last_frame


def _prepare_slp_render_input(
    source: Path, normalized: Path
) -> tuple[Path, tuple[int, int], dict | None]:
    data = source.read_bytes()
    plan = _slp_render_plan(data, source)
    frame_range = (plan.first_frame, plan.last_frame)
    if not plan.selected_chunks:
        return source, frame_range, None
    selected_raw = b"".join(data[start:stop] for start, stop in plan.selected_chunks)
    if plan.raw_pos:
        if not selected_raw or selected_raw[0] != 0x35:
            raise ValueError("normalized UBJSON SLP is missing its message-size table")
        prefix = bytearray(data[: plan.raw_pos])
        prefix[plan.raw_pos - 4 : plan.raw_pos] = len(selected_raw).to_bytes(4, "big")
        payload = bytes(prefix) + selected_raw + b"U\x08metadata{}}"
    else:
        payload = selected_raw
    normalized.write_bytes(payload)
    return (
        normalized,
        frame_range,
        {
            "policy": "first-complete-game-v1",
            "selectedSegment": plan.selected_segment,
            "firstFrame": plan.first_frame,
            "lastFrame": plan.last_frame,
            "renderInputBytes": len(payload),
            "renderInputSha256": hashlib.sha256(payload).hexdigest(),
        },
    )


def _annotate_replay_normalization(
    metadata: dict, job: dict, replay_normalization: dict | None
) -> None:
    if replay_normalization is None:
        return
    metadata["file"] = {
        "name": "input.slp",
        "bytes": int(job["sourceBytes"]),
        "sha256": str(job["sourceSha256"]),
    }
    metadata["replayNormalization"] = replay_normalization


def _probe_video(path: Path) -> dict:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
            "nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}")
    stream = streams[0]
    frames = stream.get("nb_read_frames")
    if not frames or frames == "N/A":
        raise RuntimeError(f"ffprobe could not count frames in {path}")
    stream["frames"] = int(frames)
    return stream


def _packet_pts_bounds(path: Path) -> tuple[float, float]:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    values = [
        float(line.rstrip(","))
        for line in completed.stdout.splitlines()
        if line.rstrip(",") not in {"", "N/A"}
    ]
    if not values:
        raise RuntimeError(f"ffprobe could not read packet PTS from {path}")
    return values[0], max(values)


def _validate_no_audio_and_zero_pts(path: Path) -> None:
    streams = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ]
    )
    types = [row.get("codec_type") for row in json.loads(streams.stdout).get("streams", [])]
    if types != ["video"]:
        raise RuntimeError(f"expected one video stream and no audio in {path}, got {types}")
    first_pts = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-read_intervals",
            "%+#1",
            "-of",
            "csv=p=0",
            str(path),
        ]
    ).stdout.strip()
    if first_pts not in {"0.000000", "0.000000,"}:
        raise RuntimeError(f"output PTS must start at zero, got {first_pts!r}")


def _allocated_cpus() -> int:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        quota, period = cpu_max.read_text().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    return max(1, os.cpu_count() or 1)


def render_job(
    job: dict,
    root: Path,
    llvm_threads: int,
    client: CoordinatorClient | SharedQueueClient,
) -> Path:
    job_dir = root / f"{int(job['id']):06d}"
    shutil.rmtree(job_dir, ignore_errors=True)
    render_dir = job_dir / "render"
    recording_dir = job_dir / "recording"
    render_dir.mkdir(parents=True)
    recording_dir.mkdir(parents=True)
    slp = job_dir / "input.slp"
    normalized_slp = job_dir / "render-input.slp"
    playback = job_dir / "playback.json"
    client.download(job, slp)
    render_slp, (source_first_frame, source_last_frame), replay_normalization = (
        _prepare_slp_render_input(slp, normalized_slp)
    )
    metadata_path = recording_dir / "metadata.json"
    _run(
        [
            "/opt/slippi-renderer/extract-slp-metadata.mjs",
            str(render_slp),
            str(metadata_path),
            job["reference"],
            "gdrive",
        ],
        timeout=120,
    )
    metadata = json.loads(metadata_path.read_text())
    _annotate_replay_normalization(metadata, job, replay_normalization)
    match_frames = ((metadata.get("match") or {}).get("frames") or {})
    first_playable_frame = int(match_frames.get("firstPlayable", -39))
    last_source_frame = int(match_frames.get("last", source_last_frame))
    render_start_frame = source_first_frame
    render_end_frame = source_last_frame + 1
    raw_first_slp_frame = render_start_frame + 1
    raw_last_slp_frame = render_end_frame - 1
    expected_raw_timeline_frames = raw_last_slp_frame - raw_first_slp_frame + 1
    if not raw_first_slp_frame <= first_playable_frame <= last_source_frame <= raw_last_slp_frame:
        raise RuntimeError(
            "invalid playable/render frame mapping: "
            f"raw={raw_first_slp_frame}..{raw_last_slp_frame}, "
            f"playable={first_playable_frame}..{last_source_frame}"
        )
    playback.write_text(
        json.dumps(
            {
                "replay": str(render_slp),
                "commandId": f"daytona-{job['id']}",
                "startFrame": render_start_frame,
                "endFrame": render_end_frame,
                "stopFrame": last_source_frame,
            },
            indent=2,
        )
        + "\n"
    )
    iso = Path(os.environ.get("SMASH_REMOTE_ISO", "/mnt/smash-assets/melee.iso"))
    if not iso.is_file():
        raise FileNotFoundError(iso)
    env = os.environ.copy()
    env.update(
        {
            "LP_NUM_THREADS": str(llvm_threads),
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "EGL_PLATFORM": "surfaceless",
            "MESA_GLTHREAD": "false",
            "SLIPPI_DOLPHIN_BIN": "/opt/slippi/dolphin-emu-nogui",
            "SLIPPI_DUMP_ONLY": "1",
            "SLIPPI_DUMP_FRAMES": "True",
            "SLIPPI_USE_FFV1": "False",
            "SLIPPI_DUMP_CODEC": "rawvideo",
            "SLIPPI_DUMP_FORMAT": "avi",
            "SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS": "True",
            "SLIPPI_EFB_SCALE": "0",
            "SLIPPI_CPU_THREAD": "False",
            "SLIPPI_RENDER_TO_MAIN": "False",
            "SLIPPI_RENDER_WIDTH": "204",
            "SLIPPI_RENDER_HEIGHT": "168",
            "SLIPPI_END_FRAME_POLL_SECONDS": "0.05",
            "SLIPPI_MAX_RAW_BYTES": str(
                int(expected_raw_timeline_frames * 224 * 184 * 3 / 2 * 1.10)
                + 64 * 1024 * 1024
            ),
        }
    )
    render_started = time.monotonic()
    _run(
        [
            "/tmp/render-ffv1-replay.sh",
            "--replay-json",
            str(playback),
            "--iso",
            str(iso),
            "--output-dir",
            str(render_dir),
            "--user-dir",
            str(job_dir / "dolphin-user"),
            "--timeout-seconds",
            os.environ.get("SMASH_RENDER_TIMEOUT_SECONDS", "1200"),
            "--video-backend",
            "OGL",
            "--cpu-core",
            "1",
            "--audio-backend",
            "Null",
            "--no-xvfb",
        ],
        timeout=int(os.environ.get("SMASH_RENDER_TIMEOUT_SECONDS", "1200")) + 30,
        env=env,
    )
    manifest_path = render_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"renderer did not write manifest.json for job {job['id']}")
    manifest = json.loads(manifest_path.read_text())
    rendered_last = (manifest.get("currentFrameRange") or {}).get("last")
    if rendered_last is None or int(rendered_last) < last_source_frame:
        raise RuntimeError(
            f"render stopped before replay end: {rendered_last} < {last_source_frame}"
        )
    if manifest.get("targetEndFrame") != render_end_frame:
        raise RuntimeError("renderer changed the strict capture endpoint")
    if manifest.get("targetStopFrame") != last_source_frame:
        raise RuntimeError("renderer did not use the reachable replay stop frame")
    sources = sorted(
        path
        for path in render_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if len(sources) != 1:
        raise RuntimeError(f"expected one raw video for job {job['id']}, found {len(sources)}")
    raw = sources[0]
    raw_probe = _probe_video(raw)
    if (int(raw_probe["width"]), int(raw_probe["height"])) != (224, 184):
        raise RuntimeError(
            f"unexpected raw resolution: {raw_probe['width']}x{raw_probe['height']}"
        )
    if int(raw_probe["frames"]) > expected_raw_timeline_frames:
        raise RuntimeError(
            "raw capture contains duplicate frames on its Slippi timeline: "
            f"{raw_probe['frames']} > {expected_raw_timeline_frames}"
        )
    raw_first_pts, raw_last_pts = _packet_pts_bounds(raw)
    expected_last_pts = (expected_raw_timeline_frames - 1) / SOURCE_FPS
    max_head_gap_pts = 1 / SOURCE_FPS + 0.001
    if raw_first_pts < -0.0001 or raw_first_pts > max_head_gap_pts:
        raise RuntimeError(f"raw capture has invalid first PTS: {raw_first_pts}")
    if abs(raw_last_pts - expected_last_pts) > 0.001:
        raise RuntimeError(
            "raw capture does not reach its final Slippi timeline PTS: "
            f"{raw_last_pts} != {expected_last_pts}"
        )
    # Held images are represented as AVI packet-duration gaps. The `fps=60` filter below
    # materializes the authoritative game timeline, so stored packet count can be smaller than
    # the strict playback interval even for a complete render.
    first_selected_index = first_playable_frame - raw_first_slp_frame
    semantic_last_index = last_source_frame - raw_first_slp_frame
    last_selected_index = first_selected_index + (
        (semantic_last_index - first_selected_index) // 3
    ) * 3
    last_selected_slp_frame = raw_first_slp_frame + last_selected_index
    cropped_tail_frames = last_source_frame - last_selected_slp_frame
    expected_frames = (last_selected_index - first_selected_index) // 3 + 1
    target = recording_dir / "video.mp4"
    temporary = recording_dir / "video.partial.mp4"
    select_expression = (
        f"between(n\\,{first_selected_index}\\,{last_selected_index})*"
        f"not(mod(n-{first_selected_index}\\,3))"
    )
    filter_graph = (
        f"fps=60:start_time=0,select='{select_expression}',"
        "setpts=N/(20*TB),"
        "scale=252:208:flags=lanczos,format=yuv420p"
    )
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw),
            "-vf",
            filter_graph,
            "-vsync",
            "cfr",
            "-r",
            str(VIDEO_FPS),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            os.environ.get("SMASH_H264_PRESET", "veryfast"),
            "-crf",
            os.environ.get("SMASH_H264_CRF", "18"),
            "-g",
            "40",
            "-bf",
            "2",
            "-threads",
            str(llvm_threads),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        timeout=600,
    )
    output_probe = _probe_video(temporary)
    expected = {
        "codec_name": "h264",
        "width": VIDEO_WIDTH,
        "height": VIDEO_HEIGHT,
        "pix_fmt": "yuv420p",
        "r_frame_rate": f"{VIDEO_FPS}/1",
        "avg_frame_rate": f"{VIDEO_FPS}/1",
        "frames": expected_frames,
    }
    for key, value in expected.items():
        actual = output_probe.get(key)
        if actual != value:
            raise RuntimeError(f"invalid output {key}: expected {value}, got {actual}")
    _validate_no_audio_and_zero_pts(temporary)
    temporary.replace(target)
    duration = float((metadata.get("match") or {}).get("duration", {}).get("seconds") or 0)
    result = {
        "file": target.name,
        "container": "mp4",
        "codec": "h264",
        "pixelFormat": output_probe["pix_fmt"],
        "frames": output_probe["frames"],
        "frameRate": output_probe["avg_frame_rate"],
        "width": output_probe["width"],
        "height": output_probe["height"],
        "inputBytes": raw.stat().st_size,
        "outputBytes": target.stat().st_size,
        "rawFrames": raw_probe["frames"],
        "expectedRawTimelineFrames": expected_raw_timeline_frames,
        "rawFirstPts": raw_first_pts,
        "rawLastPts": raw_last_pts,
        "sourceFps": SOURCE_FPS,
        "targetFps": VIDEO_FPS,
        "firstSelectedSlpFrame": first_playable_frame,
        "lastSourceSlpFrame": last_source_frame,
        "lastSelectedSlpFrame": last_selected_slp_frame,
        "sourceFrameStep": 3,
        "croppedTailSourceFrames": cropped_tail_frames,
        "strictCfrEndpointCompatible": cropped_tail_frames == 0,
        "renderSeconds": round(time.monotonic() - render_started, 3),
        "gameplaySeconds": duration,
        "realtimeFactor": round(duration / max(0.001, time.monotonic() - render_started), 4),
        "cpuOnly": True,
        "rendererSnapshot": os.environ.get("SMASH_RENDERER_SNAPSHOT", "unknown"),
    }
    metadata["video"] = result
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    bundle = job_dir / "result.tar"
    with tarfile.open(bundle, "w") as output:
        output.add(target, arcname="video.mp4", recursive=False)
        output.add(metadata_path, arcname="metadata.json", recursive=False)
    raw.unlink(missing_ok=True)
    shutil.rmtree(job_dir / "dolphin-user", ignore_errors=True)
    return bundle


def run_worker(worker_id: str, processes: int, result_batch_size: int) -> None:
    import queue

    os.environ["SMASH_WORKER_ID"] = worker_id
    root = Path(os.environ.get("SMASH_WORK_DIR", "/tmp/smash-worker")) / worker_id
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    cpu_count = _allocated_cpus()
    if processes < 1 or processes > cpu_count:
        raise ValueError(f"render processes must be between 1 and allocated CPUs ({cpu_count})")
    llvm_threads = max(1, cpu_count // processes)
    completed: "queue.Queue[tuple[dict, Path] | None]" = queue.Queue(
        maxsize=result_batch_size
    )
    stop = threading.Event()
    heartbeat_stop = threading.Event()
    errors: list[BaseException] = []
    active: dict[tuple[int, str], dict] = {}
    active_lock = threading.Lock()

    def remember(job: dict) -> None:
        with active_lock:
            active[(int(job["id"]), str(job["leaseToken"]))] = job

    def forget(job: dict) -> None:
        with active_lock:
            active.pop((int(job["id"]), str(job["leaseToken"])), None)

    def heartbeat() -> None:
        client = _worker_client(worker_id)
        while not heartbeat_stop.wait(30):
            with active_lock:
                jobs = list(active.values())
            for job in jobs:
                with contextlib.suppress(Exception):
                    client.heartbeat(job)

    def uploader() -> None:
        client = _worker_client(worker_id)
        failed = False
        while True:
            item = completed.get()
            try:
                if item is None:
                    return
                job, path = item
                if failed:
                    with contextlib.suppress(Exception):
                        client.fail(job, "worker result uploader stopped")
                    forget(job)
                    shutil.rmtree(path.parent, ignore_errors=True)
                    continue
                for attempt in range(1, 7):
                    try:
                        client.upload(job, path)
                        path.unlink(missing_ok=True)
                        shutil.rmtree(path.parent, ignore_errors=True)
                        forget(job)
                        _json_line("worker_uploaded", worker=worker_id, sample=job["id"])
                        break
                    except Exception:
                        if attempt == 6:
                            raise
                        time.sleep(min(30, 2**attempt))
            except BaseException as error:
                errors.append(error)
                stop.set()
                failed = True
                if item is not None:
                    job, path = item
                    with contextlib.suppress(Exception):
                        client.fail(job, f"result upload failed: {error}")
                    forget(job)
                    shutil.rmtree(path.parent, ignore_errors=True)
            finally:
                completed.task_done()

    def renderer(index: int) -> None:
        try:
            sandbox_index = int(worker_id.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            sandbox_index = 0
        client = _worker_client(
            f"{worker_id}-{index}",
            worker_slot=sandbox_index * processes + index,
        )
        lease_failures = 0
        while not stop.is_set():
            try:
                job, done = client.lease()
                lease_failures = 0
                if done:
                    return
                if job is None:
                    time.sleep(2)
                    continue
                remember(job)
                started = time.monotonic()
                try:
                    result = render_job(
                        job,
                        root / f"process-{index}",
                        llvm_threads,
                        client,
                    )
                    enqueued = False
                    while not stop.is_set():
                        try:
                            completed.put((job, result), timeout=1)
                            enqueued = True
                            break
                        except queue.Full:
                            continue
                    if not enqueued:
                        with contextlib.suppress(Exception):
                            client.fail(job, "worker result uploader stopped")
                        forget(job)
                        shutil.rmtree(result.parent, ignore_errors=True)
                        return
                    _json_line(
                        "worker_rendered",
                        worker=worker_id,
                        process=index,
                        sample=job["id"],
                        seconds=round(time.monotonic() - started, 3),
                    )
                except BaseException as error:
                    with contextlib.suppress(Exception):
                        client.fail(job, f"{type(error).__name__}: {error}")
                    forget(job)
                    shutil.rmtree(
                        root / f"process-{index}" / f"{int(job['id']):06d}",
                        ignore_errors=True,
                    )
                    _json_line(
                        "worker_failed",
                        worker=worker_id,
                        process=index,
                        sample=job["id"],
                        error=str(error),
                    )
            except BaseException as error:
                lease_failures += 1
                if lease_failures >= 6:
                    errors.append(error)
                    stop.set()
                    return
                time.sleep(min(30, 2**lease_failures))

    upload_thread = threading.Thread(target=uploader, name="result-uploader")
    heartbeat_thread = threading.Thread(target=heartbeat, name="lease-heartbeat")
    render_threads = [
        threading.Thread(target=renderer, args=(index,), name=f"renderer-{index}")
        for index in range(processes)
    ]
    upload_thread.start()
    heartbeat_thread.start()
    for thread in render_threads:
        thread.start()
    for thread in render_threads:
        thread.join()
    while True:
        try:
            completed.put(None, timeout=1)
            break
        except queue.Full:
            continue
    completed.join()
    upload_thread.join()
    heartbeat_stop.set()
    heartbeat_thread.join()
    if errors:
        raise RuntimeError(str(errors[0]))


def coordinator_main(args: argparse.Namespace) -> None:
    token = os.environ.get("SMASH_COORDINATOR_TOKEN")
    if not token:
        raise ValueError("SMASH_COORDINATOR_TOKEN is required")
    state = CoordinatorState(
        drive=DriveClient(args.remote, Path(args.config)),
        source_root=args.source_root,
        target_root=args.target_root,
        spool=Path(args.spool),
        sample_limit=args.sample_limit,
        prefetch=args.prefetch,
        lease_seconds=args.lease_seconds,
        max_attempts=args.max_attempts,
        upload_batch_size=args.upload_batch_size,
        upload_max_bytes=args.upload_max_bytes,
        upload_concurrency=args.upload_concurrency,
        upload_tps_limit=args.upload_tps_limit,
        upload_min_batch=args.upload_min_batch,
        upload_max_attempts=args.upload_max_attempts,
        spool_max_bytes=args.spool_max_bytes,
        queue_root=Path(args.queue_root) if args.queue_root else None,
    )
    state.start()
    server = CoordinatorServer(("0.0.0.0", args.port), state, token)
    _json_line("coordinator_ready", port=args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        state.stop.set()
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only Daytona frame-capture runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    coordinator = subparsers.add_parser("coordinator")
    coordinator.add_argument("--remote", required=True)
    coordinator.add_argument("--source-root", required=True)
    coordinator.add_argument("--target-root", required=True)
    coordinator.add_argument("--config", required=True)
    coordinator.add_argument("--spool", required=True)
    coordinator.add_argument("--port", type=int, default=8765)
    coordinator.add_argument("--sample-limit", type=int, default=0)
    coordinator.add_argument("--prefetch", type=int, default=512)
    coordinator.add_argument("--lease-seconds", type=int, default=300)
    coordinator.add_argument("--max-attempts", type=int, default=3)
    coordinator.add_argument("--upload-batch-size", type=int, default=100)
    coordinator.add_argument("--upload-concurrency", type=int, default=1)
    coordinator.add_argument("--upload-tps-limit", type=int, default=1)
    coordinator.add_argument("--upload-min-batch", type=int, default=64)
    coordinator.add_argument("--upload-max-attempts", type=int, default=3)
    coordinator.add_argument("--upload-max-bytes", type=int, default=2 * 1024**3)
    coordinator.add_argument(
        "--spool-max-bytes", type=int, default=DEFAULT_SPOOL_MAX_BYTES
    )
    coordinator.add_argument("--queue-root")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--worker-id", required=True)
    worker.add_argument("--processes", type=int, default=1)
    worker.add_argument("--result-batch-size", type=int, default=10)
    return parser


def runtime_main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "coordinator":
        coordinator_main(args)
    elif args.command == "worker":
        run_worker(args.worker_id, args.processes, args.result_batch_size)
    else:  # pragma: no cover - argparse enforces the choices.
        raise AssertionError(args.command)


if __name__ == "__main__":
    runtime_main()
