"""Fast CPU-only Daytona fleet for Slippi replay capture.

The coordinator streams each Drive source archive once into a bounded shared
volume.  Fixed worker slots eliminate distributed locks: job ``id % slots``
has exactly one owner.  One central scheduler assigns disjoint result batches
to fixed upload lanes; every lane uploads its manifest last as the commit
record.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import tempfile
import threading
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from . import replay_renderer
except ImportError:  # The same two files are executed directly inside Daytona.
    import replay_renderer  # type: ignore


SOURCE_ROOT = "hal-fox-captain-falcon-battlefield"
TARGET_ROOT = f"{SOURCE_ROOT}/recordings-642x528-20fps-slippi-pts-v4"
WORKER_SNAPSHOT = "smash-cpu-renderer-e7711b1-v3"
COORDINATOR_SNAPSHOT = "smash-cpu-renderer-e7711b1-v3-2cpu-repair"
ASSET_SNAPSHOT = "smash-cpu-renderer-e7711b1-1cpu-v1"
ASSET_VOLUME = "smash-frame-assets-v1"
MOUNT = "/mnt/smash-assets"
PREFETCH = 512
BATCH_SIZE = 100
BATCH_MIN_SIZE = 20
BATCH_BYTES = 1536 * 1024**2
UPLOAD_ATTEMPTS = 8
UPLOAD_RELAY_COUNT = 2
UPLOAD_CHUNK_SIZE = "64M"
UPLOAD_IDLE_TIMEOUT = "10m"
STATE_MISS_LIMIT = 6
HEALTH_MISS_LIMIT = 2
COORDINATOR_INIT_TIMEOUT = 1800


def _event(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      **fields}, separators=(",", ":"), sort_keys=True), flush=True)


def _run(command: list[str], *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if check and process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip() or shlex.join(command))
    return process


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_file(path: Path, size: int, digest: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if path.stat().st_size == size and _sha256(path) == digest:
                return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"shared file did not converge: {path}")
        time.sleep(.25)


def _copy_verified(source: Path, target: Path, size: int, digest: str) -> None:
    deadline = time.monotonic() + 120
    while True:
        try:
            shutil.copyfile(source, target)
            if target.stat().st_size == size and _sha256(target) == digest:
                return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"could not read complete shared file: {source}")
        time.sleep(.25)


def _publish_visible(source: Path, target: Path, size: int) -> None:
    """Close one object-store write, then wait for that object to become visible."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    shutil.copyfile(source, target)
    deadline = time.monotonic() + 300
    while True:
        try:
            if target.stat().st_size == size:
                return
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError(f"published file did not become visible: {target}")
        time.sleep(.25)


def _json(path: Path, value: dict | list, *, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path.with_suffix(path.suffix + ".partial") if atomic else path
    target.write_text(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")
    if atomic:
        target.replace(path)


def _read_json(path: Path, timeout: float = 30) -> dict | list:
    """Tolerate a shared-volume object becoming visible before its final byte."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(.1)


def _published_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _matching_shared_marker(marker: Path, job: dict, path_key: str, size_key: str) -> dict | None:
    """Return a marker only when its referenced shared object is fully visible."""
    row = _published_json(marker)
    if not row or row.get("id") != job["id"] or row.get("reference") != job["reference"]:
        return None
    try:
        path = Path(str(row[path_key]))
        expected = int(row[size_key])
        if expected <= 0 or path.stat().st_size != expected:
            return None
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return row


def _matching_result(marker: Path, job: dict) -> dict | None:
    row = _matching_shared_marker(marker, job, "resultPath", "resultBytes")
    if not row:
        return None
    try:
        if Path(str(row["sourcePath"])).stat().st_size != int(row["sourceBytes"]):
            return None
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return row


def _remote(name: str, path: str) -> str:
    return f"{name.rstrip(':')}:{path.strip('/')}"


def _cpus() -> int:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        quota, period = cpu_max.read_text().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class Sandbox:
    id: str
    name: str
    cpu: int
    memory: int
    disk: int
    gpu: int
    state: str

    @classmethod
    def parse(cls, row: dict) -> "Sandbox":
        return cls(row["id"], row["name"], int(row["cpu"]), int(row["memory"]),
                   int(row["disk"]), int(row.get("gpu", 0)), row["state"])


class DaytonaConnector:
    """Provision, supervise, and remove one CPU-only Daytona capture fleet."""

    def __init__(self) -> None:
        self.daytona = os.environ.get("SMASH_DAYTONA_BIN", shutil.which("daytona") or "daytona")
        self.config = Path(os.environ.get("SMASH_GDRIVE_CONFIG", "~/.config/rclone/rclone.conf")).expanduser()
        self.remote = os.environ.get("SMASH_GDRIVE_REMOTE", "smash-drive")
        self.source = os.environ.get("SMASH_GDRIVE_ROOT", SOURCE_ROOT).strip("/")
        self.target = os.environ.get("SMASH_GDRIVE_RECORDING_DIR", TARGET_ROOT).strip("/")
        self.worker_snapshot = os.environ.get("SMASH_DAYTONA_RENDERER_SNAPSHOT", WORKER_SNAPSHOT)
        self.coordinator_snapshot = os.environ.get("SMASH_DAYTONA_COORDINATOR_SNAPSHOT", COORDINATOR_SNAPSHOT)
        self.asset_snapshot = os.environ.get("SMASH_DAYTONA_ASSET_SNAPSHOT", ASSET_SNAPSHOT)
        self.volume = os.environ.get("SMASH_DAYTONA_ASSET_VOLUME", ASSET_VOLUME)
        self.iso = Path(os.environ.get("SMASH_MELEE_ISO", "/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso")).expanduser()
        self.processes = int(os.environ.get("SMASH_PROCESSES_PER_SANDBOX", "4"))
        self.default_workers = int(os.environ.get("SMASH_WORKER_COUNT", "23"))
        self.region = os.environ.get("SMASH_DAYTONA_REGION", "us")
        self.keep = os.environ.get("SMASH_KEEP_DAYTONA_RESOURCES") == "1"
        rclone_bin = os.environ.get("SMASH_RCLONE_BIN")
        self.rclone_bin = Path(rclone_bin).expanduser() if rclone_bin else None
        suffix = hashlib.sha256(os.urandom(16)).hexdigest()[:6]
        self.run_id = os.environ.get("SMASH_RUN_ID", time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + suffix)
        self.root = f"{MOUNT}/runs/{self.run_id}"
        self.owned: list[str] = []
        self._lock = threading.Lock()
        if not self.config.is_file() or not self.iso.is_file():
            raise FileNotFoundError(self.config if not self.config.is_file() else self.iso)
        if self.rclone_bin is not None and not self.rclone_bin.is_file():
            raise FileNotFoundError(self.rclone_bin)

    def _daytona(self, *args: str, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
        return _run([self.daytona, *args], timeout=timeout, check=check)

    def _exec(self, sandbox: Sandbox, command: list[str], timeout: int = 600,
              check: bool = True) -> subprocess.CompletedProcess[str]:
        return self._daytona("exec", sandbox.name, "--timeout", str(timeout), "--", "bash", "-lc",
                             shlex.quote(shlex.join(command)), timeout=timeout + 30, check=check)

    def _snapshots(self) -> dict[str, dict]:
        rows = json.loads(self._daytona("snapshot", "list", "--format", "json").stdout)
        snapshots = {row["name"]: row for row in rows}
        for name in (self.worker_snapshot, self.coordinator_snapshot, self.asset_snapshot):
            row = snapshots.get(name)
            if not row or row.get("state") != "active" or int(row.get("gpu", 0)) != 0:
                raise RuntimeError(f"active CPU-only Daytona snapshot required: {name}")
        if not 1 <= self.processes <= int(snapshots[self.worker_snapshot]["cpu"]):
            raise ValueError("render processes must fit the worker CPU allocation")
        return snapshots

    def _names(self) -> set[str]:
        page = json.loads(self._daytona("list", "--format", "json", "--limit", "100").stdout)
        return {row["name"] for row in page["items"] if not str(row["name"]).startswith("DESTROYED_")}

    def _create(self, name: str, snapshot: str, role: str, **labels: str) -> Sandbox:
        if name in self._names():
            raise RuntimeError(f"refusing to reuse Daytona sandbox {name}")
        command = ["create", "--name", name, "--snapshot", snapshot, "--target", self.region,
                   "--auto-stop", "720", "--auto-delete", "1440",
                   "--volume", f"{self.volume}:{MOUNT}"]
        for key, value in sorted({"smash-run-id": self.run_id, "smash-role": role, **labels}.items()):
            command += ["--label", f"{key}={value}"]
        self._daytona(*command)
        with self._lock:
            self.owned.append(name)
        sandbox = Sandbox.parse(json.loads(self._daytona("info", name, "--format", "json").stdout))
        if sandbox.gpu:
            self._delete(name)
            raise RuntimeError(f"GPU sandbox forbidden: {name}")
        print(json.dumps({"event": "sandbox_created", **asdict(sandbox)}), flush=True)
        return sandbox

    def _deploy(self, sandbox: Sandbox) -> None:
        root = Path(__file__).parents[3]
        paths = {
            "/tmp/daytona_connector.py": Path(__file__),
            "/tmp/replay_renderer.py": Path(__file__).with_name("replay_renderer.py"),
            "/tmp/render-ffv1-replay.sh": root / "experiments_dump/fast_replay_probe/render-ffv1-replay.sh",
        }
        payloads = {name: base64.b64encode(zlib.compress(path.read_bytes(), 9)).decode() for name, path in paths.items()}
        script = "import base64,pathlib,zlib,os; data=" + repr(payloads) + "; " \
                 "[(pathlib.Path(p).write_bytes(zlib.decompress(base64.b64decode(v)))) for p,v in data.items()]; " \
                 "os.chmod('/tmp/render-ffv1-replay.sh',0o755)"
        self._exec(sandbox, ["python3", "-c", script], timeout=120)

    def _upload(self, sandbox: Sandbox, local: Path) -> Path:
        shim = Path(__file__).parents[3] / "experiments_dump/fast_replay_probe/daytona_ssh_upload"
        environment = os.environ.copy()
        environment.update({"PATH": str(shim) + os.pathsep + environment.get("PATH", ""),
                            "DAYTONA_UPLOAD_EXTRA": str(local), "DAYTONA_UPLOAD_DEST": "/tmp/"})
        remote = Path("/tmp") / local.name
        process = subprocess.Popen([self.daytona, "ssh", sandbox.name, "--expires", "10"], env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        expected_size, expected_hash = local.stat().st_size, _sha256(local)
        deadline = time.monotonic() + 900
        try:
            while time.monotonic() < deadline:
                try:
                    size = self._exec(sandbox, ["stat", "-c%s", str(remote)], timeout=30).stdout.strip()
                    if size == str(expected_size):
                        observed = self._exec(sandbox, ["sha256sum", str(remote)], timeout=300).stdout.split()[0]
                        if observed == expected_hash:
                            return remote
                except Exception:
                    pass
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    raise RuntimeError(f"Daytona upload failed for {local}: {stderr.strip()}")
                time.sleep(1)
            raise TimeoutError(f"Daytona upload timed out: {local}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()

    def _delete(self, name: str) -> None:
        result = self._daytona("delete", name, timeout=300, check=False)
        if result.returncode and "not found" not in result.stderr.lower():
            raise RuntimeError(result.stderr.strip())
        with self._lock, contextlib.suppress(ValueError):
            self.owned.remove(name)

    def _prepare_assets(self) -> None:
        volumes = json.loads(self._daytona("volume", "list", "--format", "json").stdout)
        if not any(row["name"] == self.volume for row in volumes):
            self._daytona("volume", "create", self.volume, "--size", "10")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            volume = json.loads(self._daytona("volume", "get", self.volume, "--format", "json").stdout)
            if volume["state"] == "ready":
                break
            if volume["state"] == "error":
                raise RuntimeError(str(volume.get("errorReason")))
            time.sleep(2)
        else:
            raise TimeoutError(f"Daytona volume not ready: {self.volume}")
        seed = self._create(f"smash-assets-{self.run_id}", self.asset_snapshot, "asset-seed")
        try:
            expected = _sha256(self.iso)
            marker = self._exec(seed, ["cat", f"{MOUNT}/melee.iso.sha256"], check=False).stdout.strip()
            size = self._exec(seed, ["stat", "-c%s", f"{MOUNT}/melee.iso"], check=False).stdout.strip()
            existing = expected if marker == expected and size == str(self.iso.stat().st_size) else ""
            if not existing:
                fields = self._exec(seed, ["sha256sum", f"{MOUNT}/melee.iso"], check=False).stdout.split()
                existing = fields[0] if fields else ""
            if existing != expected:
                with tempfile.TemporaryDirectory() as temporary:
                    link = Path(temporary) / "melee.iso"
                    link.symlink_to(self.iso)
                    uploaded = self._upload(seed, link)
                    self._exec(seed, ["cp", str(uploaded), f"{MOUNT}/melee.iso"], timeout=900)
                observed = self._exec(seed, ["sha256sum", f"{MOUNT}/melee.iso"], timeout=300).stdout.split()[0]
            else:
                observed = existing
            if observed != expected:
                raise RuntimeError("shared ISO checksum mismatch")
            self._exec(seed, ["bash", "-lc", f"printf '%s\\n' {expected} > {MOUNT}/melee.iso.sha256"])
        finally:
            self._delete(seed.name)

    def _coordinator(self, sample_limit: int) -> Sandbox:
        sandbox = self._create(f"smash-coord-{self.run_id}", self.coordinator_snapshot, "coordinator")
        self._deploy(sandbox)
        self._install_rclone(sandbox)
        command = ["python3", "/tmp/daytona_connector.py", "coordinator", "--remote", self.remote,
                   "--source", self.source, "--target", self.target, "--root", self.root,
                   "--config", "/home/daytona/.config/rclone/rclone.conf", "--sample-limit", str(sample_limit),
                   "--state", "/home/daytona/smash-coordinator/state.json"]
        shell = shlex.join(command)
        local_log = "/home/daytona/smash-coordinator/run.log"
        shared_log = f"{self.root}/diagnostics/logs/{sandbox.name}.log"
        wrapper = (
            f"mkdir -p /home/daytona/smash-coordinator {shlex.quote(str(Path(shared_log).parent))}; "
            f": >{shlex.quote(local_log)}; {shell} >{shlex.quote(local_log)} 2>&1; status=$?; "
            f"cp -- {shlex.quote(local_log)} {shlex.quote(shared_log)} 2>/dev/null || true; exit $status"
        )
        self._exec(sandbox, ["bash", "-lc",
                             "mkdir -p /home/daytona/smash-coordinator; "
                             f"nohup bash -lc {shlex.quote(wrapper)} >/dev/null 2>&1 & "
                             "echo $! >/home/daytona/smash-coordinator/run.pid"], timeout=30)
        return sandbox

    def _install_rclone(self, sandbox: Sandbox) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "rclone.conf"
            config.symlink_to(self.config)
            uploaded = self._upload(sandbox, config)
            self._exec(sandbox, ["bash", "-lc", f"mkdir -p /home/daytona/.config/rclone; install -m600 {shlex.quote(str(uploaded))} /home/daytona/.config/rclone/rclone.conf"])
        if self.rclone_bin is not None:
            uploaded_bin = self._upload(sandbox, self.rclone_bin)
            self._exec(sandbox, ["install", "-m755", str(uploaded_bin), "/usr/local/bin/rclone"])

    def _state(self, coordinator: Sandbox) -> dict | None:
        try:
            result = self._exec(coordinator, ["cat", "/home/daytona/smash-coordinator/state.json"],
                                timeout=15, check=False)
        except Exception as error:
            _event("supervisor_state_read_failed", error=f"{type(error).__name__}: {error}")
            return None
        if result.returncode:
            detail = (result.stderr.strip() or result.stdout.strip() or "state read failed")[-2000:]
            _event("supervisor_state_read_failed", returnCode=result.returncode, error=detail)
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            _event("supervisor_state_read_failed", error="invalid coordinator state JSON")
            return None

    def _snapshot_logs(self, rows: list[tuple[Sandbox, str]], *, processes: bool = False) -> list[str]:
        """Copy local sandbox logs to the durable run root before a sandbox is removed."""
        if not rows:
            return []
        errors: list[str] = []

        def snapshot(sandbox: Sandbox, source: str) -> None:
            log_root = f"{self.root}/diagnostics/logs"
            destination = f"{log_root}/{sandbox.name}.log"
            commands = [
                f"mkdir -p {shlex.quote(log_root)}",
                f"if test -f {shlex.quote(source)}; then cp -- {shlex.quote(source)} {shlex.quote(destination)}; fi",
            ]
            if processes:
                process_log = f"/tmp/{sandbox.name}.processes.txt"
                process_destination = f"{log_root}/{sandbox.name}.processes.txt"
                commands += [f"ps auxww >{shlex.quote(process_log)}",
                             f"cp -- {shlex.quote(process_log)} {shlex.quote(process_destination)}"]
            result = self._exec(sandbox, ["bash", "-lc", "; ".join(commands)], timeout=120, check=False)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "log snapshot failed")

        with ThreadPoolExecutor(max_workers=min(8, len(rows))) as pool:
            futures = {pool.submit(snapshot, sandbox, source): sandbox.name for sandbox, source in rows}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{futures[future]}: {error}")
        return errors

    def _write_supervisor_diagnostic(self, coordinator: Sandbox, error: BaseException) -> None:
        payload = {
            "event": "supervisor_failure",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runId": self.run_id,
            "error": f"{type(error).__name__}: {error}",
            "traceback": "".join(traceback.format_exception(error)),
        }
        path = f"{self.root}/diagnostics/supervisor-failure.json"
        script = (
            "import pathlib; "
            f"p=pathlib.Path({path!r}); p.parent.mkdir(parents=True,exist_ok=True); "
            f"p.write_text({(json.dumps(payload, separators=(',', ':'), sort_keys=True) + chr(10))!r})"
        )
        self._exec(coordinator, ["python3", "-c", script], timeout=120)
        self._exec(coordinator, ["bash", "-lc",
                                 f"if test -f /home/daytona/smash-coordinator/state.json; then "
                                 f"cp -- /home/daytona/smash-coordinator/state.json "
                                 f"{shlex.quote(self.root + '/diagnostics/coordinator-state.json')}; fi"],
                   timeout=120, check=False)

    def _worker(self, index: int, generation: int, workers: int) -> Sandbox:
        suffix = f"{index:03d}" + (f"-r{generation}" if generation else "")
        sandbox = self._create(f"smash-worker-{self.run_id}-{suffix}", self.worker_snapshot, "worker",
                               **{"smash-worker-index": str(index)})
        self._deploy(sandbox)
        command = shlex.join(["python3", "/tmp/daytona_connector.py", "worker", "--root", self.root,
                              "--index", str(index), "--workers", str(workers), "--processes", str(self.processes),
                              "--snapshot", self.worker_snapshot])
        local_log = "/tmp/smash-worker.log"
        shared_log = f"{self.root}/diagnostics/logs/{sandbox.name}.log"
        supervisor = (
            f"mkdir -p {shlex.quote(str(Path(shared_log).parent))}; : >{local_log}; "
            f"while true; do {command} >>{local_log} 2>&1; s=$?; "
            f"cp -- {local_log} {shlex.quote(shared_log)} 2>/dev/null || true; "
            "[ $s -eq 0 ] && exit 0; sleep 5; done"
        )
        self._exec(sandbox, ["bash", "-lc", f"nohup bash -lc {shlex.quote(supervisor)} >/dev/null 2>&1 & "
                             "echo $! >/tmp/smash-worker.pid"], timeout=30)
        return sandbox

    def _uploader(self, index: int, generation: int) -> Sandbox:
        lane = index + 1  # Lane zero runs inside the coordinator sandbox.
        suffix = f"{index:02d}" + (f"-r{generation}" if generation else "")
        sandbox = self._create(f"smash-uploader-{self.run_id}-{suffix}", self.asset_snapshot, "uploader",
                               **{"smash-upload-lane": str(lane)})
        self._deploy(sandbox)
        self._install_rclone(sandbox)
        command = shlex.join(["python3", "/tmp/daytona_connector.py", "uploader", "--root", self.root,
                              "--lane", str(lane), "--remote", self.remote, "--target", self.target,
                              "--config", "/home/daytona/.config/rclone/rclone.conf"])
        local_log = "/tmp/smash-uploader.log"
        shared_log = f"{self.root}/diagnostics/logs/{sandbox.name}.log"
        supervisor = (
            f"mkdir -p {shlex.quote(str(Path(shared_log).parent))}; : >{local_log}; "
            f"while true; do {command} >>{local_log} 2>&1; s=$?; "
            f"cp -- {local_log} {shlex.quote(shared_log)} 2>/dev/null || true; "
            "[ $s -eq 0 ] && exit 0; sleep 5; done"
        )
        self._exec(sandbox, ["bash", "-lc", f"nohup bash -lc {shlex.quote(supervisor)} >/dev/null 2>&1 & "
                             "echo $! >/tmp/smash-uploader.pid"], timeout=30)
        return sandbox

    def _launch_workers(self, count: int) -> list[tuple[int, int, Sandbox]]:
        rows: list[tuple[int, int, Sandbox]] = []
        with ThreadPoolExecutor(max_workers=min(12, count)) as pool:
            futures = {pool.submit(self._worker, index, 0, count): index for index in range(count)}
            for future in as_completed(futures):
                index = futures[future]
                rows.append((index, 0, future.result()))
        return sorted(rows)

    def _launch_uploaders(self, count: int) -> list[tuple[int, int, Sandbox]]:
        rows: list[tuple[int, int, Sandbox]] = []
        if not count:
            return rows
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = {pool.submit(self._uploader, index, 0): index for index in range(count)}
            for future in as_completed(futures):
                index = futures[future]
                rows.append((index, 0, future.result()))
        return sorted(rows)

    def run(self, sample_limit: int = 0, worker_count: int = 0) -> dict:
        coordinator: Sandbox | None = None
        workers: list[tuple[int, int, Sandbox]] = []
        uploaders: list[tuple[int, int, Sandbox]] = []
        started = time.monotonic()
        primary_error: BaseException | None = None
        try:
            self._snapshots()
            self._prepare_assets()
            coordinator = self._coordinator(sample_limit)
            deadline = time.monotonic() + COORDINATOR_INIT_TIMEOUT
            state = None
            while time.monotonic() < deadline:
                with contextlib.suppress(Exception):
                    state = self._state(coordinator)
                if state and state["state"] in {"ready", "complete", "failed"}:
                    break
                time.sleep(2)
            if not state:
                raise TimeoutError("Daytona coordinator did not initialize")
            if state["state"] == "failed":
                raise RuntimeError(state.get("error", "coordinator failed"))
            remaining = int(state["remaining"])
            if not remaining:
                return state
            count = worker_count or min(self.default_workers, math.ceil(remaining / self.processes))
            if count < 1:
                raise ValueError("at least one worker is required")
            uploader_count = int(os.environ.get("SMASH_UPLOAD_RELAY_COUNT", str(UPLOAD_RELAY_COUNT)))
            if uploader_count < 0:
                raise ValueError("upload relay count cannot be negative")
            uploaders = self._launch_uploaders(uploader_count)
            fleet = {"workers": count, "processes": self.processes, "uploaders": uploader_count}
            self._exec(coordinator, ["python3", "-c",
                                     f"import json,pathlib; pathlib.Path({self.root!r}+'/fleet.json').write_text(json.dumps({fleet!r}))"])
            workers = self._launch_workers(count)
            last_check = time.monotonic()
            state_misses = 0
            coordinator_health_misses = 0
            worker_health_misses: dict[int, int] = {}
            uploader_health_misses: dict[int, int] = {}
            while True:
                state = self._state(coordinator)
                if state is None:
                    state_misses += 1
                    if state_misses >= STATE_MISS_LIMIT:
                        raise RuntimeError(f"coordinator state unavailable for {state_misses} consecutive polls")
                    time.sleep(15)
                    continue
                state_misses = 0
                if state and state["state"] in {"complete", "failed"}:
                    if state["state"] == "failed":
                        raise RuntimeError(state.get("error", "Daytona pipeline failed"))
                    return {**state, "seconds": round(time.monotonic() - started, 3),
                            "workers": count, "processesPerWorker": self.processes,
                            "target": _remote(self.remote, self.target)}
                if time.monotonic() - last_check >= 120:
                    snapshot_errors = self._snapshot_logs(
                        [(coordinator, "/home/daytona/smash-coordinator/run.log"),
                         *[(sandbox, "/tmp/smash-uploader.log") for _, _, sandbox in uploaders]]
                    )
                    if snapshot_errors:
                        print(json.dumps({"event": "diagnostic_snapshot_failed",
                                          "errors": snapshot_errors}), flush=True)
                    coordinator_alive = self._exec(
                        coordinator,
                        ["bash", "-lc", "kill -0 $(cat /home/daytona/smash-coordinator/run.pid)"],
                        timeout=30,
                        check=False,
                    )
                    if coordinator_alive.returncode:
                        latest = self._state(coordinator)
                        if latest and latest["state"] in {"complete", "failed"}:
                            state = latest
                            continue
                        coordinator_health_misses += 1
                        _event("supervisor_health_check_failed", role="coordinator",
                               consecutive=coordinator_health_misses,
                               returnCode=coordinator_alive.returncode)
                        if coordinator_health_misses >= HEALTH_MISS_LIMIT:
                            raise RuntimeError("Daytona coordinator process exited before completion")
                    else:
                        coordinator_health_misses = 0
                    for position, (index, generation, sandbox) in enumerate(workers):
                        alive = self._exec(sandbox, ["bash", "-lc", "kill -0 $(cat /tmp/smash-worker.pid)"], timeout=30, check=False)
                        slot_files = " && ".join(
                            f"test -f {self.root}/workers/{slot:04d}.done"
                            for slot in range(index * self.processes, (index + 1) * self.processes)
                        )
                        complete = self._exec(sandbox, ["bash", "-lc", slot_files], timeout=30, check=False)
                        if alive.returncode and complete.returncode:
                            worker_health_misses[index] = worker_health_misses.get(index, 0) + 1
                            if worker_health_misses[index] < HEALTH_MISS_LIMIT:
                                continue
                            self._delete(sandbox.name)
                            replacement = self._worker(index, generation + 1, count)
                            workers[position] = (index, generation + 1, replacement)
                            worker_health_misses[index] = 0
                        else:
                            worker_health_misses[index] = 0
                    for position, (index, generation, sandbox) in enumerate(uploaders):
                        alive = self._exec(sandbox, ["bash", "-lc", "kill -0 $(cat /tmp/smash-uploader.pid)"],
                                           timeout=30, check=False)
                        if alive.returncode:
                            uploader_health_misses[index] = uploader_health_misses.get(index, 0) + 1
                            if uploader_health_misses[index] < HEALTH_MISS_LIMIT:
                                continue
                            self._delete(sandbox.name)
                            replacement = self._uploader(index, generation + 1)
                            uploaders[position] = (index, generation + 1, replacement)
                            uploader_health_misses[index] = 0
                        else:
                            uploader_health_misses[index] = 0
                    last_check = time.monotonic()
                time.sleep(15)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if not self.keep:
                cleanup_errors: list[str] = []
                if coordinator is not None:
                    with contextlib.suppress(Exception):
                        self._exec(coordinator, ["touch", f"{self.root}/stop"], timeout=30)
                if primary_error is not None and coordinator is not None:
                    try:
                        self._write_supervisor_diagnostic(coordinator, primary_error)
                    except Exception as error:
                        cleanup_errors.append(f"supervisor diagnostic: {error}")
                    log_rows = [(coordinator, "/home/daytona/smash-coordinator/run.log")]
                    log_rows += [(sandbox, "/tmp/smash-worker.log") for _, _, sandbox in workers]
                    log_rows += [(sandbox, "/tmp/smash-uploader.log") for _, _, sandbox in uploaders]
                    cleanup_errors += [f"diagnostic log: {error}"
                                       for error in self._snapshot_logs(log_rows, processes=True)]
                worker_names = [name for name in self.owned if coordinator is None or name != coordinator.name]
                with ThreadPoolExecutor(max_workers=min(12, max(1, len(worker_names)))) as pool:
                    futures = {pool.submit(self._delete, name): name for name in worker_names}
                    for future, name in ((future, futures[future]) for future in as_completed(futures)):
                        try:
                            future.result()
                        except Exception as error:
                            cleanup_errors.append(f"{name}: {error}")
                if coordinator is not None:
                    if not cleanup_errors and primary_error is None:
                        try:
                            self._exec(coordinator, ["rm", "-rf", "--", self.root], timeout=300)
                        except Exception as error:
                            cleanup_errors.append(f"queue: {error}")
                    elif primary_error is not None:
                        print(json.dumps({"event": "queue_preserved", "root": self.root}), flush=True)
                    try:
                        self._delete(coordinator.name)
                    except Exception as error:
                        cleanup_errors.append(f"{coordinator.name}: {error}")
                if cleanup_errors:
                    message = "Daytona cleanup incomplete: " + "; ".join(cleanup_errors)
                    if primary_error is None:
                        raise RuntimeError(message)
                    with contextlib.suppress(AttributeError):
                        primary_error.add_note(message)


class Drive:
    def __init__(self, remote: str, config: Path) -> None:
        self.remote, self.config = remote, config

    def command(self, *args: str, timeout: int = 900) -> str:
        command = ["rclone", *args, "--config", str(self.config), "--retries", "10",
                   "--low-level-retries", "20"]
        if "--tpslimit" not in args:
            command += ["--tpslimit", "8"]
        return _run(command, timeout=timeout).stdout

    def files(self, root: str) -> list[str]:
        try:
            output = self.command("lsf", _remote(self.remote, root), "-R", "--files-only", "--format", "p")
        except RuntimeError as error:
            if "not found" in str(error).lower() or "directory not found" in str(error).lower():
                return []
            raise
        return sorted(output.splitlines())

    def file_sizes(self, root: str) -> dict[str, int]:
        try:
            output = self.command("lsf", _remote(self.remote, root), "-R", "--files-only",
                                  "--format", "ps", "--separator", "\t")
        except RuntimeError as error:
            if "not found" in str(error).lower() or "directory not found" in str(error).lower():
                return {}
            raise
        return {path: int(size) for path, size in (line.rsplit("\t", 1) for line in output.splitlines())}

    def cat(self, path: str) -> bytes:
        process = subprocess.run(["rclone", "cat", _remote(self.remote, path), "--config", str(self.config),
                                  "--retries", "10", "--low-level-retries", "20", "--tpslimit", "8"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode:
            raise RuntimeError(process.stderr.decode(errors="replace"))
        return process.stdout


def _jobs(drive: Drive, source: str, target: str, limit: int) -> tuple[list[dict], set[str]]:
    source_files = drive.files(source)
    jobs = []
    for index_path in (path for path in source_files if path.endswith(".tar.zst.slp-index.jsonl")):
        archive = index_path.removesuffix(".slp-index.jsonl")
        for line in drive.cat(f"{source}/{index_path}").splitlines():
            row = json.loads(line)
            if str(row["member"]).lower().endswith(".slp"):
                jobs.append({"reference": f"{archive}::{row['member']}", "archive": archive, "member": row["member"]})
    jobs.sort(key=lambda row: row["reference"])
    jobs = jobs[:limit] if limit > 0 else jobs
    if not jobs:
        raise RuntimeError(f"no indexed SLPs found under {_remote(drive.remote, source)}")
    for job_id, job in enumerate(jobs):
        job["id"] = job_id
    target_files = drive.file_sizes(f"{target}/batches")
    uploaded: set[str] = set()
    manifests = {path: size for path, size in target_files.items()
                 if path.endswith(".manifest.jsonl") and size > 0}
    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary)
        if manifests:
            drive.command("copy", _remote(drive.remote, f"{target}/batches"), str(local),
                          "--include", "*.manifest.jsonl", "--transfers", "8")
        for manifest, size in manifests.items():
            archive = manifest.removesuffix(".manifest.jsonl") + ".tar.zst"
            path = local / manifest
            if target_files.get(archive, 0) <= 0 or not path.is_file():
                continue
            if path.stat().st_size != size:
                raise RuntimeError(f"incomplete committed manifest: {manifest}")
            lines = path.read_bytes().splitlines()
            if not lines:
                raise RuntimeError(f"empty committed manifest: {manifest}")
            for line in lines:
                uploaded.add(json.loads(line)["sourceReference"])
    uploaded &= {job["reference"] for job in jobs}
    return jobs, uploaded


def _stream_sources(drive: Drive, source_root: str, root: Path, jobs: list[dict], errors: list[str]) -> None:
    try:
        result_markers = {int(path.stem): path for path in (root / "results").glob("*.json")
                          if path.stem.isdigit()}
        ready_markers = {int(path.stem): path for path in (root / "ready").glob("*.json")
                         if path.stem.isdigit()}
        by_archive: dict[str, dict[str, dict]] = {}
        for job in jobs:
            job_id = int(job["id"])
            result_marker = result_markers.get(job_id)
            if result_marker and _matching_result(result_marker, job):
                continue
            ready_marker = ready_markers.get(job_id)
            if ready_marker and _matching_shared_marker(ready_marker, job, "sourcePath", "sourceBytes"):
                continue
            by_archive.setdefault(job["archive"], {})[job["member"].removeprefix("./")] = job
        for archive, wanted in sorted(by_archive.items()):
            for attempt in range(5):
                failure: BaseException | None = None
                detail = ""
                with tempfile.TemporaryFile() as process_log:
                    rclone = subprocess.Popen(["rclone", "cat", _remote(drive.remote, f"{source_root}/{archive}"),
                                               "--config", str(drive.config), "--retries", "10", "--low-level-retries", "20",
                                               "--tpslimit", "8"], stdout=subprocess.PIPE, stderr=process_log)
                    assert rclone.stdout
                    zstd = subprocess.Popen(["zstd", "-dc"], stdin=rclone.stdout, stdout=subprocess.PIPE, stderr=process_log)
                    rclone.stdout.close()
                    assert zstd.stdout
                    try:
                        with tarfile.open(fileobj=zstd.stdout, mode="r|") as stream:
                            for member in stream:
                                key = member.name.removeprefix("./")
                                job = wanted.get(key)
                                if not job or not member.isfile():
                                    continue
                                while len(list((root / "ready").glob("*.json"))) >= PREFETCH:
                                    if (root / "stop").exists():
                                        return
                                    time.sleep(1)
                                extracted = stream.extractfile(member)
                                if extracted is None:
                                    raise RuntimeError(f"cannot extract {member.name}")
                                path = root / "sources" / f"{job['id']:06d}.slp"
                                path.unlink(missing_ok=True)
                                digest, size = hashlib.sha256(), 0
                                with path.open("wb") as output:
                                    while chunk := extracted.read(1024 * 1024):
                                        output.write(chunk); digest.update(chunk); size += len(chunk)
                                _json(root / "ready" / f"{job['id']:06d}.json",
                                      {**job, "sourcePath": str(path), "sourceBytes": size,
                                       "sourceSha256": digest.hexdigest()})
                                wanted.pop(key, None)
                        zstd_status, rclone_status = zstd.wait(), rclone.wait()
                        if zstd_status or rclone_status or wanted:
                            raise RuntimeError(f"stream statuses zstd={zstd_status}, rclone={rclone_status}; "
                                               f"missing source members: {list(wanted)[:3]}")
                    except BaseException as error:
                        failure = error
                        zstd.kill(); rclone.kill(); zstd.wait(); rclone.wait()
                    process_log.seek(0)
                    detail = process_log.read().decode(errors="replace").strip()
                if failure is None or not wanted:
                    break
                if attempt == 4:
                    raise RuntimeError(
                        f"source archive failed after {attempt + 1} attempts: {archive}: {failure}; {detail}"
                    )
                time.sleep(2**attempt)
        (root / "producer.done").touch()
    except BaseException as error:
        errors.append(f"source producer: {type(error).__name__}: {error}")
        (root / "stop").touch()


def _batch_archive(rows: list[dict], key: str) -> Path:
    """Build one seekable local archive so Drive retries never rebuild the stream."""
    started = time.monotonic()
    input_bytes = sum(int(row["sourceBytes"]) + int(row["resultBytes"]) for row in rows)
    _event("archive_build_started", batch=key, count=len(rows), inputBytes=input_bytes)
    upload_root = Path(tempfile.gettempdir()) / "smash-upload"
    upload_root.mkdir(parents=True, exist_ok=True)
    archive_path = upload_root / f"batch-{key}.tar.zst"
    partial = archive_path.with_suffix(archive_path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with tempfile.TemporaryFile() as process_log:
        archive_output = partial.open("wb")
        zstd = subprocess.Popen(["zstd", "-T1", "-1", "-c"], stdin=subprocess.PIPE,
                                stdout=archive_output, stderr=process_log)
        assert zstd.stdin
        try:
            with tarfile.open(fileobj=zstd.stdin, mode="w|") as output:
                for row in rows:
                    prefix = f"slp_with_video/{row['id']}"
                    output.add(row["sourcePath"], arcname=f"{prefix}/input.slp", recursive=False)
                    with tarfile.open(row["resultPath"]) as result:
                        for name in (["metadata.json"] if row.get("skipReason") else ["video.mp4", "metadata.json"]):
                            member = result.getmember(name)
                            extracted = result.extractfile(member)
                            if extracted is None:
                                raise RuntimeError(f"missing {name} in result {row['id']}")
                            member.name = f"{prefix}/{name}"
                            output.addfile(member, extracted)
            zstd.stdin.close()
            zstd_status = zstd.wait()
        except BaseException:
            zstd.kill(); zstd.wait()
            raise
        finally:
            archive_output.close()
        process_log.seek(0)
        detail = process_log.read().decode(errors="replace")
        if zstd_status:
            partial.unlink(missing_ok=True)
            raise RuntimeError(detail)
    partial.replace(archive_path)
    _event("archive_build_completed", batch=key, count=len(rows), inputBytes=input_bytes,
           archiveBytes=archive_path.stat().st_size, seconds=round(time.monotonic() - started, 3))
    return archive_path


def _upload_batch(drive: Drive, root: Path, target: str, markers: list[Path], lane: int) -> None:
    rows = [_read_json(path) for path in markers]
    rows.sort(key=lambda row: row["id"])
    preflight_started = time.monotonic()
    _event("upload_preflight_started", lane=lane, count=len(rows),
           bytes=sum(int(row["sourceBytes"]) + int(row["resultBytes"]) for row in rows))
    for row in rows:
        _wait_file(Path(row["resultPath"]), int(row["resultBytes"]), row["resultSha256"])
        _wait_file(Path(row["sourcePath"]), int(row["sourceBytes"]), row["sourceSha256"])
    _event("upload_preflight_completed", lane=lane, count=len(rows),
           seconds=round(time.monotonic() - preflight_started, 3))
    key = hashlib.sha256("\n".join(row["reference"] for row in rows).encode()).hexdigest()[:20]
    manifest_path = root / f"batch-{key}.manifest.jsonl"
    archive_relative = f"{target}/batches/batch-{key}.tar.zst"
    lines = []
    for row in rows:
        entry = {"schemaVersion": 1, "status": "complete", "sample": row["id"],
                 "sourceReference": row["reference"], "sourceSha256": row["sourceSha256"],
                 "archive": archive_relative}
        if row.get("skipReason"):
            entry.update({"artifact": "skipped", "skipReason": row["skipReason"]})
        lines.append(json.dumps(entry, separators=(",", ":")))
    manifest_path.write_text("\n".join(lines) + "\n")
    archive_path = _batch_archive(rows, key)

    def upload(path: Path, destination: str, stage: str) -> None:
        for attempt in range(UPLOAD_ATTEMPTS):
            started = time.monotonic()
            size = path.stat().st_size
            _event("upload_started", lane=lane, stage=stage, batch=key, attempt=attempt + 1,
                   bytes=size, destination=destination)
            try:
                command = ["rclone", "copyto", str(path), _remote(drive.remote, destination),
                           "--tpslimit", "1", "--drive-chunk-size", UPLOAD_CHUNK_SIZE,
                           "--timeout", UPLOAD_IDLE_TIMEOUT, "--contimeout", "15s",
                           "--config", str(drive.config), "--retries", "1", "--low-level-retries", "10",
                           "--stats", "30s", "--stats-one-line", "--stats-log-level", "NOTICE"]
                process = subprocess.Popen(command, text=True, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE)
                messages: list[str] = []

                def relay() -> None:
                    assert process.stderr
                    for line in process.stderr:
                        message = line.rstrip()
                        if not message:
                            continue
                        messages.append(message)
                        del messages[:-20]
                        _event("upload_progress", lane=lane, stage=stage, batch=key,
                               attempt=attempt + 1, message=message)

                reader = threading.Thread(target=relay, daemon=True)
                reader.start()
                try:
                    status = process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("rclone copy timed out after 3600 seconds")
                finally:
                    reader.join(timeout=10)
                if status:
                    raise RuntimeError("; ".join(messages[-5:]) or f"rclone exited {status}")
                seconds = time.monotonic() - started
                _event("upload_completed", lane=lane, stage=stage, batch=key, attempt=attempt + 1,
                       bytes=size, seconds=round(seconds, 3),
                       mibPerSecond=round(size / max(seconds, .001) / 1024**2, 3))
                return
            except Exception as error:
                _event("upload_retry", lane=lane, stage=stage, batch=key, attempt=attempt + 1,
                       seconds=round(time.monotonic() - started, 3),
                       error=f"{type(error).__name__}: {error}")
                if attempt == UPLOAD_ATTEMPTS - 1:
                    raise
                time.sleep(min(120, 15 * 2**attempt))

    upload(archive_path, archive_relative, "archive")
    upload(manifest_path, f"{target}/batches/{manifest_path.name}", "manifest")
    archive_path.unlink(missing_ok=True)


def _publish_json(path: Path, value: dict | list) -> None:
    """Publish one closed JSON object to the shared object-store volume."""
    with tempfile.NamedTemporaryFile("w", delete=False) as output:
        temporary = Path(output.name)
        output.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")
    try:
        _publish_visible(temporary, path, temporary.stat().st_size)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_job_files(root: Path, job_id: int, *, keep_source: bool = False) -> None:
    paths = [root / "results" / f"{job_id:06d}.json",
             root / "results" / f"{job_id:06d}.tar",
             root / "ready" / f"{job_id:06d}.json",
             root / "done" / f"{job_id:06d}.json"]
    if not keep_source:
        paths.append(root / "sources" / f"{job_id:06d}.slp")
    for path in paths:
        path.unlink(missing_ok=True)


def _prepare_resume(root: Path, all_jobs: list[dict], uploaded: set[str]) -> dict:
    """Reconcile a preserved run root against manifest-backed Drive commits."""
    jobs_by_id = {int(job["id"]): job for job in all_jobs}
    previous_failures = []
    for path in sorted((root / "failures").glob("*.json")):
        previous_failures.append({"file": path.name, "detail": _published_json(path) or path.read_text(errors="replace")})

    preserved_results: set[int] = set()
    invalid_results = 0
    for marker in sorted((root / "results").glob("*.json")):
        try:
            job_id = int(marker.stem)
        except ValueError:
            marker.unlink(missing_ok=True)
            invalid_results += 1
            continue
        job = jobs_by_id.get(job_id)
        if job and job["reference"] not in uploaded and _matching_result(marker, job):
            preserved_results.add(job_id)
            continue
        _remove_job_files(root, job_id)
        invalid_results += 1

    preserved_ready: set[int] = set()
    invalid_ready = 0
    for marker in sorted((root / "ready").glob("*.json")):
        try:
            job_id = int(marker.stem)
        except ValueError:
            marker.unlink(missing_ok=True)
            invalid_ready += 1
            continue
        job = jobs_by_id.get(job_id)
        if job_id in preserved_results:
            marker.unlink(missing_ok=True)
            continue
        if (job and job["reference"] not in uploaded
                and _matching_shared_marker(marker, job, "sourcePath", "sourceBytes")):
            preserved_ready.add(job_id)
            continue
        _remove_job_files(root, job_id)
        invalid_ready += 1

    keep_sources = preserved_results | preserved_ready
    for source in (root / "sources").glob("*.slp"):
        try:
            keep = int(source.stem) in keep_sources
        except ValueError:
            keep = False
        if not keep:
            source.unlink(missing_ok=True)

    cleared = {}
    for name in ("done", "failures", "workers", "upload-queue", "upload-acks"):
        paths = list((root / name).glob("*"))
        cleared[name] = len(paths)
        for path in paths:
            if path.is_file():
                path.unlink(missing_ok=True)
    for name in ("stop", "producer.done", "fleet.json"):
        (root / name).unlink(missing_ok=True)

    summary = {
        "event": "run_root_reconciled",
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "uploaded": len(uploaded),
        "preservedResults": len(preserved_results),
        "preservedReady": len(preserved_ready),
        "invalidResultsRemoved": invalid_results,
        "invalidReadyRemoved": invalid_ready,
        "cleared": cleared,
        "previousFailures": previous_failures,
    }
    _publish_json(root / "diagnostics" / "recoveries" / f"{time.time_ns()}.json", summary)
    return summary


def _choose_batch(markers: list[Path], limit: int) -> tuple[list[Path], int, list[dict]]:
    chosen: list[Path] = []
    rows: list[dict] = []
    size = 0
    for marker in markers[:limit]:
        row = _read_json(marker)
        item_size = int(row["sourceBytes"]) + int(row["resultBytes"])
        if chosen and size + item_size > BATCH_BYTES:
            break
        chosen.append(marker)
        rows.append(row)
        size += item_size
    return chosen, size, rows


def _cleanup_upload(root: Path, job: dict) -> None:
    """Idempotently retire the shared inputs after a manifest-last commit."""
    records = job.get("records", [])
    for record in records:
        job_id = int(record["id"])
        _json(root / "done" / f"{job_id:06d}.json",
              {"id": job_id, "reference": record["reference"]})
        for path in (root / "results" / f"{job_id:06d}.json",
                     root / "results" / f"{job_id:06d}.tar",
                     root / "sources" / f"{job_id:06d}.slp",
                     root / "ready" / f"{job_id:06d}.json"):
            path.unlink(missing_ok=True)


def uploader(args: argparse.Namespace) -> None:
    """Run one fixed upload lane; the coordinator is the only job scheduler."""
    root = Path(args.root)
    for name in ("upload-queue", "upload-acks", "done", "failures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    lane = int(args.lane)
    job_path = root / "upload-queue" / f"{lane:03d}.json"
    ack_path = root / "upload-acks" / f"{lane:03d}.json"
    drive = Drive(args.remote, Path(args.config))
    try:
        while not (root / "stop").exists():
            if not job_path.exists():
                time.sleep(.25)
                continue
            job = _read_json(job_path)
            token = str(job["token"])
            ack = _published_json(ack_path)
            if not ack or ack.get("token") != token:
                markers = [Path(path) for path in job["markers"]]
                _upload_batch(drive, root, args.target, markers, lane)
                _publish_json(ack_path, {"token": token})
            cleanup_started = time.monotonic()
            _event("upload_cleanup_started", lane=lane, count=len(job.get("records", [])))
            _cleanup_upload(root, job)
            job_path.unlink(missing_ok=True)
            ack_path.unlink(missing_ok=True)
            _event("upload_cleanup_completed", lane=lane, count=len(job.get("records", [])),
                   seconds=round(time.monotonic() - cleanup_started, 3))
    except BaseException as error:
        detail = {"error": f"uploader lane {lane}: {type(error).__name__}: {error}",
                  "lane": lane, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "traceback": traceback.format_exc()}
        _publish_json(root / "failures" / f"uploader-{lane:03d}.json", detail)
        _publish_json(root / "diagnostics" / "uploader-failures" /
                      f"lane-{lane:03d}-{time.time_ns()}.json", detail)
        (root / "stop").touch()


def coordinator(args: argparse.Namespace) -> None:
    root, state_path = Path(args.root), Path(args.state)
    for name in ("sources", "ready", "results", "done", "failures", "workers",
                 "upload-queue", "upload-acks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    drive = Drive(args.remote, Path(args.config))
    try:
        all_jobs, uploaded = _jobs(drive, args.source, args.target, args.sample_limit)
        recovery = _prepare_resume(root, all_jobs, uploaded)
        _event("run_root_reconciled", **{key: value for key, value in recovery.items()
                                         if key not in {"event", "time", "previousFailures"}})
        jobs = [job for job in all_jobs if job["reference"] not in uploaded]
        _json(root / "jobs.json", jobs)
        state = {"state": "complete" if not jobs else "ready", "total": len(all_jobs),
                 "alreadyUploaded": len(uploaded), "remaining": len(jobs), "uploaded": len(uploaded)}
        _json(state_path, state, atomic=True)
        if not jobs:
            return
        while not (root / "fleet.json").exists():
            if (root / "stop").exists():
                return
            time.sleep(1)
        fleet = _read_json(root / "fleet.json")
        slots = int(fleet["workers"]) * int(fleet["processes"])
        lanes = 1 + int(fleet.get("uploaders", 0))
        state["state"] = "running"; state["slots"] = slots; state["uploadLanes"] = lanes
        _json(state_path, state, atomic=True)
        errors: list[str] = []
        producer = threading.Thread(target=_stream_sources, args=(drive, args.source, root, jobs, errors), daemon=True)
        producer.start()
        local_uploader = argparse.Namespace(root=str(root), lane=0, remote=args.remote,
                                            target=args.target, config=args.config)
        threading.Thread(target=uploader, args=(local_uploader,), daemon=True).start()
        assigned: dict[int, set[Path]] = {}
        for lane in range(lanes):
            lane_job = root / "upload-queue" / f"{lane:03d}.json"
            if lane_job.exists():
                job = _read_json(lane_job)
                assigned[lane] = {Path(path) for path in job["markers"]}
        last_done = -1
        while True:
            failures = sorted((root / "failures").glob("*.json"))
            if errors or failures:
                detail = errors[0] if errors else failures[0].read_text()
                raise RuntimeError(detail)
            for lane, claimed in list(assigned.items()):
                lane_job = root / "upload-queue" / f"{lane:03d}.json"
                if not lane_job.exists() and all(not path.exists() for path in claimed):
                    assigned.pop(lane)
            claimed = set().union(*assigned.values()) if assigned else set()
            markers = [path for path in sorted((root / "results").glob("*.json"))
                       if path not in claimed and _published_json(path) is not None]
            terminal = (root / "producer.done").exists() and len(list((root / "workers").glob("*.done"))) == slots
            old = markers and time.time() - min(path.stat().st_mtime for path in markers) >= 300
            done = len(list((root / "done").glob("*.json")))
            if done != last_done:
                last_done = done
                state["uploaded"] = len(uploaded) + done
                state["remaining"] = len(all_jobs) - state["uploaded"]
                state["uploading"] = sum(len(batch) for batch in assigned.values())
                _json(state_path, state, atomic=True)
            free = [lane for lane in range(lanes) if lane not in assigned]
            enough_for_lanes = len(markers) >= BATCH_MIN_SIZE * len(free)
            if free and markers and (len(markers) >= BATCH_SIZE or enough_for_lanes or terminal or old):
                while free and markers:
                    lane = free.pop(0)
                    per_lane = min(BATCH_SIZE, max(1, math.ceil(len(markers) / (len(free) + 1))))
                    chosen, size, rows = _choose_batch(markers, per_lane)
                    if not chosen:
                        raise RuntimeError("could not form a non-empty upload batch")
                    token = hashlib.sha256(
                        ("\n".join(row["reference"] for row in rows) + f"\n{time.time_ns()}").encode()
                    ).hexdigest()[:24]
                    job = {"token": token, "lane": lane, "count": len(chosen), "bytes": size,
                           "markers": [str(path) for path in chosen],
                           "records": [{"id": row["id"], "reference": row["reference"]} for row in rows]}
                    _publish_json(root / "upload-queue" / f"{lane:03d}.json", job)
                    assigned[lane] = set(chosen)
                    chosen_set = set(chosen)
                    markers = [path for path in markers if path not in chosen_set]
                state["uploading"] = sum(len(batch) for batch in assigned.values())
                _json(state_path, state, atomic=True)
            elif terminal and not markers and not assigned:
                if list((root / "ready").glob("*.json")):
                    raise RuntimeError("workers exited with unprocessed sources")
                state.update({"state": "complete", "uploaded": len(all_jobs), "remaining": 0})
                _json(state_path, state, atomic=True)
                return
            elif (root / "stop").exists():
                return
            else:
                time.sleep(1)
    except BaseException as error:
        detail = {"state": "failed", "error": f"{type(error).__name__}: {error}"}
        _json(state_path, detail, atomic=True)
        with contextlib.suppress(Exception):
            _publish_json(root / "diagnostics" / "coordinator-failure.json",
                          {**detail, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "traceback": traceback.format_exc()})
        (root / "stop").touch()


def _skip_result(path: Path) -> None:
    metadata = path.parent / "metadata.json"
    metadata.write_text(json.dumps({"skipReason": "no_playable_frames"}) + "\n")
    with tarfile.open(path, "w") as archive:
        archive.add(metadata, arcname="metadata.json", recursive=False)


def worker(args: argparse.Namespace) -> None:
    root = Path(args.root)
    for name in ("results", "failures", "workers"):
        (root / name).mkdir(parents=True, exist_ok=True)
    jobs = _read_json(root / "jobs.json")
    slots = args.workers * args.processes
    cpu = _cpus()
    if args.processes > cpu:
        raise ValueError(f"{args.processes} processes exceed {cpu} CPUs")
    os.environ["SMASH_RENDERER_SNAPSHOT"] = args.snapshot
    errors: list[str] = []

    def slot(local_slot: int) -> None:
        slot_id = args.index * args.processes + local_slot
        work = Path("/tmp/smash-worker") / str(slot_id)
        work.mkdir(parents=True, exist_ok=True)
        pending = {job["id"]: job for job in jobs if job["id"] % slots == slot_id}
        failure_cycles: dict[int, int] = {}
        deferred_until: dict[int, float] = {}
        render_attempts = int(os.environ.get("SMASH_RENDER_ATTEMPTS", "3"))
        max_failure_cycles = int(os.environ.get("SMASH_RENDER_FAILURE_CYCLES", "3"))
        retry_cooldown = int(os.environ.get("SMASH_RENDER_RETRY_COOLDOWN_SECONDS", "60"))
        if render_attempts < 1 or max_failure_cycles < 1 or retry_cooldown < 0:
            raise ValueError("render retry settings must be positive")
        producer_finished_at = None
        while pending:
            if (root / "stop").exists():
                return
            for job_id in list(pending):
                marker = root / "results" / f"{job_id:06d}.json"
                committed = (root / "done" / f"{job_id:06d}.json").exists()
                if marker.exists() and not committed:
                    try:
                        row = _read_json(marker)
                        committed = row.get("id") == job_id and row.get("reference") == pending[job_id]["reference"]
                    except (AttributeError, OSError, json.JSONDecodeError):
                        marker.unlink(missing_ok=True)
                if committed:
                    pending.pop(job_id)
            now = time.monotonic()
            ready = None
            has_assigned_ready = False
            for path in (root / "ready").glob("*.json"):
                if not path.stem.isdigit() or int(path.stem) not in pending:
                    continue
                has_assigned_ready = True
                if deferred_until.get(int(path.stem), 0) <= now:
                    ready = path
                    break
            if ready is None:
                if has_assigned_ready:
                    time.sleep(.25)
                    continue
                if (root / "producer.done").exists():
                    producer_finished_at = producer_finished_at or time.monotonic()
                    if time.monotonic() - producer_finished_at > 30:
                        raise RuntimeError(f"producer omitted assigned jobs: {sorted(pending)[:3]}")
                time.sleep(.25)
                continue
            producer_finished_at = None
            job = _read_json(ready)
            source = work / "input.slp"
            _copy_verified(Path(job["sourcePath"]), source, job["sourceBytes"], job["sourceSha256"])
            result = work / "result.tar"
            render_error: Exception | None = None
            for attempt in range(render_attempts):
                try:
                    result = replay_renderer.render(job, source, work, max(1, cpu // args.processes))
                    skip = None
                    break
                except replay_renderer.NoPlayableFrames:
                    _skip_result(result); skip = "no_playable_frames"; break
                except Exception as error:
                    disk = shutil.disk_usage(work)
                    detail = {
                        "event": "render_attempt_failed",
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "workerIndex": args.index,
                        "slotId": slot_id,
                        "localSlot": local_slot,
                        "jobId": int(job["id"]),
                        "reference": job["reference"],
                        "attempt": attempt + 1,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
                    }
                    _publish_json(root / "diagnostics" / "render-failures" /
                                  f"{int(job['id']):06d}-{time.time_ns()}-attempt-{attempt + 1}.json", detail)
                    _event("render_attempt_failed", workerIndex=args.index, slotId=slot_id,
                           jobId=int(job["id"]), attempt=attempt + 1,
                           diskFree=disk.free, error=detail["error"][-2000:])
                    if attempt == render_attempts - 1:
                        render_error = error
                        break
                    time.sleep(2**attempt)
            if render_error is not None:
                job_id = int(job["id"])
                cycle = failure_cycles.get(job_id, 0) + 1
                failure_cycles[job_id] = cycle
                if cycle >= max_failure_cycles:
                    raise render_error
                deferred_until[job_id] = time.monotonic() + retry_cooldown
                _event("render_job_deferred", workerIndex=args.index, slotId=slot_id,
                       jobId=job_id, failureCycle=cycle, maxFailureCycles=max_failure_cycles,
                       retryAfterSeconds=retry_cooldown)
                continue
            shared = root / "results" / f"{job['id']:06d}.tar"
            result_bytes, result_sha256 = result.stat().st_size, _sha256(result)
            _publish_visible(result, shared, result_bytes)
            row = {**job, "resultPath": str(shared), "resultBytes": result_bytes,
                   "resultSha256": result_sha256}
            if skip:
                row["skipReason"] = skip
            _json(root / "results" / f"{job['id']:06d}.json", row)
            shutil.rmtree(work / f"{job['id']:06d}", ignore_errors=True)
            pending.pop(job["id"])
        (root / "workers" / f"{slot_id:04d}.done").touch()

    def guarded(local_slot: int) -> None:
        try:
            slot(local_slot)
        except BaseException as error:
            message = f"slot {local_slot}: {type(error).__name__}: {error}"
            errors.append(message)
            _publish_json(root / "failures" / f"{args.index:03d}-{local_slot}.json",
                          {"error": message, "workerIndex": args.index, "localSlot": local_slot,
                           "traceback": traceback.format_exc()})
            (root / "stop").touch()

    threads = [threading.Thread(target=guarded, args=(index,)) for index in range(args.processes)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError(errors[0])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CPU-only Daytona replay capture")
    commands = parser.add_subparsers(dest="command", required=True)
    coord = commands.add_parser("coordinator")
    for name in ("remote", "source", "target", "root", "config", "state"):
        coord.add_argument(f"--{name}", required=True)
    coord.add_argument("--sample-limit", type=int, default=0)
    render = commands.add_parser("worker")
    render.add_argument("--root", required=True); render.add_argument("--index", type=int, required=True)
    render.add_argument("--workers", type=int, required=True); render.add_argument("--processes", type=int, required=True)
    render.add_argument("--snapshot", required=True)
    upload = commands.add_parser("uploader")
    upload.add_argument("--root", required=True); upload.add_argument("--lane", type=int, required=True)
    for name in ("remote", "target", "config"):
        upload.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    if args.command == "coordinator":
        coordinator(args)
    elif args.command == "worker":
        worker(args)
    else:
        uploader(args)


if __name__ == "__main__":
    main()
