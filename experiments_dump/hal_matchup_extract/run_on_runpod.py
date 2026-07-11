#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / ".runpod-state.json"
REMOTE_ROOT = "/workspace/hal-matchup"
DRIVE_DESTINATION = "hal-fox-captain-falcon-battlefield"
ARCHIVES = [
    (1, "1pFjgh1dapX34s0T-Q1TC7JUO-qjYbQZf", "ranked-anonymized-1-116248.7z", 70_894_826_033),
    (2, "1jEIzvhpV3778J2s2-Np9vCVqSLf9lZnk", "ranked-anonymized-2-151807.7z", 93_487_127_678),
    (3, "1glzlkAPxHC58oXZljJXQV8dsTBKmlhkE", "ranked-anonymized-3-128787.zip", 98_825_155_729),
    (4, "1qdIZUW4Er_Vu6rD3-VUvyak3lKa1KxVk", "ranked-anonymized-4-148358.zip", 115_039_986_805),
    (5, "1Hqmj6C8g1BzuRAIqOrQcMDL0MX4GtffE", "ranked-anonymized-5-133261.zip", 103_634_403_158),
    (6, "1g8yZ-Q4ldyhDEmXLSPBoWxywJRMRVGc3", "ranked-anonymized-6-171694.zip", 134_721_309_960),
]


@dataclass(frozen=True)
class Host:
    id: str
    name: str
    host: str
    port: int
    shard: int
    drive_id: str
    archive_name: str
    archive_bytes: int


class Controller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api_key = self._env("RUNPOD_API_KEY")
        self.public_key = self._public_key()
        self.base_url = "https://rest.runpod.io/v1"
        self.rclone_config = Path.home() / ".config" / "rclone" / "rclone.conf"

    def provision(self) -> None:
        if STATE_PATH.exists():
            raise FileExistsError(f"state already exists: {STATE_PATH}")
        if not self.rclone_config.exists():
            raise FileNotFoundError(self.rclone_config)
        created: list[str] = []
        try:
            pods = []
            for archive in ARCHIVES:
                pod = self._create_pod(archive)
                created.append(pod["id"])
                pods.append(pod)
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                hosts = list(executor.map(lambda pod: self._wait_for_ssh(pod), pods))
            self._save_state({"phase": "bootstrapping", "hosts": [host.__dict__ for host in hosts]})
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                list(executor.map(self._bootstrap_and_launch, hosts))
        except Exception:
            for pod_id in created:
                with contextlib.suppress(Exception):
                    self.request("DELETE", f"/pods/{pod_id}")
            STATE_PATH.unlink(missing_ok=True)
            raise
        self._save_state({"phase": "scanning", "hosts": [host.__dict__ for host in hosts]})
        print(json.dumps({"event": "all_scans_started", "pods": len(hosts), "vcpuPerPod": 32, "diskGbPerPod": self.args.disk_gb}))

    def stage_sources(self) -> None:
        def stage(archive: tuple[int, str, str, int]) -> None:
            _, drive_id, archive_name, _ = archive
            subprocess.run(
                [
                    "rclone",
                    "backend",
                    "copyid",
                    "smash-drive:",
                    drive_id,
                    f"smash-drive:_hal_sources/{archive_name}",
                    "--config",
                    str(self.rclone_config),
                    "--retries",
                    "20",
                    "--low-level-retries",
                    "20",
                ],
                check=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(stage, ARCHIVES))
        listing = subprocess.run(
            ["rclone", "lsl", "smash-drive:_hal_sources", "--config", str(self.rclone_config)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        observed = {line.split(maxsplit=3)[3]: int(line.split(maxsplit=1)[0]) for line in listing.splitlines()}
        missing = [name for _, _, name, size in ARCHIVES if observed.get(name) != size]
        if missing:
            raise RuntimeError(f"source staging verification failed: {missing}")
        print(json.dumps({"event": "sources_staged", "archives": len(ARCHIVES)}))

    def status(self) -> None:
        state = self._load_state()
        rows = []
        for host in self._hosts(state):
            pod = self.request("GET", f"/pods/{host.id}")
            output = self.ssh(
                host,
                f"printf 'ready='; test -f {REMOTE_ROOT}/READY && echo yes || echo no; "
                f"printf 'uploaded='; test -f {REMOTE_ROOT}/UPLOAD_COMPLETE && echo yes || echo no; "
                f"find {REMOTE_ROOT}/archive -maxdepth 1 -type f -printf 'archive=%s bytes\\n' 2>/dev/null || true; "
                f"du -sh {REMOTE_ROOT}/output {REMOTE_ROOT}/shard-{host.shard:02d}.tar.zst 2>/dev/null || true; "
                f"tail -c 65536 {REMOTE_ROOT}/job.log 2>/dev/null | tr '\\r' '\\n' | tail -n {self.args.lines} || true",
            )
            rows.append(
                {
                    "shard": host.shard,
                    "podId": host.id,
                    "status": pod.get("status") or pod.get("desiredStatus"),
                    "vcpuCount": pod.get("vcpuCount"),
                    "cpuFlavorId": pod.get("cpuFlavorId"),
                    "costPerHr": pod.get("costPerHr"),
                    "output": output,
                }
            )
        print(json.dumps({"phase": state["phase"], "pods": rows}, indent=2))

    def restart(self) -> None:
        state = self._load_state()
        hosts = self._hosts(state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(self._restart, hosts))
        state["phase"] = "scanning"
        self._save_state(state)
        print(json.dumps({"event": "all_scans_restarted", "pods": len(hosts)}))

    def upload(self) -> None:
        state = self._load_state()
        hosts = self._hosts(state)
        not_ready = [
            host.shard
            for host in hosts
            if self.ssh(host, f"test -f {REMOTE_ROOT}/READY && echo yes || echo no").strip() != "yes"
        ]
        if not_ready:
            raise RuntimeError(f"shards are not ready: {not_ready}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            pids = list(executor.map(self._start_upload, hosts))
        state["phase"] = "uploading"
        state["uploadPids"] = dict(zip((str(host.shard) for host in hosts), pids, strict=True))
        self._save_state(state)
        print(json.dumps({"event": "batch_upload_started", "shards": len(hosts), "destination": DRIVE_DESTINATION}))

    def verify(self) -> None:
        state = self._load_state()
        hosts = self._hosts(state)
        incomplete = [
            host.shard
            for host in hosts
            if self.ssh(host, f"test -f {REMOTE_ROOT}/UPLOAD_COMPLETE && echo yes || echo no").strip() != "yes"
        ]
        if incomplete:
            raise RuntimeError(f"uploads are not complete: {incomplete}")
        expected = {}
        for host in hosts:
            line = self.ssh(host, f"md5sum {REMOTE_ROOT}/shard-{host.shard:02d}.tar.zst").strip()
            digest, _ = line.split(maxsplit=1)
            expected[f"shard-{host.shard:02d}.tar.zst"] = digest
        remote = subprocess.run(
            ["rclone", "md5sum", f"smash-drive:{DRIVE_DESTINATION}"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        actual = {}
        for line in remote.splitlines():
            digest, name = line.split(maxsplit=1)
            actual[name] = digest
        mismatch = {name: {"expected": digest, "actual": actual.get(name)} for name, digest in expected.items() if actual.get(name) != digest}
        if mismatch:
            raise RuntimeError(f"Drive verification failed: {mismatch}")
        state["phase"] = "verified"
        state["driveMd5"] = expected
        self._save_state(state)
        print(json.dumps({"event": "drive_verified", "shards": len(expected), "destination": DRIVE_DESTINATION}))

    def cleanup(self) -> None:
        state = self._load_state()
        if state["phase"] != "verified" and not self.args.force:
            raise RuntimeError("refusing cleanup before verified upload; pass --force")
        hosts = self._hosts(state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(lambda host: self.request("DELETE", f"/pods/{host.id}"), hosts))
        STATE_PATH.unlink(missing_ok=True)
        print(json.dumps({"event": "pods_deleted", "count": len(hosts)}))

    def cleanup_sources(self) -> None:
        state = self._load_state()
        if state["phase"] != "verified" and not self.args.force:
            raise RuntimeError("refusing source cleanup before verified upload; pass --force")
        subprocess.run(
            [
                "rclone",
                "purge",
                "smash-drive:_hal_sources",
                "--drive-use-trash=false",
                "--config",
                str(self.rclone_config),
            ],
            check=True,
        )
        print(json.dumps({"event": "staged_sources_deleted"}))

    def _create_pod(self, archive: tuple[int, str, str, int]) -> dict:
        shard, drive_id, archive_name, archive_bytes = archive
        payload = {
            "name": f"smash-hal-scan-{shard}-{time.time_ns()}",
            "imageName": self.args.image,
            "computeType": "CPU",
            "cloudType": self.args.cloud_type,
            "cpuFlavorIds": self.args.cpu_flavor,
            "cpuFlavorPriority": "availability",
            "vcpuCount": 32,
            "containerDiskInGb": self.args.disk_gb,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "dockerEntrypoint": ["/bin/bash", "-lc"],
            "dockerStartCmd": [self._start_command()],
            "env": {"PUBLIC_KEY": self.public_key},
        }
        last_error = None
        for attempt in range(6):
            try:
                pod = self.request("POST", "/pods", payload)
                break
            except RuntimeError as error:
                last_error = error
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        else:
            raise last_error or RuntimeError("RunPod pod creation failed")
        pod_id = str(pod.get("id") or pod.get("podId"))
        print(json.dumps({"event": "pod_created", "shard": shard, "podId": pod_id}), flush=True)
        return {"id": pod_id, "shard": shard, "drive_id": drive_id, "archive_name": archive_name, "archive_bytes": archive_bytes}

    def _wait_for_ssh(self, pod: dict) -> Host:
        deadline = time.monotonic() + self.args.wait_seconds
        while time.monotonic() < deadline:
            value = self.request("GET", f"/pods/{pod['id']}")
            mappings = value.get("portMappings") or {}
            port = mappings.get("22") or mappings.get("22/tcp")
            if isinstance(port, dict):
                port = port.get("hostPort") or port.get("port")
            host = value.get("publicIp") or value.get("ip")
            if host and port:
                result = Host(
                    id=pod["id"],
                    name=str(value.get("name") or ""),
                    host=str(host),
                    port=int(port),
                    shard=pod["shard"],
                    drive_id=pod["drive_id"],
                    archive_name=pod["archive_name"],
                    archive_bytes=pod["archive_bytes"],
                )
                try:
                    self.ssh(result, "true")
                    return result
                except RuntimeError:
                    pass
            time.sleep(5)
        raise TimeoutError(f"pod {pod['id']} did not expose SSH")

    def _bootstrap_and_launch(self, host: Host) -> None:
        bootstrap = (
            "set -e; apt-get update >/dev/null; "
            "apt-get install -y --no-install-recommends build-essential ca-certificates curl git rsync unzip zstd >/dev/null; "
            "curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; "
            "git clone --depth 1 https://github.com/ericyuegu/hal.git /workspace/hal >/dev/null 2>&1; "
            "/root/.local/bin/uv venv --python 3.14 /workspace/venv >/dev/null; "
            "/root/.local/bin/uv pip install --python /workspace/venv/bin/python "
            "numpy pyarrow fsspec loguru tqdm tyro 'py7zr>=1.1.0' "
            "'git+https://github.com/ericyuegu/libmelee@canonical-schema' "
            "'git+https://github.com/ericyuegu/peppi-py@libmelee-parity' >/dev/null; "
            "curl -fsSL https://rclone.org/install.sh | bash >/dev/null"
        )
        self.ssh(host, bootstrap)
        self.ssh(host, f"mkdir -p {REMOTE_ROOT} /root/.config/rclone")
        self.rsync(host, ROOT / "process_archive.py", f"{REMOTE_ROOT}/process_archive.py")
        self.rsync(host, self.rclone_config, "/root/.config/rclone/rclone.conf")
        self.ssh(host, "chmod 600 /root/.config/rclone/rclone.conf")
        self._launch(host)

    def _restart(self, host: Host) -> None:
        self.rsync(host, ROOT / "process_archive.py", f"{REMOTE_ROOT}/process_archive.py")
        self.ssh(
            host,
            f"if test -f {REMOTE_ROOT}/scan.pid; then kill -- -$(cat {REMOTE_ROOT}/scan.pid) 2>/dev/null || true; fi; "
            f"pkill -f '[r]clone copyto smash-drive:_hal_sources' || true; "
            f"pkill -f '[p]rocess_archive.py' || true; sleep 1; "
            f"rm -rf {REMOTE_ROOT}/archive {REMOTE_ROOT}/output {REMOTE_ROOT}/shard-{host.shard:02d}.tar.zst "
            f"{REMOTE_ROOT}/READY {REMOTE_ROOT}/UPLOAD_COMPLETE {REMOTE_ROOT}/job.log {REMOTE_ROOT}/scan.pid",
        )
        self._launch(host)

    def _launch(self, host: Host) -> None:
        download_command = (
            f"rclone copyto smash-drive:_hal_sources/{shlex.quote(host.archive_name)} "
            f"archive/{shlex.quote(host.archive_name)} --config /root/.config/rclone/rclone.conf "
            f"--transfers 1 --checkers 2 --multi-thread-streams 16 --retries 20 --low-level-retries 20; "
        )
        command = (
            f"set -euo pipefail; cd {REMOTE_ROOT}; mkdir -p archive output; "
            f"{download_command}"
            f"test $(stat -c %s archive/{shlex.quote(host.archive_name)}) -eq {host.archive_bytes}; "
            f"PYTHONPATH=/workspace/hal /workspace/venv/bin/python process_archive.py "
            f"--archive archive/{shlex.quote(host.archive_name)} --output output --workers 32; "
            f"tar --zstd -cf shard-{host.shard:02d}.tar.zst -C output files manifest.jsonl report.json paths.txt; "
            f"touch READY"
        )
        self.ssh(
            host,
            f"nohup setsid bash -lc {shlex.quote(command)} > {REMOTE_ROOT}/job.log 2>&1 < /dev/null & "
            f"echo $! > {REMOTE_ROOT}/scan.pid",
        )
        print(json.dumps({"event": "scan_started", "shard": host.shard, "podId": host.id}), flush=True)

    def _start_upload(self, host: Host) -> int:
        command = (
            f"set -euo pipefail; cd {REMOTE_ROOT}; "
            f"rclone copyto shard-{host.shard:02d}.tar.zst "
            f"smash-drive:{DRIVE_DESTINATION}/shard-{host.shard:02d}.tar.zst "
            f"--config /root/.config/rclone/rclone.conf --drive-chunk-size 128M --transfers 1 --checkers 4 "
            f"--contimeout 30s --timeout 2m --retries 10 --low-level-retries 20 --stats 15s -v; "
            f"rclone copyto output/report.json smash-drive:{DRIVE_DESTINATION}/report-{host.shard:02d}.json "
            f"--config /root/.config/rclone/rclone.conf; "
            f"touch UPLOAD_COMPLETE"
        )
        return int(
            self.ssh(
                host,
                f"nohup setsid bash -lc {shlex.quote(command)} >> {REMOTE_ROOT}/upload.log 2>&1 < /dev/null & echo $!",
            ).strip()
        )

    def _start_command(self) -> str:
        return (
            "apt-get update && apt-get install -y --no-install-recommends openssh-server ca-certificates && "
            "mkdir -p /run/sshd /root/.ssh /workspace && chmod 700 /root/.ssh && "
            "printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && "
            "/usr/sbin/sshd && sleep infinity"
        )

    def request(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read().decode()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"RunPod {method} {path} failed: HTTP {error.code}: {detail}") from error
        return json.loads(raw) if raw else {}

    def ssh(self, host: Host, command: str) -> str:
        completed = subprocess.run(self._ssh_base(host) + [command], text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed.stdout

    def rsync(self, host: Host, source: Path, target: str) -> None:
        command = [
            "rsync",
            "-a",
            "-e",
            shlex.join(self._ssh_base(host)[:-1]),
            str(source),
            f"root@{host.host}:{target}",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    def _ssh_base(self, host: Host) -> list[str]:
        return [
            "ssh",
            "-p",
            str(host.port),
            "-i",
            str(Path(self.args.ssh_private_key).expanduser()),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
            f"root@{host.host}",
        ]

    def _hosts(self, state: dict) -> list[Host]:
        return [Host(**row) for row in state["hosts"]]

    def _save_state(self, state: dict) -> None:
        STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    def _load_state(self) -> dict:
        if not STATE_PATH.exists():
            raise FileNotFoundError(f"no state: {STATE_PATH}")
        return json.loads(STATE_PATH.read_text())

    def _env(self, name: str) -> str:
        value = os.environ.get(name)
        if value:
            return value
        zshrc = Path.home() / ".zshrc"
        if zshrc.exists():
            match = re.search(rf"^export {name}=[\"']?([^\"'\n]+)", zshrc.read_text(), re.M)
            if match:
                return match.group(1).strip()
        raise ValueError(f"{name} is required")

    def _public_key(self) -> str:
        path = Path(os.environ.get("RUNPOD_PUBLIC_KEY_FILE", "~/.ssh/id_ed25519.pub")).expanduser()
        return path.read_text().strip()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "command",
        choices=("stage_sources", "provision", "status", "restart", "upload", "verify", "cleanup_sources", "cleanup"),
    )
    value.add_argument("--image", default="ubuntu:22.04")
    value.add_argument("--cloud-type", default="COMMUNITY")
    value.add_argument("--cpu-flavor", action="append", default=["cpu5c", "cpu3c"])
    value.add_argument("--disk-gb", type=int, default=250)
    value.add_argument("--wait-seconds", type=int, default=1200)
    value.add_argument("--lines", type=int, default=20)
    value.add_argument("--ssh-private-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    value.add_argument("--force", action="store_true")
    return value


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    getattr(Controller(args), args.command)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
