#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
REMOTE_ROOT = "/workspace/hf-video-migration"


@dataclass(frozen=True)
class Host:
    id: str
    name: str
    host: str
    port: int


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api_key = self._env("RUNPOD_API_KEY")
        self.hf_token = self._env("HF_TOKEN")
        self.public_key = self._public_key()
        self.base_url = "https://rest.runpod.io/v1"

    def provision(self):
        source = self.hf_inventory(self.args.repo)
        required_gb = math.ceil(source["bytes"] * 1.15 / 1_000_000_000 / 500) * 500
        disk_gb = self.args.disk_gb or required_gb
        if disk_gb < required_gb:
            raise ValueError(f"--disk-gb must be at least {required_gb} for this {source['bytes']}-byte repo")
        name = f"smash-hf-h264-migration-{time.time_ns()}"
        payload = {
            "name": name,
            "imageName": self.args.image,
            "computeType": "CPU",
            "cloudType": self.args.cloud_type,
            "cpuFlavorIds": self.args.cpu_flavor,
            "cpuFlavorPriority": "custom",
            "vcpuCount": 32,
            "containerDiskInGb": 20,
            "volumeInGb": disk_gb,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "dockerEntrypoint": ["/bin/bash", "-lc"],
            "dockerStartCmd": [self.start_command()],
            "env": {"PUBLIC_KEY": self.public_key},
        }
        pod = self.request("POST", "/pods", payload)
        pod_id = str(pod.get("id") or pod.get("podId"))
        print(json.dumps({"event": "pod_created", "podId": pod_id, "diskGb": disk_gb}), flush=True)
        try:
            host = self.wait_for_ssh(pod_id)
            self.bootstrap(host)
            self.require_remote_disk(host, source["bytes"])
            self.copy_worker(host)
            self.write_env(host)
            pid = self.launch(host, "prepare")
        except Exception:
            self.request("DELETE", f"/pods/{pod_id}")
            raise
        self.save_state(
            {
                "repo": self.args.repo,
                "podId": host.id,
                "name": host.name,
                "host": host.host,
                "port": host.port,
                "diskGb": disk_gb,
                "source": source,
                "phase": "prepare",
                "pid": pid,
            }
        )
        print(json.dumps({"event": "prepare_started", "podId": host.id, "pid": pid}), flush=True)

    def status(self):
        state = self.load_state()
        pod = self.request("GET", f"/pods/{state['podId']}")
        print(
            json.dumps(
                {
                    "podId": state["podId"],
                    "status": pod.get("status") or pod.get("desiredStatus"),
                    "vcpuCount": pod.get("vcpuCount"),
                    "cpuFlavorId": pod.get("cpuFlavorId"),
                    "containerDiskInGb": pod.get("containerDiskInGb"),
                    "volumeInGb": pod.get("volumeInGb"),
                    "costPerHr": pod.get("costPerHr"),
                }
            )
        )
        host = self.host_from_state(state)
        output = self.ssh(
            host,
            f"tail -n {self.args.lines} {REMOTE_ROOT}/migration.log 2>/dev/null || true; "
            f"test -f {REMOTE_ROOT}/migration-report.json && "
            f"(set -a; . {REMOTE_ROOT}/.env; set +a; "
            f"python3 {REMOTE_ROOT}/migrate.py status --repo {shlex.quote(state['repo'])} "
            f"--work-dir {REMOTE_ROOT}) || true",
        )
        print(output)

    def fetch(self):
        state = self.load_state()
        host = self.host_from_state(state)
        target = ROOT / "spotchecks"
        target.mkdir(parents=True, exist_ok=True)
        self.rsync(host, f"{REMOTE_ROOT}/spotchecks/", str(target) + "/", from_remote=True)
        self.rsync(host, f"{REMOTE_ROOT}/migration-report.json", str(target / "migration-report.json"), from_remote=True)
        print(json.dumps({"event": "spotchecks_fetched", "path": str(target)}))

    def cutover(self):
        state = self.load_state()
        host = self.host_from_state(state)
        report = json.loads(self.ssh(host, f"cat {REMOTE_ROOT}/migration-report.json"))
        if report.get("status") != "prepared":
            raise RuntimeError(f"migration is not prepared: {report.get('status')}")
        pid = self.launch(host, "cutover")
        state.update({"phase": "cutover", "pid": pid})
        self.save_state(state)
        print(json.dumps({"event": "cutover_started", "podId": host.id, "pid": pid}), flush=True)

    def cleanup(self):
        state = self.load_state()
        if not self.args.force:
            host = self.host_from_state(state)
            report = json.loads(self.ssh(host, f"cat {REMOTE_ROOT}/migration-report.json"))
            if report.get("status") != "complete":
                raise RuntimeError("refusing to delete migration pod before verified completion; pass --force")
        self.request("DELETE", f"/pods/{state['podId']}")
        STATE_PATH.unlink(missing_ok=True)
        print(json.dumps({"event": "pod_deleted", "podId": state["podId"]}))

    def start_command(self) -> str:
        return (
            "apt-get update && "
            "apt-get install -y --no-install-recommends openssh-server ca-certificates rsync python3 python3-pip && "
            "mkdir -p /run/sshd /root/.ssh /workspace && chmod 700 /root/.ssh && "
            "printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && "
            "chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd && sleep infinity"
        )

    def wait_for_ssh(self, pod_id: str) -> Host:
        deadline = time.monotonic() + self.args.wait_seconds
        while time.monotonic() < deadline:
            pod = self.request("GET", f"/pods/{pod_id}")
            status = pod.get("status") or pod.get("desiredStatus")
            if status in {"EXITED", "TERMINATED"}:
                raise RuntimeError(f"RunPod pod {pod_id} stopped during initialization")
            mappings = pod.get("portMappings") or {}
            port = mappings.get("22") or mappings.get("22/tcp")
            if isinstance(port, dict):
                port = port.get("hostPort") or port.get("port")
            host = pod.get("publicIp") or pod.get("ip")
            if host and port:
                value = Host(pod_id, str(pod.get("name") or ""), str(host), int(port))
                try:
                    self.ssh(value, "true")
                    return value
                except RuntimeError:
                    pass
            time.sleep(5)
        raise TimeoutError(f"pod {pod_id} did not expose SSH")

    def bootstrap(self, host: Host):
        self.ssh(
            host,
            "apt-get update && apt-get install -y --no-install-recommends ffmpeg && "
            "python3 -m pip install --upgrade 'huggingface_hub[hf_xet]>=1.0.0'",
        )
        print(json.dumps({"event": "pod_bootstrapped", "podId": host.id}), flush=True)

    def require_remote_disk(self, host: Host, source_bytes: int):
        available = int(self.ssh(host, "df -B1 --output=avail /workspace | tail -1").strip())
        required = math.ceil(source_bytes * 1.10)
        if available < required:
            raise RuntimeError(f"RunPod under-allocated /workspace: available={available} required={required}")

    def copy_worker(self, host: Host):
        self.ssh(host, f"mkdir -p {REMOTE_ROOT}")
        self.rsync(host, str(ROOT / "migrate.py"), f"{REMOTE_ROOT}/migrate.py")

    def write_env(self, host: Host):
        content = "\n".join(
            [
                f"export HF_TOKEN={shlex.quote(self.hf_token)}",
                "export HF_XET_HIGH_PERFORMANCE=1",
                "export HF_HUB_ENABLE_HF_TRANSFER=0",
            ]
        )
        self.ssh_stdin(host, f"cat > {REMOTE_ROOT}/.env && chmod 600 {REMOTE_ROOT}/.env", content)

    def launch(self, host: Host, phase: str) -> int:
        command = (
            f"set -a; . {REMOTE_ROOT}/.env; set +a; "
            f"nohup python3 {REMOTE_ROOT}/migrate.py {phase} "
            f"--repo {shlex.quote(self.args.repo)} --work-dir {REMOTE_ROOT} "
            f">> {REMOTE_ROOT}/migration.log 2>&1 < /dev/null & echo $!"
        )
        return int(self.ssh(host, command).strip())

    def hf_inventory(self, repo: str) -> dict:
        request = urllib.request.Request(
            f"https://huggingface.co/api/datasets/{repo}?blobs=true",
            headers={"Authorization": f"Bearer {self.hf_token}"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
        siblings = data.get("siblings", [])
        return {
            "sha": data["sha"],
            "files": len(siblings),
            "bytes": sum(int(row.get("size") or 0) for row in siblings),
            "aviCount": sum(row["rfilename"].endswith("/video.avi") for row in siblings),
        }

    def request(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"RunPod {method} {path} failed: HTTP {error.code}: {detail}") from error
        return json.loads(raw) if raw else {}

    def ssh(self, host: Host, command: str) -> str:
        completed = subprocess.run(self.ssh_base(host) + [command], text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed.stdout

    def ssh_stdin(self, host: Host, command: str, content: str):
        completed = subprocess.run(
            self.ssh_base(host) + [command], input=content, text=True, capture_output=True
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    def rsync(self, host: Host, source: str, target: str, from_remote: bool = False):
        remote = f"root@{host.host}:"
        if from_remote:
            source = remote + source
        else:
            target = remote + target
        command = ["rsync", "-a", "--partial", "--inplace", "-e", shlex.join(self.ssh_base(host)[:-1]), source, target]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    def ssh_base(self, host: Host) -> list[str]:
        command = [
            "ssh",
            "-p",
            str(host.port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=10",
        ]
        private_key = Path(self.args.ssh_private_key).expanduser()
        if private_key.exists():
            command += ["-i", str(private_key)]
        return command + [f"root@{host.host}"]

    def save_state(self, state: dict):
        STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

    def load_state(self) -> dict:
        if not STATE_PATH.exists():
            raise FileNotFoundError("no migration state; run provision first")
        return json.loads(STATE_PATH.read_text())

    def host_from_state(self, state: dict) -> Host:
        return Host(state["podId"], state["name"], state["host"], int(state["port"]))

    def _env(self, name: str) -> str:
        value = os.environ.get(name) or self._zshrc_export(name)
        if not value:
            raise ValueError(f"{name} is required")
        return value

    def _zshrc_export(self, name: str) -> str:
        path = Path.home() / ".zshrc"
        if not path.exists():
            return ""
        match = re.search(rf"^export {name}=[\"']?([^\"'\n]+)", path.read_text(), re.M)
        return match.group(1).strip() if match else ""

    def _public_key(self) -> str:
        value = os.environ.get("RUNPOD_PUBLIC_KEY")
        if value:
            return value
        path = Path(os.environ.get("RUNPOD_PUBLIC_KEY_FILE", "~/.ssh/id_ed25519.pub")).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text().strip()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("command", choices=("provision", "status", "fetch", "cutover", "cleanup"))
    value.add_argument("--repo", default="DhruvBhatia0/smash-battlefield-fox")
    value.add_argument("--image", default="ubuntu:22.04")
    value.add_argument("--cpu-flavor", action="append", default=["cpu5c", "cpu3c"])
    value.add_argument("--cloud-type", default="COMMUNITY")
    value.add_argument("--disk-gb", type=int, default=0)
    value.add_argument("--wait-seconds", type=int, default=1200)
    value.add_argument("--lines", type=int, default=80)
    value.add_argument("--ssh-private-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    value.add_argument("--force", action="store_true")
    return value


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    getattr(Runner(args), args.command)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
