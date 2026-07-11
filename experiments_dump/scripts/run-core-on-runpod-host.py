#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ISO = "/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"


@dataclass(frozen=True)
class HostPod:
    id: str
    name: str
    ssh_host: str
    ssh_port: int


class RunpodHostRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.api_key = self._env("RUNPOD_API_KEY")
        self.hf_token = self._optional_env("HF_TOKEN")
        self.worker_image = self._env("SMASH_RUNPOD_IMAGE")
        self.worker_registry_auth_id = self._optional_env("SMASH_RUNPOD_CONTAINER_REGISTRY_AUTH_ID")
        self.public_key = self._public_key()
        self.base_url = "https://rest.runpod.io/v1"
        self.host: HostPod | None = None

    def run(self) -> int:
        pod_id = self.create_host()
        try:
            self.host = self.wait_for_ssh(pod_id)
            self.bootstrap_host()
            self.sync_repo()
            self.sync_storage_config()
            self.write_remote_env()
            self.seed_remote_raw_slps()
            self.sync_iso()
            return self.run_remote_pipeline()
        finally:
            if not self.args.keep_host:
                try:
                    self.delete_pod(self.host.id if self.host else pod_id)
                except RuntimeError as error:
                    print(json.dumps({"event": "host_cleanup_failed", "error": str(error)}), flush=True)
                try:
                    self.delete_worker_pods()
                except RuntimeError as error:
                    print(json.dumps({"event": "worker_cleanup_failed", "error": str(error)}), flush=True)

    def create_host(self) -> str:
        name = f"smash-runner-host-{time.time_ns()}"
        payload = {
            "name": name,
            "imageName": self.args.host_image,
            "computeType": "CPU",
            "cloudType": self.args.cloud_type,
            "cpuFlavorIds": self.args.cpu_flavor,
            "cpuFlavorPriority": "availability",
            "vcpuCount": self.args.host_vcpus,
            "containerDiskInGb": self.args.host_disk_gb,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "dockerEntrypoint": ["/bin/bash", "-lc"],
            "dockerStartCmd": [self.host_start_command()],
            "env": {"PUBLIC_KEY": self.public_key},
        }
        for attempt in range(1, self.args.create_retries + 1):
            try:
                pod = self.request("POST", "/pods", payload)
                pod_id = str(pod.get("id") or pod.get("podId"))
                print(json.dumps({"event": "host_created", "podId": pod_id, "name": name}), flush=True)
                return pod_id
            except RuntimeError as error:
                pod_id = self.find_pod_id(name)
                if pod_id:
                    print(json.dumps({"event": "host_created", "podId": pod_id, "name": name}), flush=True)
                    return pod_id
                if attempt == self.args.create_retries or not self.transient_error(error):
                    raise
                time.sleep(min(30, 2**attempt))
        raise RuntimeError("RunPod host creation failed")

    def host_start_command(self) -> str:
        return (
            "apt-get update && "
            "apt-get install -y --no-install-recommends "
            "openssh-server openssh-client rsync python3 python3-pip git ca-certificates && "
            "mkdir -p /run/sshd /root/.ssh /workspace && chmod 700 /root/.ssh && "
            "printf '%s\\n' \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys && "
            "chmod 600 /root/.ssh/authorized_keys && "
            "/usr/sbin/sshd && sleep infinity"
        )

    def wait_for_ssh(self, pod_id: str) -> HostPod:
        deadline = time.monotonic() + self.args.wait_seconds
        while time.monotonic() < deadline:
            pod = self.request("GET", f"/pods/{pod_id}")
            host = pod.get("publicIp") or pod.get("ip")
            mappings = pod.get("portMappings") or {}
            port = mappings.get("22") or mappings.get("22/tcp")
            if isinstance(port, dict):
                port = port.get("hostPort") or port.get("port")
            if host and port:
                ready = HostPod(
                    id=pod_id,
                    name=str(pod.get("name") or ""),
                    ssh_host=str(host),
                    ssh_port=int(port),
                )
                try:
                    self.ssh(ready, "true")
                    print(json.dumps({"event": "host_ready", "podId": pod_id}), flush=True)
                    return ready
                except RuntimeError:
                    pass
            time.sleep(5)
        raise TimeoutError(f"host pod {pod_id} did not expose SSH")

    def bootstrap_host(self):
        host = self.require_host()
        commands = []
        if self.args.storage_provider == "hf":
            commands.append("python3 -m pip install 'huggingface_hub[hf_xet]>=1.0.0'")
        else:
            commands.append("apt-get update && apt-get install -y --no-install-recommends rclone zstd")
        commands.append(
            "ssh-keygen -t ed25519 -N '' -f /root/.ssh/smash_worker_key <<<y >/dev/null 2>&1 || true"
        )
        self.ssh(host, " && ".join(commands))
        print(json.dumps({"event": "host_bootstrapped", "podId": host.id}), flush=True)

    def sync_repo(self):
        host = self.require_host()
        self.ssh(host, "mkdir -p /workspace/smash/core")
        self.rsync(
            f"{ROOT}/requirements.txt",
            host,
            "/workspace/smash/requirements.txt",
        )
        self.rsync(
            f"{ROOT}/core/",
            host,
            "/workspace/smash/core/",
            extra=[
                "--delete",
                "--exclude",
                "__pycache__",
            ],
        )
        print(json.dumps({"event": "repo_synced", "podId": host.id}), flush=True)

    def sync_storage_config(self):
        if self.args.storage_provider != "gdrive":
            return
        host = self.require_host()
        config = Path(self.args.gdrive_config).expanduser()
        if not config.exists():
            raise FileNotFoundError(config)
        self.ssh(host, "mkdir -p /root/.config/rclone")
        self.rsync(str(config), host, "/root/.config/rclone/rclone.conf")
        self.ssh(host, "chmod 600 /root/.config/rclone/rclone.conf")
        print(json.dumps({"event": "gdrive_config_synced", "podId": host.id}), flush=True)

    def sync_iso(self):
        host = self.require_host()
        iso = Path(self.args.iso).expanduser()
        if not iso.exists():
            raise FileNotFoundError(f"missing ISO: {iso}")
        self.ssh(host, "mkdir -p /workspace/iso")
        self.rsync(str(iso), host, "/workspace/iso/melee.iso")
        print(json.dumps({"event": "iso_synced", "podId": host.id}), flush=True)

    def run_remote_pipeline(self) -> int:
        host = self.require_host()
        self.write_remote_env()
        command = (
            "cd /workspace/smash && "
            "set -a && . /workspace/smash/.runpod_env && set +a && "
            "export RUNPOD_PUBLIC_KEY=\"$(cat /root/.ssh/smash_worker_key.pub)\" && "
            "python3 -m core.data.frame-recorder.runner"
        )
        print(json.dumps({"event": "remote_runner_started", "podId": host.id}), flush=True)
        completed = subprocess.run(self.ssh_base(host) + [command], text=True)
        print(json.dumps({"event": "remote_runner_finished", "returncode": completed.returncode}), flush=True)
        return completed.returncode

    def seed_remote_raw_slps(self):
        if not self.args.seed_source_hf_repo:
            return
        host = self.require_host()
        command = (
            "cd /workspace/smash && "
            "set -a && . /workspace/smash/.runpod_env && set +a && "
            "python3 -m core.data.frame-recorder.hf_source_seeder"
        )
        print(json.dumps({"event": "remote_seed_started", "podId": host.id}), flush=True)
        completed = subprocess.run(self.ssh_base(host) + [command], text=True)
        print(json.dumps({"event": "remote_seed_finished", "returncode": completed.returncode}), flush=True)
        if completed.returncode:
            raise RuntimeError(f"remote seed failed with exit code {completed.returncode}")

    def write_remote_env(self):
        host = self.require_host()
        env = {
            "SMASH_STORAGE_PROVIDER": self.args.storage_provider,
            "HF_TOKEN": self.hf_token,
            "RUNPOD_API_KEY": self.api_key,
            "SMASH_RUNPOD_IMAGE": self.worker_image,
            "SMASH_HF_EXPECTED_EMAIL": self.args.hf_expected_email,
            "SMASH_HF_REPO": self.args.hf_repo,
            "SMASH_HF_ROOT": self.args.hf_root,
            "SMASH_HF_RAW_SLP_DIR": self.args.hf_raw_slp_dir,
            "SMASH_HF_PRIVATE": "1" if self.args.hf_private else "0",
            "SMASH_GDRIVE_REMOTE": self.args.gdrive_remote,
            "SMASH_GDRIVE_CONFIG": "/root/.config/rclone/rclone.conf",
            "SMASH_GDRIVE_ROOT": self.args.gdrive_root,
            "SMASH_GDRIVE_RAW_SLP_DIR": self.args.gdrive_raw_slp_dir,
            "SMASH_GDRIVE_RECORDING_DIR": self.args.gdrive_recording_dir,
            "SMASH_SAMPLE_LIMIT": str(self.args.sample_limit),
            "SMASH_WORKER_COUNT": str(self.args.worker_count),
            "SMASH_RECORDER_STARTUP_RETRY_SECONDS": str(self.args.recorder_startup_retry_seconds),
            "SMASH_RECORDER_STARTUP_MAX_ATTEMPTS": str(self.args.recorder_startup_max_attempts),
            "SMASH_RENDER_TIMEOUT_SECONDS": str(self.args.render_timeout_seconds),
            "SMASH_UPLOAD_BATCH_SIZE": str(self.args.upload_batch_size),
            "SMASH_UPLOAD_RETRIES": str(self.args.upload_retries),
            "SMASH_UPLOAD_RETRY_SECONDS": str(self.args.upload_retry_seconds),
            "SMASH_SKIP_EXISTING_PROCESSED": "1" if self.args.skip_existing_processed else "0",
            "SMASH_RUNPOD_WAIT_SECONDS": str(self.args.wait_seconds),
            "SMASH_RUNPOD_NAME_PREFIX": self.args.worker_name_prefix,
            "SMASH_RUNPOD_GPU_TYPE": self.args.worker_gpu_type,
            "SMASH_RUNPOD_CLOUD_TYPE": self.args.worker_cloud_type,
            "SMASH_RUNPOD_CLOUD_TYPES": self.args.worker_cloud_types,
            "SMASH_RUNPOD_MIN_VCPU_PER_GPU": str(self.args.worker_min_vcpu_per_gpu),
            "SMASH_RUNPOD_CONTAINER_DISK_GB": str(self.args.worker_disk_gb),
            "SMASH_RUNPOD_VOLUME_GB": str(self.args.worker_volume_gb),
            "SMASH_RUNPOD_CREATE_RETRIES": str(self.args.create_retries),
            "SMASH_MELEE_ISO": "/workspace/iso/melee.iso",
            "RUNPOD_SSH_PRIVATE_KEY": "/root/.ssh/smash_worker_key",
            "SMASH_SOURCE_HF_REPO": self.args.seed_source_hf_repo,
            "SMASH_SOURCE_HF_PREFIX": self.args.seed_source_hf_prefix,
            "SMASH_SOURCE_SAMPLE_LIMIT": str(self.args.seed_count or self.args.sample_limit),
            "SMASH_SOURCE_SEED_CONCURRENCY": str(self.args.seed_concurrency),
            "SMASH_SOURCE_SEED_BATCH_SIZE": str(self.args.seed_batch_size),
            "SMASH_SOURCE_SEED_WORK_DIR": self.args.seed_work_dir,
            "SMASH_SOURCE_DOWNLOAD_TIMEOUT_SECONDS": str(self.args.seed_download_timeout_seconds),
            "SMASH_SOURCE_DOWNLOAD_RETRIES": str(self.args.seed_download_retries),
        }
        if self.worker_registry_auth_id:
            env["SMASH_RUNPOD_CONTAINER_REGISTRY_AUTH_ID"] = self.worker_registry_auth_id
        env_text = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
        self.ssh_stdin(host, "cat > /workspace/smash/.runpod_env && chmod 600 /workspace/smash/.runpod_env", env_text)

    def request(self, method: str, path: str, payload: dict | None = None):
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read().decode()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"RunPod {method} {path} failed: HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"RunPod {method} {path} failed: {error.reason}") from error
        return json.loads(raw) if raw else {}

    def find_pod_id(self, name: str) -> str:
        try:
            pods = self.request("GET", "/pods")
        except RuntimeError:
            return ""
        if not isinstance(pods, list):
            pods = pods.get("pods") or pods.get("data") or []
        for pod in pods:
            if pod.get("name") == name:
                return str(pod.get("id") or pod.get("podId"))
        return ""

    def delete_pod(self, pod_id: str):
        try:
            self.request("DELETE", f"/pods/{pod_id}")
            print(json.dumps({"event": "host_deleted", "podId": pod_id}), flush=True)
        except RuntimeError as error:
            if "HTTP 404" not in str(error):
                raise

    def delete_worker_pods(self):
        pods = self.request("GET", "/pods")
        if not isinstance(pods, list):
            pods = pods.get("pods") or pods.get("data") or []
        deleted = []
        for pod in pods:
            if str(pod.get("name", "")).startswith(self.args.worker_name_prefix):
                pod_id = str(pod.get("id") or pod.get("podId"))
                self.request("DELETE", f"/pods/{pod_id}")
                deleted.append(pod_id)
        print(
            json.dumps(
                {
                    "event": "worker_pods_deleted",
                    "prefix": self.args.worker_name_prefix,
                    "count": len(deleted),
                }
            ),
            flush=True,
        )

    def ssh(self, host: HostPod, command: str):
        completed = subprocess.run(self.ssh_base(host) + [command], text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed.stdout

    def ssh_stdin(self, host: HostPod, command: str, stdin: str):
        completed = subprocess.run(
            self.ssh_base(host) + [command],
            input=stdin,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    def rsync(self, source: str, host: HostPod, target: str, extra: list[str] | None = None):
        command = ["rsync", "-a", "--partial", "--inplace"]
        if extra:
            command.extend(extra)
        command += ["-e", shlex.join(self.ssh_base(host)[:-1]), source, f"root@{host.ssh_host}:{target}"]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())

    def ssh_base(self, host: HostPod) -> list[str]:
        private_key = Path(self.args.ssh_private_key).expanduser()
        command = [
            "ssh",
            "-p",
            str(host.ssh_port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=10",
            "-o",
            "TCPKeepAlive=yes",
        ]
        if private_key.exists():
            command += ["-i", str(private_key)]
        return command + [f"root@{host.ssh_host}"]

    def require_host(self) -> HostPod:
        if self.host is None:
            raise RuntimeError("host has not been created")
        return self.host

    def _env(self, name: str) -> str:
        value = os.environ.get(name) or self._zshrc_export(name)
        if not value:
            raise ValueError(f"{name} is required")
        return value

    def _optional_env(self, name: str) -> str:
        return os.environ.get(name) or self._zshrc_export(name)

    def _zshrc_export(self, name: str) -> str:
        zshrc = Path.home() / ".zshrc"
        if not zshrc.exists():
            return ""
        match = re.search(rf"^export {name}=[\"']?([^\"'\n]+)", zshrc.read_text(), re.M)
        return match.group(1).strip() if match else ""

    def _public_key(self) -> str:
        key = os.environ.get("RUNPOD_PUBLIC_KEY")
        if key:
            return key
        path = Path(os.environ.get("RUNPOD_PUBLIC_KEY_FILE", "~/.ssh/id_ed25519.pub")).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"missing public key: {path}")
        return path.read_text().strip()

    def transient_error(self, error: RuntimeError) -> bool:
        text = str(error)
        return any(f"HTTP {code}" in text for code in (429, 500, 502, 503, 504)) or "timed out" in text


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--storage-provider", choices=("hf", "gdrive"), default="hf")
    arg_parser.add_argument("--hf-repo", default="")
    arg_parser.add_argument("--hf-root", default="")
    arg_parser.add_argument("--hf-raw-slp-dir", default="raw_slp")
    arg_parser.add_argument("--hf-expected-email", default="dhruv.bhatia.j@gmail.com")
    arg_parser.add_argument("--gdrive-remote", default="smash-drive")
    arg_parser.add_argument("--gdrive-config", default="~/.config/rclone/rclone.conf")
    arg_parser.add_argument("--gdrive-root", default="")
    arg_parser.add_argument("--gdrive-raw-slp-dir", default="")
    arg_parser.add_argument("--gdrive-recording-dir", default="slp_with_video")
    arg_parser.add_argument("--sample-limit", type=int, default=10)
    arg_parser.add_argument("--worker-count", type=int, default=2)
    arg_parser.add_argument("--recorder-startup-retry-seconds", type=float, default=60)
    arg_parser.add_argument("--recorder-startup-max-attempts", type=int, default=0)
    arg_parser.add_argument("--host-image", default="ubuntu:22.04")
    arg_parser.add_argument("--host-vcpus", type=int, default=8)
    arg_parser.add_argument("--host-disk-gb", type=int, default=20)
    arg_parser.add_argument("--worker-gpu-type", default="NVIDIA GeForce RTX 4090")
    arg_parser.add_argument("--worker-cloud-type", default="SECURE")
    arg_parser.add_argument("--worker-cloud-types", default="SECURE")
    arg_parser.add_argument("--worker-min-vcpu-per-gpu", type=int, default=6)
    arg_parser.add_argument("--worker-disk-gb", type=int, default=60)
    arg_parser.add_argument("--worker-volume-gb", type=int, default=100)
    arg_parser.add_argument("--cpu-flavor", action="append", default=["cpu3c"])
    arg_parser.add_argument("--cloud-type", default="COMMUNITY")
    arg_parser.add_argument("--iso", default=DEFAULT_ISO)
    arg_parser.add_argument("--ssh-private-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    arg_parser.add_argument("--wait-seconds", type=int, default=900)
    arg_parser.add_argument("--render-timeout-seconds", type=int, default=600)
    arg_parser.add_argument("--upload-batch-size", type=int, default=1)
    arg_parser.add_argument("--upload-retries", type=int, default=5)
    arg_parser.add_argument("--upload-retry-seconds", type=float, default=10)
    arg_parser.add_argument("--create-retries", type=int, default=4)
    arg_parser.add_argument("--seed-source-hf-repo", default="")
    arg_parser.add_argument("--seed-source-hf-prefix", default="")
    arg_parser.add_argument("--seed-count", type=int, default=0)
    arg_parser.add_argument("--seed-concurrency", type=int, default=16)
    arg_parser.add_argument("--seed-batch-size", type=int, default=100)
    arg_parser.add_argument("--seed-work-dir", default="/workspace/slp-seed")
    arg_parser.add_argument("--seed-download-timeout-seconds", type=int, default=60)
    arg_parser.add_argument("--seed-download-retries", type=int, default=3)
    arg_parser.add_argument("--worker-name-prefix", default="smash-core-worker")
    arg_parser.add_argument("--skip-existing-processed", action=argparse.BooleanOptionalAction, default=False)
    arg_parser.add_argument("--hf-private", action=argparse.BooleanOptionalAction, default=False)
    arg_parser.add_argument("--keep-host", action=argparse.BooleanOptionalAction, default=False)
    return arg_parser


def main(argv: list[str]) -> int:
    return RunpodHostRunner(parser().parse_args(argv)).run()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
