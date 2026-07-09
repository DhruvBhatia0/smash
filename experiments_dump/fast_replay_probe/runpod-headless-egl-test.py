#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_GPUS = [
    ("NVIDIA GeForce RTX 4090", 0.69),
    ("NVIDIA GeForce RTX 5090", 0.99),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 2.09),
]


class RunPodClient:
    def __init__(self, api_key: str, base_url: str = "https://rest.runpod.io/v1"):
        if not api_key:
            raise SystemExit("RUNPOD_API_KEY is not set")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod API {method} {path} failed: HTTP {error.code}: {detail}") from error
        return json.loads(raw) if raw else None

    def list_pods(self) -> list[dict[str, Any]]:
        payload = self.request("GET", "/pods")
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            pods = payload.get("pods") or payload.get("data") or []
            return pods if isinstance(pods, list) else []
        return []

    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.request("POST", "/pods", payload)
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected RunPod create-pod response")
        return result

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        result = self.request("GET", f"/pods/{pod_id}")
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected RunPod pod response for {pod_id}")
        return result

    def delete_pod(self, pod_id: str) -> None:
        self.request("DELETE", f"/pods/{pod_id}")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            if key.lower() in {"env"} and isinstance(nested, dict):
                out[key] = {env_key: "<set>" for env_key in nested}
            elif "key" in key.lower() and isinstance(nested, str):
                out[key] = "<set>"
            else:
                out[key] = scrub(nested)
        return out
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd, text=True, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {cmd[0]}")
    return completed


def capture(cmd: list[str], *, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"command failed with exit {completed.returncode}: {cmd[0]}")
    return completed


def ssh_base(host: str, port: int, key_path: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "root@" + host,
    ]


def rsync_base(host: str, port: int, key_path: Path) -> list[str]:
    ssh_cmd = " ".join(shlex.quote(part) for part in ssh_base(host, port, key_path)[:-1])
    return ["rsync", "-a", "--partial", "--inplace", "-e", ssh_cmd]


def pod_ssh_endpoint(pod: dict[str, Any]) -> tuple[str | None, int | None]:
    host = pod.get("publicIp") or pod.get("public_ip")
    mappings = pod.get("portMappings") or pod.get("port_mappings") or {}
    port = mappings.get("22") or mappings.get(22)
    if not host or not port:
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") if isinstance(runtime, dict) else None
        if isinstance(ports, list):
            for item in ports:
                if not isinstance(item, dict):
                    continue
                private_port = item.get("privatePort") or item.get("private_port")
                if str(private_port) == "22":
                    host = host or item.get("ip") or item.get("host")
                    port = port or item.get("publicPort") or item.get("public_port")
    return (str(host) if host else None, int(port) if port else None)


def wait_for_ssh(
    client: RunPodClient,
    pod_id: str,
    key_path: Path,
    *,
    timeout_seconds: int,
    out_path: Path,
) -> tuple[str, int, dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    last_pod: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last_pod = client.get_pod(pod_id)
        host, port = pod_ssh_endpoint(last_pod)
        if host and port:
            test = capture(
                ssh_base(host, port, key_path) + ["true"],
                timeout=15,
                check=False,
            )
            if test.returncode == 0:
                write_json(out_path, {"podId": pod_id, "host": host, "port": port, "pod": scrub(last_pod)})
                return host, port, last_pod
        time.sleep(5)
    write_json(out_path, {"podId": pod_id, "pod": scrub(last_pod), "error": "timed out waiting for SSH"})
    raise TimeoutError(f"timed out waiting for SSH on pod {pod_id}")


def create_cheapest_available_pod(
    client: RunPodClient,
    *,
    image: str,
    name: str,
    public_key: str,
    output_dir: Path,
) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    start_cmd = (
        "mkdir -p /run/sshd /root/.ssh && "
        "chmod 700 /root/.ssh && "
        "if [ -n \"${PUBLIC_KEY:-}\" ]; then "
        "printf '%s\\n' \"$PUBLIC_KEY\" > /root/.ssh/authorized_keys && "
        "chmod 600 /root/.ssh/authorized_keys; "
        "fi && "
        "/usr/sbin/sshd && "
        "sleep infinity"
    )
    for gpu_name, published_price in DEFAULT_GPUS:
        payload = {
            "name": name,
            "imageName": image,
            "computeType": "GPU",
            "cloudType": "COMMUNITY",
            "gpuTypeIds": [gpu_name],
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "containerDiskInGb": 30,
            "volumeInGb": 20,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "minVCPUPerGPU": 6,
            "minRAMPerGPU": 16,
            "dockerEntrypoint": ["/bin/bash", "-lc"],
            "dockerStartCmd": [start_cmd],
            "env": {
                "PUBLIC_KEY": public_key,
                "SMASH_FRAME_QUEUE_RUN_ID": name,
            },
        }
        try:
            pod = client.create_pod(payload)
        except RuntimeError as error:
            errors.append(
                {
                    "gpu": gpu_name,
                    "publishedPricePerHr": published_price,
                    "error": str(error),
                }
            )
            continue
        write_json(
            output_dir / "runpod-headless-egl-pod.json",
            {
                "createdAt": utc_now(),
                "requestedGpu": gpu_name,
                "publishedPricePerHr": published_price,
                "image": image,
                "payload": scrub(payload),
                "pod": scrub(pod),
                "errorsBeforeSuccess": errors,
            },
        )
        return str(pod["id"]), gpu_name, pod, errors
    write_json(output_dir / "runpod-headless-egl-errors.json", {"errors": errors})
    raise RuntimeError("RunPod could not allocate any allowed GPU")


def render_remote(
    *,
    host: str,
    port: int,
    key_path: Path,
    output_dir: Path,
    iso_path: Path,
    slp_path: Path,
    start_frame: int,
    end_frame: int,
    timeout_seconds: int,
) -> None:
    remote_root = "/workspace/headless-egl-probe"
    remote_iso = f"{remote_root}/iso/melee.iso"
    remote_slp = f"{remote_root}/replay/realtimeTest.slp"
    remote_playback = f"{remote_root}/playback.json"
    remote_out = f"{remote_root}/out"
    local_playback = output_dir / "playback.remote.json"
    write_json(
        local_playback,
        {
            "replay": remote_slp,
            "startFrame": start_frame,
            "endFrame": end_frame,
            "commandId": "headless-egl-runpod-full",
        },
    )

    run(ssh_base(host, port, key_path) + [f"rm -rf {remote_root} && mkdir -p {remote_root}/iso {remote_root}/replay {remote_out}"])
    run(rsync_base(host, port, key_path) + [str(iso_path), f"root@{host}:{remote_iso}"])
    run(rsync_base(host, port, key_path) + [str(slp_path), f"root@{host}:{remote_slp}"])
    run(rsync_base(host, port, key_path) + [str(local_playback), f"root@{host}:{remote_playback}"])

    remote_script = f"""\
set -euo pipefail
cd {remote_root}
mkdir -p out
uname -a > out/uname.txt
nvidia-smi > out/nvidia-smi.txt 2>&1 || true
eglinfo -B > out/eglinfo.txt 2>&1 || true
/opt/slippi/dolphin-emu-nogui --version > out/dolphin-version.txt 2>&1 || true
(nvidia-smi dmon -s pucvmet -d 1 -o DT > out/nvidia-dmon.log 2>&1 & echo $! > out/nvidia-dmon.pid) || true
start_ns="$(date +%s%N)"
set +e
/opt/slippi-renderer/render-ffv1-replay.sh \\
  --replay-json {remote_playback} \\
  --output-dir {remote_out} \\
  --iso {remote_iso} \\
  --timeout-seconds {timeout_seconds} \\
  --video-backend OGL \\
  --cpu-core 1 \\
  --audio-backend Null \\
  --no-xvfb > out/wrapper.stdout 2> out/wrapper.stderr
status=$?
set -e
end_ns="$(date +%s%N)"
if [ -f out/nvidia-dmon.pid ]; then
  kill "$(cat out/nvidia-dmon.pid)" 2>/dev/null || true
fi
python3 - "$start_ns" "$end_ns" "$status" <<'PY'
import json
import sys
from pathlib import Path
start_ns = int(sys.argv[1])
end_ns = int(sys.argv[2])
status = int(sys.argv[3])
out = Path("out")
manifest_path = out / "manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {{}}
videos = manifest.get("videos") or []
ffprobe = None
if videos:
    import subprocess
    video = videos[0]["path"]
    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-show_format", "-show_streams", "-print_format", "json", video],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.stdout:
        try:
            ffprobe = json.loads(probe.stdout)
        except Exception:
            ffprobe = {{"parseError": probe.stdout[:5000], "stderr": probe.stderr[:5000]}}
summary = {{
    "status": status,
    "elapsedSeconds": (end_ns - start_ns) / 1_000_000_000,
    "manifest": manifest,
    "ffprobe": ffprobe,
}}
(out / "remote-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
PY
exit "$status"
"""
    escaped = "cat > /tmp/headless-egl-run.sh <<'EOS'\n" + remote_script + "\nEOS\nbash /tmp/headless-egl-run.sh"
    run(ssh_base(host, port, key_path) + [escaped], check=False)
    run(["mkdir", "-p", str(output_dir)])
    run(rsync_base(host, port, key_path) + [f"root@{host}:{remote_out}/", str(output_dir) + "/"], check=False)


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(description="Run the headless EGL Slippi video probe on a GPU RunPod pod.")
    arg_parser.add_argument("--image", required=True, help="Published Docker image tag for the headless-EGL renderer.")
    arg_parser.add_argument("--output-dir", type=Path, default=Path("runs/runpod-headless-egl"))
    arg_parser.add_argument("--iso", type=Path, default=Path.home() / "Downloads" / "Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso")
    arg_parser.add_argument("--slp", type=Path, default=Path("../replays/realtimeTest.slp"))
    arg_parser.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh" / "id_ed25519")
    arg_parser.add_argument("--ssh-public-key", type=Path, default=Path.home() / ".ssh" / "id_ed25519.pub")
    arg_parser.add_argument("--wait-seconds", type=int, default=900)
    arg_parser.add_argument("--render-timeout-seconds", type=int, default=90)
    arg_parser.add_argument("--start-frame", type=int, default=-123)
    arg_parser.add_argument("--end-frame", type=int, default=2182)
    arg_parser.add_argument("--keep-pod", action="store_true", help="Do not delete the pod after the run.")
    return arg_parser


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    iso_path = args.iso.expanduser().resolve()
    slp_path = args.slp.expanduser().resolve()
    key_path = args.ssh_key.expanduser().resolve()
    public_key_path = args.ssh_public_key.expanduser().resolve()
    for path in [iso_path, slp_path, key_path, public_key_path]:
        if not path.exists():
            raise SystemExit(f"missing required path: {path}")

    client = RunPodClient(os.environ.get("RUNPOD_API_KEY", ""))
    public_key = public_key_path.read_text().strip()
    name = f"smash-gpu-headless-egl-{int(time.time())}"
    pod_id = ""
    exit_code = 1
    try:
        pod_id, gpu_name, pod, errors = create_cheapest_available_pod(
            client,
            image=args.image,
            name=name,
            public_key=public_key,
            output_dir=output_dir,
        )
        host, port, ready_pod = wait_for_ssh(
            client,
            pod_id,
            key_path,
            timeout_seconds=args.wait_seconds,
            out_path=output_dir / "runpod-headless-egl-ssh.json",
        )
        render_remote(
            host=host,
            port=port,
            key_path=key_path,
            output_dir=output_dir,
            iso_path=iso_path,
            slp_path=slp_path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            timeout_seconds=args.render_timeout_seconds,
        )
        summary_path = output_dir / "remote-summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            exit_code = int(summary.get("status", 1))
        else:
            exit_code = 1
    finally:
        if pod_id and not args.keep_pod:
            try:
                client.delete_pod(pod_id)
                write_json(output_dir / "runpod-headless-egl-cleanup.json", {"deletedPodId": pod_id, "deletedAt": utc_now()})
            except Exception as error:
                write_json(output_dir / "runpod-headless-egl-cleanup.json", {"podId": pod_id, "error": str(error)})
    write_json(
        output_dir / "runpod-headless-egl-open-pods.json",
        {"checkedAt": utc_now(), "pods": [scrub(pod) for pod in client.list_pods()]},
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
