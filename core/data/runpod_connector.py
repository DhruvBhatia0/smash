import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .models import HfLocation, SlpSample


@dataclass(frozen=True)
class RunpodInstance:
    """One CPU pod owned by one frame recorder."""

    id: str
    name: str
    status: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None


class RunpodConnector:
    def __init__(
        self,
        image: str | None = None,
        api_key: str | None = None,
        name_prefix: str = "smash-core-worker",
    ):
        """Verify RunPod access and remember the frame-recording image."""
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.image = image or os.environ.get("SMASH_RUNPOD_IMAGE")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required")
        if not self.image:
            raise ValueError("SMASH_RUNPOD_IMAGE must point at the frame-recording image")

        self.name_prefix = name_prefix
        self.base_url = "https://rest.runpod.io/v1"
        self.cpu_flavor_ids = self._list_env("SMASH_RUNPOD_CPU_FLAVORS", ["cpu3c"])
        self.cloud_type = os.environ.get("SMASH_RUNPOD_CLOUD_TYPE", "COMMUNITY")
        self.vcpu_count = int(os.environ.get("SMASH_RUNPOD_VCPU_COUNT", "1"))
        self.container_disk_gb = int(os.environ.get("SMASH_RUNPOD_CONTAINER_DISK_GB", "10"))
        self.local_iso = os.environ.get(
            "SMASH_MELEE_ISO",
            "/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso",
        )
        self.remote_iso = os.environ.get("SMASH_RUNPOD_REMOTE_ISO", "/workspace/iso/melee.iso")
        self.renderer = os.environ.get(
            "SMASH_RUNPOD_RENDERER_COMMAND",
            "/opt/slippi-renderer/render-replay.sh",
        )
        self.ssh_private_key = os.environ.get(
            "RUNPOD_SSH_PRIVATE_KEY",
            str(Path.home() / ".ssh" / "id_ed25519"),
        )
        self.public_key = self._public_key()
        self.wait_seconds = int(os.environ.get("SMASH_RUNPOD_WAIT_SECONDS", "600"))
        self.timeout_seconds = int(os.environ.get("SMASH_RENDER_TIMEOUT_SECONDS", "120"))
        self.create_retries = int(os.environ.get("SMASH_RUNPOD_CREATE_RETRIES", "4"))
        self.video_backend = os.environ.get("SMASH_VIDEO_BACKEND", "OGL")
        self.dolphin_cpu_core = os.environ.get("SMASH_DOLPHIN_CPU_CORE", "1")
        self.audio_backend = os.environ.get("SMASH_DOLPHIN_AUDIO_BACKEND", "Null")
        self.start_frame = self._optional_int("SMASH_START_FRAME")
        self.end_frame = self._optional_int("SMASH_END_FRAME")
        self.max_frame_uploads = self._optional_int("SMASH_MAX_FRAME_UPLOADS")
        self.frame_upload_batch_size = max(1, int(os.environ.get("SMASH_FRAME_UPLOAD_BATCH_SIZE", "1")))

    def create_instance(self) -> RunpodInstance:
        """Create one CPU pod running the frame-recording image."""
        name = f"{self.name_prefix}-{time.time_ns()}"
        payload = {
            "name": name,
            "imageName": self.image,
            "computeType": "CPU",
            "cloudType": self.cloud_type,
            "cpuFlavorIds": self.cpu_flavor_ids,
            "cpuFlavorPriority": "availability",
            "vcpuCount": self.vcpu_count,
            "containerDiskInGb": self.container_disk_gb,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "dockerEntrypoint": ["/bin/bash", "-lc"],
            "dockerStartCmd": ["/opt/slippi-renderer/start-runpod-worker.sh"],
            "env": {"PUBLIC_KEY": self.public_key} if self.public_key else {},
        }
        for attempt in range(1, self.create_retries + 1):
            try:
                return self._instance(self._request("POST", "/pods", payload))
            except RuntimeError as error:
                created = self._find_instance_by_name(name)
                if created:
                    return created
                if attempt == self.create_retries or not self._transient_error(error):
                    raise
                time.sleep(min(30, 2**attempt))
        raise RuntimeError("RunPod pod creation failed")

    def attach_instance(self, pod_id: str) -> RunpodInstance:
        """Attach to an existing pod by id."""
        return self._instance(self._request("GET", f"/pods/{pod_id}"))

    def wait_for_ssh(self, instance: RunpodInstance) -> RunpodInstance:
        """Wait until RunPod exposes the pod's SSH endpoint."""
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            pod = self._request("GET", f"/pods/{instance.id}")
            ready = self._instance(pod)
            if ready.ssh_host and ready.ssh_port:
                try:
                    self._ssh(ready, "true")
                    return ready
                except RuntimeError:
                    pass
            time.sleep(5)
        raise TimeoutError(f"RunPod pod {instance.id} did not expose SSH")

    def prepare_instance(self, instance: RunpodInstance):
        """Prepare one worker for HF-backed frame recording."""
        self._ensure_hf_tools(instance)
        if not self.local_iso or not Path(self.local_iso).exists():
            return {"status": "skipped", "reason": "local ISO not found", "remoteIso": self.remote_iso}
        local_size = Path(self.local_iso).expanduser().stat().st_size
        remote_size = self._ssh(
            instance,
            f"stat -c%s {shlex.quote(self.remote_iso)} 2>/dev/null || true",
        ).stdout.strip()
        if remote_size == str(local_size):
            return {"status": "exists", "remoteIso": self.remote_iso}
        self._ssh(instance, f"mkdir -p {shlex.quote(str(Path(self.remote_iso).parent))}")
        self._rsync(instance, str(Path(self.local_iso).expanduser()), self.remote_iso)
        return {"status": "uploaded", "remoteIso": self.remote_iso}

    def record_frames(
        self,
        instance: RunpodInstance,
        sample: SlpSample,
        hf_location: HfLocation,
    ) -> dict:
        """Download one SLP on the pod, render frames, and upload them to HF."""
        if not hf_location.hf.token:
            raise ValueError("HF_TOKEN is required for pod-side download/upload")
        remote_dir = f"/workspace/smash-core/{sample.id}"
        payload = {
            "token": hf_location.hf.token,
            "repo": hf_location.repo,
            "source": sample.hf_reference,
            "target": hf_location.framed_slp_path(sample),
            "workDir": remote_dir,
            "playback": self._playback(f"{remote_dir}/input.slp", sample),
            "renderer": self.renderer,
            "iso": self.remote_iso,
            "timeoutSeconds": self.timeout_seconds,
            "videoBackend": self.video_backend,
            "dolphinCpuCore": self.dolphin_cpu_core,
            "audioBackend": self.audio_backend,
            "maxFrameUploads": self.max_frame_uploads,
            "uploadImmediately": False,
        }
        result = self._ssh_script(instance, self._remote_record_script(payload))
        return json.loads(result.stdout.strip().splitlines()[-1])

    def upload_recorded_frames(
        self,
        instance: RunpodInstance,
        hf_location: HfLocation,
        samples: list[SlpSample],
    ) -> dict:
        """Upload a batch of already-rendered sample folders to HF."""
        if not samples:
            return {"samples": [], "files": 0}
        payload = {
            "token": hf_location.hf.token,
            "repo": hf_location.repo,
            "target": hf_location._join(hf_location.root, hf_location.framed_slp_dir),
            "batchDir": f"/workspace/smash-core/upload-{time.time_ns()}",
            "jobs": [
                {
                    "sample": sample.id,
                    "workDir": f"/workspace/smash-core/{sample.id}",
                    "framesDir": f"/workspace/smash-core/{sample.id}/frames",
                    "targetName": str(sample.id),
                }
                for sample in samples
            ],
        }
        result = self._ssh_script(instance, self._remote_upload_script(payload))
        return json.loads(result.stdout.strip().splitlines()[-1])

    def delete_instance(self, instance: RunpodInstance):
        """Delete one CPU pod."""
        try:
            return self._request("DELETE", f"/pods/{instance.id}")
        except RuntimeError as error:
            if "HTTP 404" in str(error):
                return {"status": "missing", "id": instance.id}
            raise

    def delete_all_instances(self):
        """Delete every pod owned by this connector prefix."""
        deleted = []
        for pod in self._pods():
            if str(pod.get("name", "")).startswith(self.name_prefix):
                instance = self._instance(pod)
                self.delete_instance(instance)
                deleted.append(instance.id)
        return deleted

    def _ensure_hf_tools(self, instance: RunpodInstance):
        check = self._ssh(instance, "python3 - <<'PY'\nimport huggingface_hub\nPY", check=False)
        if check.returncode == 0:
            return
        self._ssh(
            instance,
            "apt-get update && "
            "apt-get install -y --no-install-recommends python3-pip && "
            "python3 -m pip install 'huggingface_hub[hf_xet]>=1.0.0'",
        )

    def _remote_record_script(self, payload: dict) -> str:
        config_json = json.dumps(json.dumps(payload))
        return f"""
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

config = json.loads({config_json})
token = config["token"]
work_dir = Path(config["workDir"])
download_dir = work_dir / "download"
frames_dir = work_dir / "frames"
slp_path = work_dir / "input.slp"
playback_path = work_dir / "playback.json"
work_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)

downloaded = hf_hub_download(
    repo_id=config["repo"],
    repo_type="dataset",
    filename=config["source"],
    token=token,
    local_dir=str(download_dir),
)
shutil.copyfile(downloaded, slp_path)

playback = config["playback"]
playback["replay"] = str(slp_path)
playback_path.write_text(json.dumps(playback, indent=2) + "\\n")

command = [
    config["renderer"],
    "--replay-json",
    str(playback_path),
    "--output-dir",
    str(frames_dir),
    "--iso",
    config["iso"],
    "--timeout-seconds",
    str(config["timeoutSeconds"]),
    "--video-backend",
    config["videoBackend"],
    "--cpu-core",
    str(config["dolphinCpuCore"]),
    "--audio-backend",
    config["audioBackend"],
]
render = subprocess.run(command, text=True, capture_output=True)
if render.returncode:
    print(render.stdout)
    print(render.stderr)
    raise SystemExit(render.returncode)

max_frame_uploads = config.get("maxFrameUploads")
if max_frame_uploads:
    frames = sorted(
        frames_dir.glob("framedump_*.png"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    for path in frames[int(max_frame_uploads):]:
        path.unlink()

if config.get("uploadImmediately"):
    HfApi(token=token).upload_folder(
        repo_id=config["repo"],
        repo_type="dataset",
        token=token,
        folder_path=str(frames_dir),
        path_in_repo=config["target"],
    )
print(json.dumps({{
    "podWorkDir": str(work_dir),
    "uploadedTo": config["target"],
    "framesDir": str(frames_dir),
    "frameCount": len(list(frames_dir.glob("framedump_*.png"))),
    "files": len([path for path in frames_dir.rglob("*") if path.is_file()]),
}}))
""".strip()

    def _remote_upload_script(self, payload: dict) -> str:
        config_json = json.dumps(json.dumps(payload))
        return f"""
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

config = json.loads({config_json})
batch_dir = Path(config["batchDir"])
if batch_dir.exists():
    shutil.rmtree(batch_dir)
batch_dir.mkdir(parents=True)

uploaded = []
for job in config["jobs"]:
    source = Path(job["framesDir"])
    if not source.exists():
        raise FileNotFoundError(str(source))
    target = batch_dir / job["targetName"]
    shutil.copytree(source, target)
    uploaded.append(job["sample"])

file_count = len([path for path in batch_dir.rglob("*") if path.is_file()])
HfApi(token=config["token"]).upload_folder(
    repo_id=config["repo"],
    repo_type="dataset",
    token=config["token"],
    folder_path=str(batch_dir),
    path_in_repo=config["target"],
)

for job in config["jobs"]:
    shutil.rmtree(job["workDir"], ignore_errors=True)
shutil.rmtree(batch_dir, ignore_errors=True)
print(json.dumps({{
    "uploadedTo": config["target"],
    "samples": uploaded,
    "files": file_count,
}}))
""".strip()

    def _ssh_script(self, instance: RunpodInstance, script: str):
        completed = subprocess.run(
            self._ssh_base(instance) + ["python3 -"],
            input=script,
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed

    def _ssh(self, instance: RunpodInstance, command: str, check: bool = True):
        completed = subprocess.run(
            self._ssh_base(instance) + [command],
            text=True,
            capture_output=True,
        )
        if check and completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed

    def _rsync(
        self,
        instance: RunpodInstance,
        source: str,
        target: str,
        from_remote: bool = False,
        check: bool = True,
    ):
        if from_remote:
            command = self._rsync_base(instance) + [f"root@{instance.ssh_host}:{source}", target]
        else:
            command = self._rsync_base(instance) + [source, f"root@{instance.ssh_host}:{target}"]
        completed = subprocess.run(command, text=True, capture_output=True)
        if check and completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        return completed

    def _ssh_base(self, instance: RunpodInstance) -> list[str]:
        if not instance.ssh_host or not instance.ssh_port:
            raise ValueError("RunPod instance is missing SSH endpoint")
        command = [
            "ssh",
            "-p",
            str(instance.ssh_port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
        ]
        if self.ssh_private_key and Path(self.ssh_private_key).exists():
            command += ["-i", self.ssh_private_key]
        return command + [f"root@{instance.ssh_host}"]

    def _rsync_base(self, instance: RunpodInstance) -> list[str]:
        ssh = self._ssh_base(instance)[:-1]
        return ["rsync", "-a", "--partial", "--inplace", "-e", shlex.join(ssh)]

    def _request(self, method: str, path: str, payload: dict | None = None):
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

    def _pods(self) -> list[dict]:
        pods = self._request("GET", "/pods")
        if isinstance(pods, list):
            return pods
        return pods.get("pods") or pods.get("data") or []

    def _find_instance_by_name(self, name: str) -> RunpodInstance | None:
        try:
            pods = self._pods()
        except RuntimeError:
            return None
        for pod in pods:
            if pod.get("name") == name:
                return self._instance(pod)
        return None

    def _instance(self, pod: dict) -> RunpodInstance:
        mappings = pod.get("portMappings") or {}
        ssh_port = mappings.get("22") or mappings.get("22/tcp")
        if isinstance(ssh_port, dict):
            ssh_port = ssh_port.get("hostPort") or ssh_port.get("port")
        return RunpodInstance(
            id=str(pod.get("id") or pod.get("podId")),
            name=str(pod.get("name") or ""),
            status=pod.get("status") or pod.get("desiredStatus"),
            ssh_host=pod.get("publicIp") or pod.get("ip"),
            ssh_port=int(ssh_port) if ssh_port else None,
        )

    def _playback(self, replay_path: str, sample: SlpSample) -> dict:
        payload = {"replay": replay_path, "commandId": f"slp-{sample.id}"}
        if self.start_frame is not None:
            payload["startFrame"] = self.start_frame
        if self.end_frame is not None:
            payload["endFrame"] = self.end_frame
        return payload

    def _public_key(self) -> str:
        if os.environ.get("RUNPOD_PUBLIC_KEY"):
            return os.environ["RUNPOD_PUBLIC_KEY"]
        path = Path(os.environ.get("RUNPOD_PUBLIC_KEY_FILE", Path.home() / ".ssh" / "id_ed25519.pub"))
        return path.read_text().strip() if path.exists() else ""

    def _optional_int(self, name: str) -> int | None:
        value = os.environ.get(name)
        return int(value) if value not in {None, ""} else None

    def _list_env(self, name: str, default: list[str]) -> list[str]:
        value = os.environ.get(name)
        return [part.strip() for part in value.split(",") if part.strip()] if value else default

    def _transient_error(self, error: RuntimeError) -> bool:
        text = str(error)
        return any(f"HTTP {code}" in text for code in (429, 500, 502, 503, 504)) or "timed out" in text
