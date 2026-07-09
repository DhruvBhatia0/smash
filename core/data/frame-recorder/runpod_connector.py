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
    """One GPU pod owned by one frame recorder."""

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
        """Verify RunPod access and remember the video-rendering image."""
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.image = image or os.environ.get("SMASH_RUNPOD_IMAGE")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required")
        if not self.image:
            raise ValueError("SMASH_RUNPOD_IMAGE must point at the video-rendering image")

        self.name_prefix = name_prefix
        self.base_url = "https://rest.runpod.io/v1"
        self.gpu_type_id = os.environ.get("SMASH_RUNPOD_GPU_TYPE", "NVIDIA GeForce RTX 4090")
        self.cloud_type = os.environ.get("SMASH_RUNPOD_CLOUD_TYPE", "SECURE")
        self.container_registry_auth_id = os.environ.get("SMASH_RUNPOD_CONTAINER_REGISTRY_AUTH_ID")
        self.container_disk_gb = int(os.environ.get("SMASH_RUNPOD_CONTAINER_DISK_GB", "60"))
        self.volume_gb = int(os.environ.get("SMASH_RUNPOD_VOLUME_GB", "30"))
        self.volume_mount_path = os.environ.get("SMASH_RUNPOD_VOLUME_MOUNT_PATH", "/workspace")
        self.min_vcpu_per_gpu = int(os.environ.get("SMASH_RUNPOD_MIN_VCPU_PER_GPU", "6"))
        self.min_ram_per_gpu = int(os.environ.get("SMASH_RUNPOD_MIN_RAM_PER_GPU", "16"))
        self.local_iso = os.environ.get(
            "SMASH_MELEE_ISO",
            "/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso",
        )
        self.remote_iso = os.environ.get("SMASH_RUNPOD_REMOTE_ISO", "/workspace/iso/melee.iso")
        self.renderer = os.environ.get(
            "SMASH_RUNPOD_RENDERER_COMMAND",
            "/opt/slippi-renderer/render-ffv1-replay.sh",
        )
        self._reject_partial_recording_env()
        self.ssh_private_key = os.environ.get(
            "RUNPOD_SSH_PRIVATE_KEY",
            str(Path.home() / ".ssh" / "id_ed25519"),
        )
        self.public_key = self._public_key()
        self.wait_seconds = int(os.environ.get("SMASH_RUNPOD_WAIT_SECONDS", "600"))
        self.render_timeout_seconds = int(os.environ.get("SMASH_RENDER_TIMEOUT_SECONDS", "600"))
        self.create_retries = int(os.environ.get("SMASH_RUNPOD_CREATE_RETRIES", "4"))
        self.dolphin_bin = os.environ.get("SMASH_SLIPPI_DOLPHIN_BIN", "/opt/slippi/dolphin-emu-nogui")
        self.video_backend = os.environ.get("SMASH_VIDEO_BACKEND", "OGL")
        self.dolphin_cpu_core = os.environ.get("SMASH_DOLPHIN_CPU_CORE", "1")
        self.audio_backend = os.environ.get("SMASH_DOLPHIN_AUDIO_BACKEND", "Null")
        self.use_ffv1 = os.environ.get("SMASH_VIDEO_USE_FFV1", "False")
        self.dump_codec = os.environ.get("SMASH_VIDEO_DUMP_CODEC", "rawvideo")
        self.dump_format = os.environ.get("SMASH_VIDEO_DUMP_FORMAT", "avi")
        self.internal_resolution_frame_dumps = os.environ.get("SMASH_VIDEO_INTERNAL_RESOLUTION_FRAME_DUMPS", "True")
        self.efb_scale = os.environ.get("SMASH_VIDEO_EFB_SCALE", "2")
        self.bitrate_kbps = os.environ.get("SMASH_VIDEO_BITRATE_KBPS", "2500")
        self.upload_batch_size = max(1, int(os.environ.get("SMASH_UPLOAD_BATCH_SIZE", "1")))
        self.upload_retries = max(1, int(os.environ.get("SMASH_UPLOAD_RETRIES", "5")))
        self.upload_retry_seconds = float(os.environ.get("SMASH_UPLOAD_RETRY_SECONDS", "10"))

    def create_instance(self) -> RunpodInstance:
        """Create one RTX 4090 pod running the video-rendering image."""
        name = f"{self.name_prefix}-{time.time_ns()}"
        payload = {
            "name": name,
            "imageName": self.image,
            "computeType": "GPU",
            "cloudType": self.cloud_type,
            "gpuTypeIds": [self.gpu_type_id],
            "gpuTypePriority": "custom",
            "gpuCount": 1,
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": self.volume_gb,
            "volumeMountPath": self.volume_mount_path,
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "minVCPUPerGPU": self.min_vcpu_per_gpu,
            "minRAMPerGPU": self.min_ram_per_gpu,
            "dockerEntrypoint": ["/bin/bash", "-lc"],
            "dockerStartCmd": ["/opt/slippi-renderer/start-runpod-worker.sh"],
            "env": {"PUBLIC_KEY": self.public_key} if self.public_key else {},
        }
        if self.container_registry_auth_id:
            payload["containerRegistryAuthId"] = self.container_registry_auth_id
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
        """Prepare one worker for HF-backed video rendering."""
        self._ensure_hf_tools(instance)
        if not self.local_iso or not Path(self.local_iso).exists():
            raise FileNotFoundError(f"missing local ISO: {self.local_iso}")
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

    def record_video(
        self,
        instance: RunpodInstance,
        sample: SlpSample,
        hf_location: HfLocation,
    ) -> dict:
        """Download one SLP on the pod and render the complete replay to video."""
        if not hf_location.hf.token:
            raise ValueError("HF_TOKEN is required for pod-side download/upload")
        remote_dir = f"/workspace/smash-core/{sample.id}"
        payload = {
            "token": hf_location.hf.token,
            "repo": hf_location.repo,
            "source": sample.hf_reference,
            "target": hf_location.recording_path(sample),
            "workDir": remote_dir,
            "commandId": f"slp-{sample.id}",
            "renderer": self.renderer,
            "iso": self.remote_iso,
            "timeoutSeconds": self.render_timeout_seconds,
            "videoBackend": self.video_backend,
            "dolphinCpuCore": self.dolphin_cpu_core,
            "audioBackend": self.audio_backend,
            "videoEnv": {
                "SLIPPI_USE_FFV1": self.use_ffv1,
                "SLIPPI_DUMP_CODEC": self.dump_codec,
                "SLIPPI_DUMP_FORMAT": self.dump_format,
                "SLIPPI_INTERNAL_RESOLUTION_FRAME_DUMPS": self.internal_resolution_frame_dumps,
                "SLIPPI_EFB_SCALE": self.efb_scale,
                "SLIPPI_BITRATE_KBPS": self.bitrate_kbps,
                "SLIPPI_DOLPHIN_BIN": self.dolphin_bin,
            },
        }
        result = self._ssh_script(instance, self._remote_record_script(payload))
        return json.loads(result.stdout.strip().splitlines()[-1])

    def upload_recorded_videos(
        self,
        instance: RunpodInstance,
        hf_location: HfLocation,
        samples: list[SlpSample],
    ) -> dict:
        """Upload a batch of already-rendered sample video folders to HF."""
        if not samples:
            return {"samples": [], "files": 0}
        payload = {
            "token": hf_location.hf.token,
            "repo": hf_location.repo,
            "target": hf_location._join(hf_location.root, hf_location.recording_dir),
            "batchDir": f"/workspace/smash-core/upload-{time.time_ns()}",
            "jobs": [
                {
                    "sample": sample.id,
                    "workDir": f"/workspace/smash-core/{sample.id}",
                    "recordingDir": f"/workspace/smash-core/{sample.id}/recording",
                    "targetName": str(sample.id),
                }
                for sample in samples
            ],
        }
        result = self._ssh_script(instance, self._remote_upload_script(payload))
        return json.loads(result.stdout.strip().splitlines()[-1])

    def delete_instance(self, instance: RunpodInstance):
        """Delete one GPU pod."""
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
            "python3 -m pip install 'huggingface_hub>=1.0.0'",
        )

    def _remote_record_script(self, payload: dict) -> str:
        config_json = json.dumps(json.dumps(payload))
        return f"""
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download

VIDEO_GLOBS = ("framedump*.avi", "framedump*.mkv", "framedump*.mp4", "framedump*.mov", "framedump*.nut")


def slp_frame_range(path):
    data = path.read_bytes()
    if not data:
        raise ValueError(f"empty SLP: {{path}}")
    raw_pos = 0 if data[0] == 0x36 or data[0] != ord("{{") else 15
    raw_len = len(data) if raw_pos == 0 else int.from_bytes(data[raw_pos - 4:raw_pos], "big")
    if raw_len <= 0 or raw_pos + raw_len > len(data):
        raw_len = len(data) - raw_pos
    raw_end = raw_pos + raw_len
    if raw_pos == 0:
        sizes = {{0x36: 0x140, 0x37: 0x6, 0x38: 0x46, 0x39: 0x1}}
    else:
        if raw_pos + 2 > len(data) or data[raw_pos] != 0x35:
            raise ValueError("SLP is missing message-size table")
        payload_len = data[raw_pos + 1]
        sizes = {{0x35: payload_len}}
        size_bytes = data[raw_pos + 2 : raw_pos + 1 + payload_len]
        for offset in range(0, len(size_bytes), 3):
            if offset + 2 >= len(size_bytes):
                break
            sizes[size_bytes[offset]] = int.from_bytes(size_bytes[offset + 1 : offset + 3], "big")

    first_frame = None
    last_frame = None
    pos = raw_pos
    while pos < raw_end:
        command = data[pos]
        size = sizes.get(command)
        if size is None:
            break
        stop = pos + size + 1
        if stop > raw_end:
            break
        if command == 0x38 and stop - pos >= 5:
            frame = struct.unpack(">i", data[pos + 1 : pos + 5])[0]
            first_frame = frame if first_frame is None else min(first_frame, frame)
            last_frame = frame if last_frame is None else max(last_frame, frame)
        pos = stop

    if first_frame is None or last_frame is None:
        raise ValueError(f"SLP has no post-frame updates: {{path}}")
    return first_frame, last_frame


def video_files(output_dir):
    files = []
    for pattern in VIDEO_GLOBS:
        files.extend(output_dir.glob(pattern))
    return sorted(files)


def print_render_diagnostics(output_dir):
    print("[render-files]")
    for item in sorted(output_dir.glob("*")):
        try:
            print(item.name, item.stat().st_size)
        except OSError as error:
            print(item.name, str(error))
    for path in [output_dir / "render-ffv1.log", output_dir / "dolphin.log", output_dir / "manifest.json"]:
        if not path.exists():
            continue
        print("[begin]", str(path))
        data = path.read_text(errors="replace")
        print(data[-20000:])
        print("[end]", str(path))


def retry_hf(action, label):
    for attempt in range(1, 7):
        try:
            return action()
        except Exception as error:
            messages = []
            seen = set()
            current = error
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                messages.append(str(current))
                current = current.__cause__ or current.__context__
            message = "\\n".join(messages)
            if attempt == 6 or ("429" not in message and "Too Many Requests" not in message):
                raise
            sleep_seconds = 30 * attempt
            print("[hf-rate-limit]", label, "attempt", attempt, "sleep", sleep_seconds, flush=True)
            time.sleep(sleep_seconds)

config = json.loads({config_json})
token = config["token"]
work_dir = Path(config["workDir"])
download_dir = work_dir / "download"
render_dir = work_dir / "render"
recording_dir = work_dir / "recording"
slp_path = work_dir / "input.slp"
playback_path = work_dir / "playback.json"
shutil.rmtree(render_dir, ignore_errors=True)
shutil.rmtree(recording_dir, ignore_errors=True)
work_dir.mkdir(parents=True, exist_ok=True)
render_dir.mkdir(parents=True, exist_ok=True)
recording_dir.mkdir(parents=True, exist_ok=True)

downloaded = retry_hf(
    lambda: hf_hub_download(
        repo_id=config["repo"],
        repo_type="dataset",
        filename=config["source"],
        token=token,
        local_dir=str(download_dir),
    ),
    "download",
)
shutil.copyfile(downloaded, slp_path)
shutil.copyfile(slp_path, recording_dir / "input.slp")

first_frame, last_frame = slp_frame_range(slp_path)
playback = {{
    "replay": str(slp_path),
    "commandId": config["commandId"],
    "startFrame": first_frame,
    "endFrame": last_frame,
}}
playback_path.write_text(json.dumps(playback, indent=2) + "\\n")

command = [
    config["renderer"],
    "--replay-json",
    str(playback_path),
    "--output-dir",
    str(render_dir),
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
    "--no-xvfb",
]
env = os.environ.copy()
env.update({{key: str(value) for key, value in config["videoEnv"].items()}})
process = subprocess.Popen(
    command,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    start_new_session=True,
)
stdout, stderr = process.communicate()

if process.returncode:
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    print_render_diagnostics(render_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    raise SystemExit("renderer failed with exit " + str(process.returncode))

videos = video_files(render_dir)
if not videos:
    print_render_diagnostics(render_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    raise SystemExit("renderer completed but did not write a video")

probe = subprocess.run(
    ["ffprobe", "-hide_banner", "-show_format", "-show_streams", "-print_format", "json", str(videos[0])],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
if probe.returncode:
    print(probe.stderr)
    print_render_diagnostics(render_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    raise SystemExit(probe.returncode)
ffprobe = json.loads(probe.stdout)

manifest_path = render_dir / "manifest.json"
if not manifest_path.exists():
    print_render_diagnostics(render_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    raise SystemExit("renderer completed without a manifest")
manifest = json.loads(manifest_path.read_text())
current_range = manifest.get("currentFrameRange") or {{}}
if current_range.get("last") is None or int(current_range["last"]) < last_frame:
    print_render_diagnostics(render_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    raise SystemExit(f"render stopped before replay end: {{current_range.get('last')}} < {{last_frame}}")
streams = [stream for stream in ffprobe.get("streams", []) if stream.get("codec_type") == "video"]
if not streams:
    print_render_diagnostics(render_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    raise SystemExit("ffprobe found no video stream")

stored_video = recording_dir / f"video{{videos[0].suffix}}"
shutil.copyfile(videos[0], stored_video)
summary = {{
    "podWorkDir": str(work_dir),
    "uploadedTo": config["target"],
    "recordingDir": str(recording_dir),
    "source": config["source"],
    "firstFrame": first_frame,
    "lastFrame": last_frame,
    "video": str(stored_video),
    "videoBytes": stored_video.stat().st_size,
    "videoFrames": streams[0].get("nb_frames"),
    "files": len([path for path in recording_dir.rglob("*") if path.is_file()]),
}}
print(json.dumps(summary))
""".strip()

    def _remote_upload_script(self, payload: dict) -> str:
        config_json = json.dumps(json.dumps(payload))
        return f"""
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi


def retry_hf(action, label):
    for attempt in range(1, 7):
        try:
            return action()
        except Exception as error:
            messages = []
            seen = set()
            current = error
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                messages.append(str(current))
                current = current.__cause__ or current.__context__
            message = "\\n".join(messages)
            if attempt == 6 or ("429" not in message and "Too Many Requests" not in message):
                raise
            sleep_seconds = 30 * attempt
            print("[hf-rate-limit]", label, "attempt", attempt, "sleep", sleep_seconds, flush=True)
            time.sleep(sleep_seconds)

config = json.loads({config_json})
batch_dir = Path(config["batchDir"])
if batch_dir.exists():
    shutil.rmtree(batch_dir)
batch_dir.mkdir(parents=True)

uploaded = []
for job in config["jobs"]:
    source = Path(job["recordingDir"])
    if not source.exists():
        raise FileNotFoundError(str(source))
    target = batch_dir / job["targetName"]
    shutil.copytree(source, target)
    uploaded.append(job["sample"])

file_count = len([path for path in batch_dir.rglob("*") if path.is_file()])
retry_hf(
    lambda: HfApi(token=config["token"]).upload_folder(
        repo_id=config["repo"],
        repo_type="dataset",
        token=config["token"],
        folder_path=str(batch_dir),
        path_in_repo=config["target"],
    ),
    "upload",
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
        return ["rsync", "-a", "--no-owner", "--no-group", "--partial", "--inplace", "-e", shlex.join(ssh)]

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

    def _reject_partial_recording_env(self):
        disallowed = [
            "SMASH_START_FRAME",
            "SMASH_END_FRAME",
            "SMASH_MAX_FRAME_UPLOADS",
        ]
        configured = [name for name in disallowed if os.environ.get(name) not in {None, ""}]
        if configured:
            values = ", ".join(f"{name}={os.environ[name]!r}" for name in configured)
            raise ValueError(
                "Full frame recording is required; unset partial-recording environment "
                f"variables: {values}"
            )

    def _public_key(self) -> str:
        if os.environ.get("RUNPOD_PUBLIC_KEY"):
            return os.environ["RUNPOD_PUBLIC_KEY"]
        path = Path(os.environ.get("RUNPOD_PUBLIC_KEY_FILE", Path.home() / ".ssh" / "id_ed25519.pub"))
        return path.read_text().strip() if path.exists() else ""

    def _transient_error(self, error: RuntimeError) -> bool:
        text = str(error)
        return any(f"HTTP {code}" in text for code in (429, 500, 502, 503, 504)) or "timed out" in text
