from __future__ import annotations

import json
import os
import shlex
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .commands import CommandRunner
from .jobs import FrameRenderJob, write_json


class RuntimeProvisioningError(RuntimeError):
    pass


class EmulatorRuntime(ABC):
    def plan(self, job: FrameRenderJob) -> dict[str, Any]:
        job.ensure_dirs()
        write_json(job.playback_json_path, job.playback_config())
        plan = {
            "runtime": type(self).__name__,
            "status": "planned",
            "playbackJson": str(job.playback_json_path),
            "rawFrameDir": str(job.raw_frame_dir),
        }
        write_json(job.job_dir / "render-plan.json", plan)
        return plan

    @abstractmethod
    def render(self, job: FrameRenderJob) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class PlanningRuntime(EmulatorRuntime):
    def __init__(self, *, runtime_name: str, reason: str):
        self.runtime_name = runtime_name
        self.reason = reason

    def render(self, job: FrameRenderJob) -> dict[str, Any]:
        return self.plan(job)

    def plan(self, job: FrameRenderJob) -> dict[str, Any]:
        job.ensure_dirs()
        write_json(job.playback_json_path, job.playback_config())
        plan = {
            "runtime": self.runtime_name,
            "status": "planned",
            "reason": self.reason,
            "playbackJson": str(job.playback_json_path),
            "rawFrameDir": str(job.raw_frame_dir),
        }
        write_json(job.job_dir / "render-plan.json", plan)
        return plan


class LocalMacPlaybackRuntime(EmulatorRuntime):
    def __init__(
        self,
        *,
        root_dir: Path,
        consumer_id: int,
        timeout_seconds: int,
        playback_app: str | None,
        iso_path: str | None,
        allow_parallel: bool,
    ):
        self.root_dir = root_dir
        self.consumer_id = consumer_id
        self.timeout_seconds = timeout_seconds
        self.playback_app = playback_app
        self.iso_path = iso_path
        self.allow_parallel = allow_parallel
        self.runner = CommandRunner(cwd=root_dir)

    def render(self, job: FrameRenderJob) -> dict[str, Any]:
        job.ensure_dirs()
        write_json(job.playback_json_path, job.playback_config())
        env = os.environ.copy()
        env["USER_DIR"] = str(job.job_dir / "dolphin-user")
        env["FRAME_OUTPUT_DIR"] = str(job.raw_frame_dir)
        env["RUN_LOG"] = str(job.logs_dir / "render-replay-debug.log")
        env["PID_FILE"] = str(job.logs_dir / "render-replay-debug.pid")
        env["TIMEOUT_SECONDS"] = str(self.timeout_seconds)
        env["KILL_EXISTING_DOLPHIN"] = "0" if self.allow_parallel else "1"
        if self.playback_app:
            env["PLAYBACK_APP"] = self.playback_app
        if self.iso_path:
            env["ISO"] = self.iso_path

        result = self.runner.run(
            ["bash", "scripts/render-replay-debug.sh", str(job.playback_json_path)],
            env=env,
            check=True,
            timeout=self.timeout_seconds + 20,
        )
        return {
            "runtime": "local-macos",
            "status": "rendered",
            "consumerId": self.consumer_id,
            "rawFrameDir": str(job.raw_frame_dir),
            "command": result.to_json(),
        }


class DockerEmulatorRuntime(EmulatorRuntime):
    def __init__(
        self,
        *,
        root_dir: Path,
        run_id: str,
        consumer_id: int,
        image: str,
        iso_path: str | None,
        renderer_command: str,
        timeout_seconds: int,
        video_backend: str,
        dolphin_cpu_core: int,
        dolphin_audio_backend: str,
        dry_run: bool,
    ):
        self.root_dir = root_dir
        self.run_id = run_id
        self.consumer_id = consumer_id
        self.image = image
        self.iso_path = iso_path
        self.renderer_command = renderer_command
        self.timeout_seconds = timeout_seconds
        self.video_backend = video_backend
        self.dolphin_cpu_core = dolphin_cpu_core
        self.dolphin_audio_backend = dolphin_audio_backend
        self.dry_run = dry_run
        self.container_name = f"smash-frame-worker-{run_id}-{consumer_id}"
        self.runner = CommandRunner(cwd=root_dir)
        self.started = False
        self.provision()

    def provision(self) -> None:
        if self.dry_run:
            return
        if not self.image:
            raise RuntimeProvisioningError("--docker-image is required for docker runtime")
        info = self.runner.run(["docker", "info"], check=False)
        if info.returncode != 0:
            raise RuntimeProvisioningError(
                "Docker daemon is not reachable. Start Docker Desktop or use --runtime runpod."
            )

        command = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            self.container_name,
            "--entrypoint",
            "/bin/bash",
            "--cpus",
            "1",
            "-v",
            f"{self.root_dir}:/workspace",
        ]
        if self.iso_path:
            command.extend(["-v", f"{Path(self.iso_path).resolve()}:/iso/melee.iso:ro"])
        command.extend([self.image, "-lc", "sleep infinity"])
        self.runner.run(command, check=True)
        self.started = True

    def render(self, job: FrameRenderJob) -> dict[str, Any]:
        job.ensure_dirs()
        container_replay = f"/workspace/{job.relative_slp_path}"
        container_playback = f"/workspace/{job.playback_json_path.relative_to(self.root_dir)}"
        container_raw = f"/workspace/{job.raw_frame_dir.relative_to(self.root_dir)}"
        write_json(job.playback_json_path, job.playback_config(container_replay))

        command = [
            "docker",
            "exec",
            self.container_name,
            self.renderer_command,
            "--replay-json",
            container_playback,
            "--output-dir",
            container_raw,
            "--timeout-seconds",
            str(self.timeout_seconds),
            "--video-backend",
            self.video_backend,
            "--cpu-core",
            str(self.dolphin_cpu_core),
            "--audio-backend",
            self.dolphin_audio_backend,
        ]
        if self.iso_path:
            command.extend(["--iso", "/iso/melee.iso"])

        if self.dry_run:
            return self.plan(job)

        result = self.runner.run(command, check=True)
        return {
            "runtime": "docker",
            "status": "rendered",
            "containerName": self.container_name,
            "rawFrameDir": str(job.raw_frame_dir),
            "command": result.to_json(),
        }

    def plan(self, job: FrameRenderJob) -> dict[str, Any]:
        job.ensure_dirs()
        container_replay = f"/workspace/{job.relative_slp_path}"
        container_playback = f"/workspace/{job.playback_json_path.relative_to(self.root_dir)}"
        container_raw = f"/workspace/{job.raw_frame_dir.relative_to(self.root_dir)}"
        write_json(job.playback_json_path, job.playback_config(container_replay))
        command = [
            "docker",
            "exec",
            self.container_name,
            self.renderer_command,
            "--replay-json",
            container_playback,
            "--output-dir",
            container_raw,
            "--timeout-seconds",
            str(self.timeout_seconds),
            "--video-backend",
            self.video_backend,
            "--cpu-core",
            str(self.dolphin_cpu_core),
            "--audio-backend",
            self.dolphin_audio_backend,
        ]
        if self.iso_path:
            command.extend(["--iso", "/iso/melee.iso"])
        plan = {
            "runtime": "docker",
            "status": "planned",
            "containerName": self.container_name,
            "image": self.image,
            "command": command,
            "renderer": {
                "timeoutSeconds": self.timeout_seconds,
                "videoBackend": self.video_backend,
                "dolphinCpuCore": self.dolphin_cpu_core,
                "audioBackend": self.dolphin_audio_backend,
            },
            "rawFrameDir": str(job.raw_frame_dir),
        }
        write_json(job.job_dir / "docker-render-plan.json", plan)
        return plan

    def close(self) -> None:
        if self.started:
            self.runner.run(["docker", "stop", self.container_name], check=False)
            self.started = False


class RunPodRestClient:
    def __init__(self, *, api_key: str, base_url: str = "https://rest.runpod.io/v1"):
        if not api_key:
            raise RuntimeProvisioningError("RUNPOD_API_KEY is required for RunPod runtime")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        body = None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeProvisioningError(
                f"RunPod API {method} {path} failed: HTTP {error.code}: {detail}"
            ) from error
        if not raw:
            return None
        return json.loads(raw)

    def list_pods(self) -> list[dict[str, Any]]:
        payload = self.request("GET", "/pods")
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            pods = payload.get("pods") or payload.get("data") or []
            return pods if isinstance(pods, list) else []
        return []

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        payload = self.request("GET", f"/pods/{pod_id}")
        if not isinstance(payload, dict):
            raise RuntimeProvisioningError(f"Unexpected RunPod pod response for {pod_id}")
        return payload

    def create_cpu_pod(
        self,
        *,
        name: str,
        image: str,
        cpu_flavor_ids: list[str],
        container_disk_gb: int,
        volume_gb: int | None,
        vcpu_count: int,
        cloud_type: str,
        ports: list[str],
        env: dict[str, str],
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "imageName": image,
            "computeType": "CPU",
            "cloudType": cloud_type,
            "cpuFlavorIds": cpu_flavor_ids,
            "cpuFlavorPriority": "availability",
            "vcpuCount": vcpu_count,
            "containerDiskInGb": container_disk_gb,
            "ports": ports,
            "dockerEntrypoint": ["/bin/bash", "-lc"],
            "dockerStartCmd": ["/opt/slippi-renderer/start-runpod-worker.sh"],
            "env": env,
        }
        if volume_gb is not None:
            payload["volumeInGb"] = volume_gb
            payload["volumeMountPath"] = "/workspace"
        result = self.request("POST", "/pods", payload=payload)
        if not isinstance(result, dict):
            raise RuntimeProvisioningError("Unexpected RunPod create pod response")
        return result

    def delete_pod(self, pod_id: str) -> Any:
        return self.request("DELETE", f"/pods/{pod_id}")


class RunPodCpuEmulatorRuntime(EmulatorRuntime):
    def __init__(
        self,
        *,
        run_id: str,
        consumer_id: int,
        image: str,
        api_key: str,
        cpu_flavor_ids: list[str],
        container_disk_gb: int,
        volume_gb: int | None,
        vcpu_count: int,
        cloud_type: str,
        renderer_command: str,
        remote_iso_path: str,
        local_iso_path: str | None,
        ssh_private_key_path: str,
        public_key: str,
        dry_run: bool,
        render_timeout_seconds: int,
        video_backend: str,
        dolphin_cpu_core: int,
        dolphin_audio_backend: str,
        wait_timeout_seconds: int,
    ):
        self.run_id = run_id
        self.consumer_id = consumer_id
        self.image = image
        self.cpu_flavor_ids = cpu_flavor_ids
        self.container_disk_gb = container_disk_gb
        self.volume_gb = volume_gb
        self.vcpu_count = vcpu_count
        self.cloud_type = cloud_type
        self.renderer_command = renderer_command
        self.remote_iso_path = remote_iso_path
        self.local_iso_path = local_iso_path
        self.ssh_private_key_path = ssh_private_key_path
        self.public_key = public_key
        self.dry_run = dry_run
        self.render_timeout_seconds = render_timeout_seconds
        self.video_backend = video_backend
        self.dolphin_cpu_core = dolphin_cpu_core
        self.dolphin_audio_backend = dolphin_audio_backend
        self.wait_timeout_seconds = wait_timeout_seconds
        self.pod_name = f"smash-frame-worker-{run_id}-{consumer_id}"
        self.client = RunPodRestClient(api_key=api_key)
        self.pod: dict[str, Any] | None = None
        self.provision()

    def provision(self) -> None:
        if self.dry_run:
            self.pod = {
                "id": f"dry-run-{self.pod_name}",
                "name": self.pod_name,
                "computeType": "CPU",
                "imageName": self.image,
                "cpuFlavorIds": self.cpu_flavor_ids,
                "vcpuCount": self.vcpu_count,
                "volumeInGb": self.volume_gb,
                "sshPrivateKeyPath": self.ssh_private_key_path,
                "publicKeyProvided": bool(self.public_key),
            }
            return
        if not self.image:
            raise RuntimeProvisioningError("--runpod-image is required for RunPod runtime")
        self.pod = self.client.create_cpu_pod(
            name=self.pod_name,
            image=self.image,
            cpu_flavor_ids=self.cpu_flavor_ids,
            container_disk_gb=self.container_disk_gb,
            volume_gb=self.volume_gb,
            vcpu_count=self.vcpu_count,
            cloud_type=self.cloud_type,
            ports=["22/tcp"],
            env={
                key: value
                for key, value in {
                    "SMASH_FRAME_QUEUE_RUN_ID": self.run_id,
                    "PUBLIC_KEY": self.public_key,
                }.items()
                if value
            },
        )

    def render(self, job: FrameRenderJob) -> dict[str, Any]:
        job.ensure_dirs()
        pod_id = str((self.pod or {}).get("id") or "")
        remote_job_dir = f"/workspace/frame-queue/{self.run_id}/{job.job_id}"
        remote_slp = f"{remote_job_dir}/input.slp"
        remote_playback = f"{remote_job_dir}/playback.json"
        remote_raw = f"{remote_job_dir}/raw-frames"
        remote_iso_dir = str(Path(self.remote_iso_path).parent)
        write_json(job.remote_playback_json_path, job.playback_config(remote_slp))

        remote_command = [
            self.renderer_command,
            "--replay-json",
            remote_playback,
            "--output-dir",
            remote_raw,
            "--iso",
            self.remote_iso_path,
            "--timeout-seconds",
            str(self.render_timeout_seconds),
            "--video-backend",
            self.video_backend,
            "--cpu-core",
            str(self.dolphin_cpu_core),
            "--audio-backend",
            self.dolphin_audio_backend,
        ]

        if self.dry_run:
            return self.plan(job)

        ssh = self.wait_for_ssh(pod_id)
        ssh_target = f"root@{ssh['host']}"
        ssh_base = [
            "ssh",
            "-p",
            str(ssh["port"]),
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        if self.ssh_private_key_path:
            ssh_base.extend(["-i", self.ssh_private_key_path])
        ssh_base.extend(["-o", "BatchMode=yes", ssh_target])
        ssh_transport = [
            "ssh",
            "-p",
            str(ssh["port"]),
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        if self.ssh_private_key_path:
            ssh_transport.extend(["-i", self.ssh_private_key_path])
        ssh_transport.extend(["-o", "BatchMode=yes"])
        rsync_base = [
            "rsync",
            "-az",
            "-e",
            shlex.join(ssh_transport),
        ]
        runner = CommandRunner(cwd=job.root_dir)
        runner.run(
            ssh_base
            + [f"mkdir -p {shlex.quote(remote_job_dir)} {shlex.quote(remote_iso_dir)}"]
        )
        runner.run(rsync_base + [str(job.slp_path), f"{ssh_target}:{remote_slp}"])
        runner.run(
            rsync_base + [str(job.remote_playback_json_path), f"{ssh_target}:{remote_playback}"]
        )
        if self.local_iso_path:
            runner.run(
                rsync_base
                + [str(Path(self.local_iso_path).expanduser()), f"{ssh_target}:{self.remote_iso_path}"]
            )
        render_result = runner.run(ssh_base + [shlex.join(remote_command)], check=False)
        job.raw_frame_dir.mkdir(parents=True, exist_ok=True)
        sync_result = runner.run(
            rsync_base + [f"{ssh_target}:{remote_raw}/", f"{job.raw_frame_dir}/"],
            check=False,
        )
        if render_result.returncode != 0:
            raise RuntimeProvisioningError(
                "RunPod render failed after remote execution. "
                f"Return code: {render_result.returncode}. "
                f"Synced remote outputs: {sync_result.returncode == 0}. "
                f"Local raw frame dir: {job.raw_frame_dir}. "
                f"Command: {shlex.join(remote_command)}"
            )
        if sync_result.returncode != 0:
            raise RuntimeProvisioningError(
                "RunPod render succeeded but output sync failed. "
                f"Return code: {sync_result.returncode}. "
                f"Local raw frame dir: {job.raw_frame_dir}"
            )
        return {
            "runtime": "runpod",
            "status": "rendered",
            "podId": pod_id,
            "podName": self.pod_name,
            "rawFrameDir": str(job.raw_frame_dir),
            "remoteRawFrameDir": remote_raw,
            "command": render_result.to_json(),
        }

    def plan(self, job: FrameRenderJob) -> dict[str, Any]:
        job.ensure_dirs()
        remote_job_dir = f"/workspace/frame-queue/{self.run_id}/{job.job_id}"
        remote_slp = f"{remote_job_dir}/input.slp"
        remote_playback = f"{remote_job_dir}/playback.json"
        remote_raw = f"{remote_job_dir}/raw-frames"
        remote_command = [
            self.renderer_command,
            "--replay-json",
            remote_playback,
            "--output-dir",
            remote_raw,
            "--iso",
            self.remote_iso_path,
            "--timeout-seconds",
            str(self.render_timeout_seconds),
            "--video-backend",
            self.video_backend,
            "--cpu-core",
            str(self.dolphin_cpu_core),
            "--audio-backend",
            self.dolphin_audio_backend,
        ]
        write_json(job.remote_playback_json_path, job.playback_config(remote_slp))
        plan = {
            "runtime": "runpod",
            "status": "planned",
            "pod": self.pod,
            "ssh": {
                "privateKeyPath": self.ssh_private_key_path,
                "publicKeyProvided": bool(self.public_key),
            },
            "remote": {
                "jobDir": remote_job_dir,
                "slp": remote_slp,
                "playbackJson": remote_playback,
                "iso": self.remote_iso_path,
                "uploadsIso": bool(self.local_iso_path),
                "rawFrameDir": remote_raw,
                "command": remote_command,
                "renderer": {
                    "timeoutSeconds": self.render_timeout_seconds,
                    "videoBackend": self.video_backend,
                    "dolphinCpuCore": self.dolphin_cpu_core,
                    "audioBackend": self.dolphin_audio_backend,
                },
            },
            "local": {
                "rawFrameDir": str(job.raw_frame_dir),
                "remotePlaybackJson": str(job.remote_playback_json_path),
            },
        }
        write_json(job.job_dir / "runpod-render-plan.json", plan)
        return plan

    def wait_for_ssh(self, pod_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.wait_timeout_seconds
        while time.monotonic() < deadline:
            pod = self.client.get_pod(pod_id)
            host = pod.get("publicIp") or pod.get("ip")
            mappings = pod.get("portMappings") or {}
            ssh_port = mappings.get("22") or mappings.get("22/tcp")
            if isinstance(ssh_port, dict):
                ssh_port = ssh_port.get("hostPort") or ssh_port.get("port")
            if host and ssh_port:
                return {"host": host, "port": int(ssh_port)}
            time.sleep(5)
        raise RuntimeProvisioningError(f"RunPod pod {pod_id} did not expose SSH in time")

    def close(self) -> None:
        if self.dry_run:
            return
        pod_id = str((self.pod or {}).get("id") or "")
        if pod_id:
            self.client.delete_pod(pod_id)


class RuntimeFactory:
    @staticmethod
    def create(
        *,
        runtime: str,
        root_dir: Path,
        run_id: str,
        consumer_id: int,
        args: Any,
    ) -> EmulatorRuntime:
        if runtime == "plan":
            return PlanningRuntime(
                runtime_name="plan",
                reason="Queue/provisioning dry run; no emulator process is started.",
            )
        if runtime == "local-macos":
            return LocalMacPlaybackRuntime(
                root_dir=root_dir,
                consumer_id=consumer_id,
                timeout_seconds=args.timeout_seconds,
                playback_app=args.playback_app,
                iso_path=args.iso,
                allow_parallel=args.allow_parallel_local,
            )
        if runtime == "docker":
            return DockerEmulatorRuntime(
                root_dir=root_dir,
                run_id=run_id,
                consumer_id=consumer_id,
                image=args.docker_image,
                iso_path=args.iso,
                renderer_command=args.renderer_command,
                timeout_seconds=args.timeout_seconds,
                video_backend=args.video_backend,
                dolphin_cpu_core=args.dolphin_cpu_core,
                dolphin_audio_backend=args.dolphin_audio_backend,
                dry_run=args.dry_run,
            )
        if runtime == "runpod":
            return RunPodCpuEmulatorRuntime(
                run_id=run_id,
                consumer_id=consumer_id,
                image=args.runpod_image,
                api_key=os.environ.get("RUNPOD_API_KEY", ""),
                cpu_flavor_ids=args.runpod_cpu_flavor_id,
                container_disk_gb=args.runpod_container_disk_gb,
                volume_gb=args.runpod_volume_gb,
                vcpu_count=args.runpod_vcpu_count,
                cloud_type=args.runpod_cloud_type,
                renderer_command=args.renderer_command,
                remote_iso_path=args.runpod_remote_iso,
                local_iso_path=args.iso,
                ssh_private_key_path=args.runpod_ssh_private_key,
                public_key=args.runpod_public_key,
                dry_run=args.dry_run,
                render_timeout_seconds=args.timeout_seconds,
                video_backend=args.video_backend,
                dolphin_cpu_core=args.dolphin_cpu_core,
                dolphin_audio_backend=args.dolphin_audio_backend,
                wait_timeout_seconds=args.runpod_wait_timeout_seconds,
            )
        raise RuntimeProvisioningError(f"Unknown runtime: {runtime}")
