from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

from .commands import CommandRunner
from .hf_storage import ProcessedFramePublisher
from .jobs import FrameRenderJob, QueueStop, write_json
from .result_store import FrameQueueResultStore
from .runtimes import EmulatorRuntime, RuntimeFactory


class FrameJobProcessor:
    def __init__(
        self,
        *,
        root_dir: Path,
        runtime: EmulatorRuntime,
        node_bin: str,
        plan_only: bool,
        extract_state: bool,
        align_frames: bool,
        attach_images: bool,
    ):
        self.root_dir = root_dir
        self.runtime = runtime
        self.node_bin = node_bin
        self.plan_only = plan_only
        self.extract_state = extract_state
        self.align_frames = align_frames
        self.attach_images = attach_images
        self.runner = CommandRunner(cwd=root_dir)

    def process(self, job: FrameRenderJob) -> dict[str, Any]:
        started = time.monotonic()
        job.ensure_dirs()
        write_json(job.playback_json_path, job.playback_config())

        steps: dict[str, Any] = {}
        if self.plan_only:
            render_result = self.runtime.plan(job)
            return {
                "status": "planned",
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "steps": {"render": render_result},
            }

        if self.extract_state:
            steps["extractState"] = self.extract_state_rows(job)

        steps["render"] = self.runtime.render(job)

        if self.align_frames:
            steps["alignFrames"] = self.align_rendered_frames(job)

        if self.attach_images:
            steps["attachImages"] = self.attach_frame_images(job)

        return {
            "status": "processed",
            "elapsedSeconds": round(time.monotonic() - started, 3),
            "steps": steps,
            "outputs": {
                "stateDir": str(job.state_dir),
                "rawFrameDir": str(job.raw_frame_dir),
                "alignedFrameDir": str(job.aligned_frame_dir),
                "imageRows": str(job.image_rows_path),
            },
        }

    def extract_state_rows(self, job: FrameRenderJob) -> dict[str, Any]:
        result = self.runner.run(
            [
                self.node_bin,
                "scripts/extract-slp.mjs",
                str(job.slp_path),
                str(job.state_dir),
            ],
            check=True,
        )
        return {
            "status": "extracted",
            "command": result.to_json(),
            "stateRows": str(job.state_dir / f"{job.replay_id}.frames.jsonl"),
            "manifest": str(job.state_dir / f"{job.replay_id}.manifest.json"),
        }

    def align_rendered_frames(self, job: FrameRenderJob) -> dict[str, Any]:
        manifest_path = job.raw_frame_dir / "manifest.json"
        if not manifest_path.exists():
            return {
                "status": "skipped",
                "reason": f"missing render manifest: {manifest_path}",
            }
        result = self.runner.run(
            [
                self.node_bin,
                "scripts/align-rendered-frames.mjs",
                str(job.raw_frame_dir),
                str(job.aligned_frame_dir),
            ],
            check=True,
        )
        return {
            "status": "aligned",
            "command": result.to_json(),
            "framesJsonl": str(job.aligned_frame_dir / "frames.jsonl"),
            "manifest": str(job.aligned_frame_dir / "manifest.json"),
        }

    def attach_frame_images(self, job: FrameRenderJob) -> dict[str, Any]:
        state_rows = job.state_dir / f"{job.replay_id}.frames.jsonl"
        frame_rows = job.aligned_frame_dir / "frames.jsonl"
        if not state_rows.exists() or not frame_rows.exists():
            return {
                "status": "skipped",
                "reason": "missing state rows or aligned frame rows",
                "stateRows": str(state_rows),
                "frameRows": str(frame_rows),
            }
        result = self.runner.run(
            [
                self.node_bin,
                "scripts/attach-frame-images.mjs",
                str(state_rows),
                str(frame_rows),
                str(job.image_rows_path),
            ],
            check=True,
        )
        return {
            "status": "attached",
            "command": result.to_json(),
            "imageRows": str(job.image_rows_path),
            "manifest": str(job.image_rows_path).replace(".jsonl", ".manifest.json"),
        }


class FrameProcessingConsumer(threading.Thread):
    def __init__(
        self,
        *,
        consumer_id: int,
        input_queue: "queue.Queue[FrameRenderJob | QueueStop]",
        root_dir: Path,
        run_id: str,
        result_store: FrameQueueResultStore,
        args: Any,
    ):
        super().__init__(name=f"frame-consumer-{consumer_id}", daemon=False)
        self.consumer_id = consumer_id
        self.input_queue = input_queue
        self.root_dir = root_dir
        self.run_id = run_id
        self.result_store = result_store
        self.runtime = RuntimeFactory.create(
            runtime=args.runtime,
            root_dir=root_dir,
            run_id=run_id,
            consumer_id=consumer_id,
            args=args,
        )
        self.processor = FrameJobProcessor(
            root_dir=root_dir,
            runtime=self.runtime,
            node_bin=args.node_bin,
            plan_only=args.plan_only,
            extract_state=args.extract_state,
            align_frames=args.align_frames,
            attach_images=args.attach_images,
        )
        self.publisher = ProcessedFramePublisher.from_args(args)

    def run(self) -> None:
        try:
            while True:
                item = self.input_queue.get(block=True)
                try:
                    if isinstance(item, QueueStop):
                        return
                    self.process_one(item)
                finally:
                    self.input_queue.task_done()
        finally:
            self.runtime.close()

    def process_one(self, job: FrameRenderJob) -> None:
        self.result_store.write_event(
            "job_started",
            {"jobId": job.job_id, "consumerId": self.consumer_id},
        )
        try:
            result = self.processor.process(job)
            result["consumerId"] = self.consumer_id
            if self.publisher is not None:
                try:
                    self.result_store.write_result_files(job=job, result=result)
                    upload_result = self.publisher.publish_job(job)
                    result["processedOutputUpload"] = upload_result
                    if upload_result.get("status") in {"uploaded", "planned"}:
                        self.result_store.write_result_files(job=job, result=result)
                        result_upload = self.publisher.publish_job_result(job)
                        result["processedOutputUpload"]["resultJson"] = result_upload
                        cleanup = self.publisher.cleanup_job(job)
                        if cleanup.get("status") != "skipped":
                            result["processedOutputUpload"]["localCleanup"] = cleanup
                except Exception as error:
                    result["processedOutputUpload"] = {
                        "status": "failed",
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                    if result.get("status") == "processed":
                        result["status"] = "processed-upload-failed"
        except Exception as error:
            result = {
                "status": "failed",
                "consumerId": self.consumer_id,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        self.result_store.write_result(job=job, result=result)
        self.result_store.write_event(
            "job_finished",
            {
                "jobId": job.job_id,
                "consumerId": self.consumer_id,
                "status": result.get("status", "unknown"),
            },
        )
