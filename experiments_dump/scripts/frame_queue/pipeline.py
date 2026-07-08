from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .consumer import FrameProcessingConsumer
from .hf_storage import ProcessedFramePublisher
from .jobs import FrameRenderJob, QueueStop
from .producer import SlpCorpusProducer
from .result_store import FrameQueueResultStore


@dataclass(frozen=True)
class FrameQueueRunResult:
    run_id: str
    run_dir: Path
    enqueued_count: int
    counts: dict[str, int]
    elapsed_seconds: float
    skipped_existing_count: int = 0
    processed_output_upload: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runId": self.run_id,
            "runDir": str(self.run_dir),
            "enqueuedCount": self.enqueued_count,
            "skippedExistingCount": self.skipped_existing_count,
            "counts": self.counts,
            "elapsedSeconds": round(self.elapsed_seconds, 3),
        }
        if self.processed_output_upload is not None:
            payload["processedOutputUpload"] = self.processed_output_upload
        return payload


class FrameQueuePipeline:
    def __init__(
        self,
        *,
        root_dir: Path,
        run_id: str,
        input_paths: list[Path],
        args: Any,
    ):
        self.root_dir = root_dir.resolve()
        self.run_id = run_id
        self.input_paths = input_paths
        self.args = args
        self.run_dir = self.root_dir / "processed-frame-queues" / run_id
        self.result_store = FrameQueueResultStore(run_dir=self.run_dir)
        self.queue: "queue.Queue[FrameRenderJob | QueueStop]" = queue.Queue(
            maxsize=args.queue_size
        )
        self.publisher = ProcessedFramePublisher.from_args(args)

    def run(self) -> FrameQueueRunResult:
        started = time.monotonic()
        resume_existing = bool(getattr(self.args, "resume_existing_run", False))
        resume_skip_status = set(getattr(self.args, "resume_skip_status", None) or ["processed"])
        existing_results = self.result_store.load_existing_results() if resume_existing else {}
        skip_job_ids = {
            job_id
            for job_id, payload in existing_results.items()
            if str(payload.get("status", "")) in resume_skip_status
        }
        if resume_existing:
            self.result_store.seed_counts(list(existing_results.values()))
        self.result_store.initialize(
            {
                "runId": self.run_id,
                "runtime": self.args.runtime,
                "planOnly": self.args.plan_only,
                "dryRun": self.args.dry_run,
                "queueSize": self.args.queue_size,
                "consumerCount": self.args.consumers,
                "inputPaths": [str(path) for path in self.input_paths],
                "resumeExistingRun": resume_existing,
                "resumeSkipStatus": sorted(resume_skip_status),
                "preexistingResultCount": len(existing_results),
                "preexistingSkippedJobCount": len(skip_job_ids),
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            reset_index=not resume_existing,
        )
        self.result_store.write_event(
            "run_started",
            {
                "runId": self.run_id,
                "runtime": self.args.runtime,
                "consumerCount": self.args.consumers,
            },
        )

        producer = SlpCorpusProducer(
            input_paths=self.input_paths,
            output_queue=self.queue,
            root_dir=self.root_dir,
            run_id=self.run_id,
            consumer_count=self.args.consumers,
            start_frame=self.args.start_frame,
            end_frame=self.args.end_frame,
            recursive=self.args.recursive,
            max_jobs=self.args.max_jobs,
            skip_job_ids=skip_job_ids,
        )
        consumers = [
            FrameProcessingConsumer(
                consumer_id=index,
                input_queue=self.queue,
                root_dir=self.root_dir,
                run_id=self.run_id,
                result_store=self.result_store,
                args=self.args,
            )
            for index in range(self.args.consumers)
        ]

        for consumer in consumers:
            consumer.start()
        producer.start()

        producer.join()
        self.queue.join()
        for consumer in consumers:
            consumer.join()

        if producer.error is not None:
            self.result_store.finalize(
                {
                    "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "enqueuedCount": producer.enqueued_count,
                    "skippedExistingCount": producer.skipped_existing_count,
                    "elapsedSeconds": round(time.monotonic() - started, 3),
                    "producerError": {
                        "type": type(producer.error).__name__,
                        "message": str(producer.error),
                    },
                }
            )
            self.result_store.write_event(
                "run_failed",
                {
                    "runId": self.run_id,
                    "enqueuedCount": producer.enqueued_count,
                    "skippedExistingCount": producer.skipped_existing_count,
                },
            )
            raise producer.error

        elapsed = time.monotonic() - started
        self.result_store.finalize(
            {
                "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "enqueuedCount": producer.enqueued_count,
                "skippedExistingCount": producer.skipped_existing_count,
                "elapsedSeconds": round(elapsed, 3),
            }
        )
        self.result_store.write_event(
            "run_finished",
            {
                "runId": self.run_id,
                "enqueuedCount": producer.enqueued_count,
                "skippedExistingCount": producer.skipped_existing_count,
                "elapsedSeconds": round(elapsed, 3),
            },
        )
        run_upload = None
        if self.publisher is not None:
            run_upload = self.publisher.publish_run_artifacts(
                run_id=self.run_id,
                run_dir=self.run_dir,
            )
        return FrameQueueRunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            enqueued_count=producer.enqueued_count,
            counts=dict(self.result_store.counts),
            elapsed_seconds=elapsed,
            skipped_existing_count=producer.skipped_existing_count,
            processed_output_upload=run_upload,
        )
