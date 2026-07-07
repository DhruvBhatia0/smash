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
    processed_output_upload: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runId": self.run_id,
            "runDir": str(self.run_dir),
            "enqueuedCount": self.enqueued_count,
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
        self.result_store.initialize(
            {
                "runId": self.run_id,
                "runtime": self.args.runtime,
                "planOnly": self.args.plan_only,
                "dryRun": self.args.dry_run,
                "queueSize": self.args.queue_size,
                "consumerCount": self.args.consumers,
                "inputPaths": [str(path) for path in self.input_paths],
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
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
                    "elapsedSeconds": round(time.monotonic() - started, 3),
                    "producerError": {
                        "type": type(producer.error).__name__,
                        "message": str(producer.error),
                    },
                }
            )
            raise producer.error

        elapsed = time.monotonic() - started
        self.result_store.finalize(
            {
                "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "enqueuedCount": producer.enqueued_count,
                "elapsedSeconds": round(elapsed, 3),
            }
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
            processed_output_upload=run_upload,
        )
