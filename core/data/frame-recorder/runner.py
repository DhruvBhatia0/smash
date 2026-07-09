import json
import os
import threading
import time
from queue import Queue

from .hf_connector import HfConnector
from .models import FrameRecorder, HfLocation, SlpProducer, SlpSample, log_event
from .runpod_connector import RunpodConnector


class DatasetRunner:
    def __init__(
        self,
        hf_location: HfLocation,
        runpod: RunpodConnector,
        desired_max: int,
        recorder_count: int,
        skip_existing_processed: bool = False,
        min_processed_files: int = 1,
    ):
        """Create the bounded queue and the producer/consumer objects."""
        self.queue: Queue[SlpSample | None] = Queue(maxsize=1000)
        self.producer = SlpProducer(
            self.queue,
            hf_location,
            desired_max,
            skip_existing_processed=skip_existing_processed,
            min_processed_files=min_processed_files,
        )
        self.hf_location = hf_location
        self.runpod = runpod
        self.recorder_count = recorder_count
        self.recorders: list[FrameRecorder] = []
        self.startup_errors: list[dict] = []
        self.lock = threading.Lock()

    def run(self) -> dict:
        """Run one producer thread and N RunPod-backed recorder threads."""
        started = time.monotonic()
        producer = threading.Thread(target=self.producer.download, name="slp-producer")
        consumers = [
            threading.Thread(target=self._record, args=(index,), name=f"frame-recorder-{index}")
            for index in range(self.recorder_count)
        ]

        for consumer in consumers:
            consumer.start()
        producer.start()

        producer.join()
        for _ in consumers:
            self.queue.put(None, block=True)
        for consumer in consumers:
            consumer.join()

        results = [result for recorder in self.recorders for result in recorder.results]
        errors = [error for recorder in self.recorders for error in recorder.errors]
        errors += self.startup_errors
        if self.producer.error:
            errors.append({"component": "SlpProducer", "error": self.producer.error})
        return {
            "queued": self.producer.count,
            "recorders": len(self.recorders),
            "processed": len(results),
            "failed": len(errors),
            "seconds": round(time.monotonic() - started, 3),
            "results": sorted(results, key=lambda row: row["sample"]),
            "errors": errors,
        }

    def _record(self, index: int):
        recorder = None
        try:
            recorder = FrameRecorder(self.queue, self.hf_location, self.runpod)
            with self.lock:
                self.recorders.append(recorder)
            recorder.record()
        except Exception as error:
            event = "recorder_startup_failed" if recorder is None else "recorder_failed"
            log_event(event, index=index, error=str(error))
            with self.lock:
                self.startup_errors.append(
                    {"component": "FrameRecorder", "index": index, "error": str(error)}
                )


def build_runner() -> DatasetRunner:
    """Build the RunPod-backed dataset runner from environment."""
    hf = HfConnector(expected_email=os.environ.get("SMASH_HF_EXPECTED_EMAIL"))
    hf_location = HfLocation(
        repo=os.environ["SMASH_HF_REPO"],
        hf=hf,
        root=os.environ.get("SMASH_HF_ROOT", ""),
    )
    return DatasetRunner(
        hf_location=hf_location,
        runpod=RunpodConnector(
            name_prefix=os.environ.get("SMASH_RUNPOD_NAME_PREFIX", "smash-core-worker")
        ),
        desired_max=int(os.environ.get("SMASH_SAMPLE_LIMIT", "1")),
        recorder_count=int(os.environ.get("SMASH_WORKER_COUNT", "1")),
        skip_existing_processed=os.environ.get("SMASH_SKIP_EXISTING_PROCESSED", "") == "1",
        min_processed_files=int(os.environ.get("SMASH_MIN_PROCESSED_FILES", "1")),
    )


def main():
    """CLI entrypoint for running the dataset pipeline."""
    print(json.dumps(build_runner().run(), indent=2))


if __name__ == "__main__":
    main()
