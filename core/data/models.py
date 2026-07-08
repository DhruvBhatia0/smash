from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .hf_connector import HfConnector
    from .runpod_connector import RunpodConnector


_LOG_LOCK = threading.Lock()


def log_event(event: str, **fields):
    """Print one compact pipeline event."""
    with _LOG_LOCK:
        print(json.dumps({"event": event, "time": round(time.time(), 3), **fields}), flush=True)


@dataclass(frozen=True)
class SlpSample:
    """One replay selected for frame recording."""

    id: int
    hf_reference: str


@dataclass(frozen=True)
class HfLocation:
    """Repo and logical folders for raw and processed replay data."""

    repo: str
    hf: HfConnector
    root: str = ""
    raw_slp_dir: str = "raw_slp"
    framed_slp_dir: str = "slp_with_frame"

    def raw_slp_path(self) -> str:
        """Return the folder where original SLP files live."""
        return self._join(self.root, self.raw_slp_dir)

    def framed_slp_path(self, sample: SlpSample) -> str:
        """Return the output folder for one processed sample."""
        return self._join(self.root, self.framed_slp_dir, str(sample.id))

    def _join(self, *parts: str) -> str:
        return "/".join(part.strip("/") for part in parts if part.strip("/"))


class SlpDownloader:
    def __init__(
        self,
        queue: Queue[SlpSample | None],
        hf_location: HfLocation,
        desired_max: int,
    ):
        """Verify HF access and remember where raw SLP files live."""
        self.queue = queue
        self.hf_location = hf_location
        self.desired_max = desired_max
        self.hf_location.hf.create_repo(self.hf_location.repo)
        self.count = 0
        self.error: str | None = None

    def download(self):
        """Read raw SLP references from HF and blockingly place SlpSample objects on the queue."""
        try:
            raw_files = [
                path
                for path in self.hf_location.hf.list_files(
                    self.hf_location.repo,
                    self.hf_location.raw_slp_path(),
                )
                if path.endswith(".slp")
            ]
        except Exception as error:
            self.error = str(error)
            log_event("producer_failed", error=str(error))
            return
        for sample_id, hf_reference in enumerate(raw_files[: self.desired_max]):
            sample = SlpSample(id=sample_id, hf_reference=hf_reference)
            self.queue.put(sample, block=True)
            self.count += 1
            log_event("added", sample=sample.id, source=sample.hf_reference)
        log_event("producer_done", count=self.count)


class FrameRecorder:
    def __init__(
        self,
        queue: Queue[SlpSample | None],
        hf_location: HfLocation,
        runpod: RunpodConnector,
    ):
        """Create or attach to a CPU worker capable of running the frame-recording image."""
        self.queue = queue
        self.hf_location = hf_location
        self.runpod = runpod
        self.results: list[dict] = []
        self.errors: list[dict] = []
        self.closed = False
        worker = self.runpod.create_instance()
        try:
            self.worker = self.runpod.wait_for_ssh(worker)
            self.prepare_result = self.runpod.prepare_instance(self.worker)
            log_event("recorder_ready", podId=self.worker.id)
        except Exception:
            self.runpod.delete_instance(worker)
            self.closed = True
            raise

    def record(self):
        """Consume queued SLP samples, record frames, and write completed outputs back to HF."""
        try:
            while True:
                sample = self.queue.get(block=True)
                try:
                    if sample is None:
                        log_event("recorder_done", podId=self.worker.id)
                        return
                    log_event("picked", sample=sample.id, podId=self.worker.id)
                    self.results.append(self._record_one(sample))
                    log_event("completed", sample=sample.id, podId=self.worker.id)
                except Exception as error:
                    self.errors.append(
                        {
                            "sample": getattr(sample, "id", None),
                            "hfReference": getattr(sample, "hf_reference", None),
                            "podId": self.worker.id,
                            "error": str(error),
                        }
                    )
                    log_event(
                        "failed",
                        sample=getattr(sample, "id", None),
                        podId=self.worker.id,
                        error=str(error),
                    )
                finally:
                    self.queue.task_done()
        finally:
            self.close()

    def close(self):
        """Delete the owned CPU worker exactly once."""
        if not self.closed:
            self.runpod.delete_instance(self.worker)
            self.closed = True

    def _record_one(self, sample: SlpSample) -> dict:
        render = self.runpod.record_frames(self.worker, sample, self.hf_location)
        return {
            "sample": sample.id,
            "source": sample.hf_reference,
            "podId": self.worker.id,
            "prepare": self.prepare_result,
            "render": render,
        }
