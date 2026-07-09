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
_VIDEO_SUFFIXES = (".avi", ".mkv", ".mp4", ".mov", ".nut")


def log_event(event: str, **fields):
    """Print one compact pipeline event."""
    with _LOG_LOCK:
        print(json.dumps({"event": event, "time": round(time.time(), 3), **fields}), flush=True)


@dataclass(frozen=True)
class SlpSample:
    """One replay selected for video rendering."""

    id: int
    hf_reference: str


@dataclass(frozen=True)
class HfLocation:
    """Repo and logical folders for raw and processed replay data."""

    repo: str
    hf: HfConnector
    root: str = ""
    raw_slp_dir: str = "raw_slp"
    recording_dir: str = "slp_with_video"

    def raw_slp_path(self) -> str:
        """Return the folder where original SLP files live."""
        return self._join(self.root, self.raw_slp_dir)

    def recording_path(self, sample: SlpSample) -> str:
        """Return the output folder for one processed sample."""
        return self._join(self.root, self.recording_dir, str(sample.id))

    def _join(self, *parts: str) -> str:
        return "/".join(part.strip("/") for part in parts if part.strip("/"))


class SlpProducer:
    def __init__(
        self,
        queue: Queue[SlpSample | None],
        hf_location: HfLocation,
        desired_max: int,
        skip_existing_processed: bool = False,
    ):
        """Verify HF access and remember where raw SLP files live."""
        self.queue = queue
        self.hf_location = hf_location
        self.desired_max = desired_max
        self.skip_existing_processed = skip_existing_processed
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
            processed_ids = self._processed_ids() if self.skip_existing_processed else set()
        except Exception as error:
            self.error = str(error)
            log_event("producer_failed", error=str(error))
            return
        if processed_ids:
            log_event("existing_processed", count=len(processed_ids))
        for sample_id, hf_reference in enumerate(sorted(raw_files)[: self.desired_max]):
            if sample_id in processed_ids:
                continue
            sample = SlpSample(id=sample_id, hf_reference=hf_reference)
            self.queue.put(sample, block=True)
            self.count += 1
            log_event("added", sample=sample.id, source=sample.hf_reference)
        log_event("producer_done", count=self.count)

    def _processed_ids(self) -> set[int]:
        prefix = self.hf_location._join(self.hf_location.root, self.hf_location.recording_dir)
        seen: dict[int, set[str]] = {}
        for path in self.hf_location.hf.list_files(self.hf_location.repo, prefix):
            suffix = path.removeprefix(prefix).strip("/")
            sample_id, _, filename = suffix.partition("/")
            if sample_id.isdigit():
                flags = seen.setdefault(int(sample_id), set())
                lower = filename.lower()
                if lower.endswith(".slp"):
                    flags.add("slp")
                if lower.endswith(_VIDEO_SUFFIXES):
                    flags.add("video")
        return {sample_id for sample_id, flags in seen.items() if {"slp", "video"} <= flags}


class FrameRecorder:
    def __init__(
        self,
        queue: Queue[SlpSample | None],
        hf_location: HfLocation,
        runpod: RunpodConnector,
    ):
        """Create a GPU worker capable of rendering replay video."""
        self.queue = queue
        self.hf_location = hf_location
        self.runpod = runpod
        self.results: list[dict] = []
        self.errors: list[dict] = []
        self.pending_uploads: list[SlpSample] = []
        self.pending_results: list[dict] = []
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
        """Consume queued SLP samples, render videos, and write completed outputs back to HF."""
        try:
            while True:
                sample = self.queue.get(block=True)
                try:
                    if sample is None:
                        self._upload_pending()
                        log_event("recorder_done", podId=self.worker.id)
                        return
                    log_event("picked", sample=sample.id, podId=self.worker.id)
                    result = self._record_one(sample)
                    self.pending_uploads.append(sample)
                    self.pending_results.append(result)
                    log_event("rendered", sample=sample.id, podId=self.worker.id)
                    if len(self.pending_uploads) >= self.runpod.upload_batch_size:
                        self._upload_pending()
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
        """Delete the owned GPU worker exactly once."""
        if not self.closed:
            self.runpod.delete_instance(self.worker)
            self.closed = True

    def _record_one(self, sample: SlpSample) -> dict:
        render = self.runpod.record_video(self.worker, sample, self.hf_location)
        return {
            "sample": sample.id,
            "source": sample.hf_reference,
            "podId": self.worker.id,
            "prepare": self.prepare_result,
            "render": render,
        }

    def _upload_pending(self):
        if not self.pending_uploads:
            return
        samples = list(self.pending_uploads)
        results = list(self.pending_results)
        for attempt in range(1, self.runpod.upload_retries + 1):
            try:
                upload = self.runpod.upload_recorded_videos(self.worker, self.hf_location, samples)
                break
            except Exception as error:
                log_event(
                    "upload_retry",
                    podId=self.worker.id,
                    count=len(samples),
                    attempt=attempt,
                    error=str(error),
                )
                if attempt == self.runpod.upload_retries:
                    for failed_sample in samples:
                        self.errors.append(
                            {
                                "sample": failed_sample.id,
                                "hfReference": failed_sample.hf_reference,
                                "podId": self.worker.id,
                                "error": str(error),
                                "phase": "upload",
                            }
                        )
                    self.pending_uploads = []
                    self.pending_results = []
                    raise
                time.sleep(self.runpod.upload_retry_seconds * attempt)
        self.pending_uploads = []
        self.pending_results = []
        self.results.extend(results)
        log_event("uploaded_batch", podId=self.worker.id, count=len(samples), files=upload.get("files"))
        for sample in samples:
            log_event("completed", sample=sample.id, podId=self.worker.id)
