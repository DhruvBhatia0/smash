from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .jobs import FrameRenderJob, write_json


class FrameQueueResultStore:
    def __init__(self, *, run_dir: Path):
        self.run_dir = run_dir
        self.results_dir = run_dir / "results"
        self.index_path = run_dir / "index.jsonl"
        self.manifest_path = run_dir / "manifest.json"
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def initialize(self, manifest: dict[str, Any]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text("")
        write_json(self.manifest_path, manifest)

    def payload_for(self, *, job: FrameRenderJob, result: dict[str, Any]) -> dict[str, Any]:
        return {"job": job.to_json(), **result}

    def write_result_files(self, *, job: FrameRenderJob, result: dict[str, Any]) -> None:
        payload = self.payload_for(job=job, result=result)
        write_json(job.result_path, payload)
        write_json(self.results_dir / f"{job.job_id}.json", payload)

    def write_result(self, *, job: FrameRenderJob, result: dict[str, Any]) -> None:
        self.write_result_files(job=job, result=result)
        payload = self.payload_for(job=job, result=result)
        line = json.dumps(payload, sort_keys=True)
        with self.lock:
            status = str(result.get("status", "unknown"))
            self.counts[status] = self.counts.get(status, 0) + 1
            with self.index_path.open("a") as handle:
                handle.write(line + "\n")

    def finalize(self, updates: dict[str, Any]) -> None:
        manifest = {}
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
        manifest.update(updates)
        manifest["counts"] = dict(sorted(self.counts.items()))
        write_json(self.manifest_path, manifest)
