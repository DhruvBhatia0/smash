from __future__ import annotations

import queue
import threading
from pathlib import Path

from .jobs import FrameRenderJob, QueueStop


class SlpCorpusProducer(threading.Thread):
    def __init__(
        self,
        *,
        input_paths: list[Path],
        output_queue: "queue.Queue[FrameRenderJob | QueueStop]",
        root_dir: Path,
        run_id: str,
        consumer_count: int,
        start_frame: int | None,
        end_frame: int | None,
        recursive: bool = True,
        max_jobs: int | None = None,
    ):
        super().__init__(name="slp-producer", daemon=False)
        self.input_paths = input_paths
        self.output_queue = output_queue
        self.root_dir = root_dir
        self.run_id = run_id
        self.consumer_count = consumer_count
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.recursive = recursive
        self.max_jobs = max_jobs
        self.enqueued_count = 0
        self.error: BaseException | None = None

    def discover_slp_files(self) -> list[Path]:
        files: list[Path] = []
        for input_path in self.input_paths:
            resolved = input_path.expanduser().resolve()
            if resolved.is_file() and resolved.suffix.lower() == ".slp":
                files.append(resolved)
            elif resolved.is_dir():
                pattern = "**/*.slp" if self.recursive else "*.slp"
                files.extend(path.resolve() for path in resolved.glob(pattern) if path.is_file())
            else:
                raise FileNotFoundError(f"Input path is not an .slp file or directory: {resolved}")

        unique = sorted(dict.fromkeys(files))
        if self.max_jobs is not None:
            return unique[: self.max_jobs]
        return unique

    def run(self) -> None:
        try:
            try:
                for slp_path in self.discover_slp_files():
                    job = FrameRenderJob.from_slp(
                        slp_path=slp_path,
                        root_dir=self.root_dir,
                        run_id=self.run_id,
                        start_frame=self.start_frame,
                        end_frame=self.end_frame,
                    )
                    self.output_queue.put(job, block=True)
                    self.enqueued_count += 1
            except BaseException as error:
                self.error = error
        finally:
            for _ in range(self.consumer_count):
                self.output_queue.put(QueueStop(), block=True)
