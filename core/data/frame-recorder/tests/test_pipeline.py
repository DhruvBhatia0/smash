from __future__ import annotations

import importlib
import threading
import time
import unittest
from queue import Queue


models = importlib.import_module("core.data.frame-recorder.models")
gdrive_module = importlib.import_module("core.data.frame-recorder.gdrive_connector")


class MemoryProvider:
    kind = "memory"

    def __init__(self, source_files: list[str], output_files: list[str] | None = None):
        self.source_files = source_files
        self.output_files = output_files or []
        self.prepared = False

    def prepare(self, namespace: str):
        self.prepared = True

    def list_files(self, namespace: str, folder: str = "") -> list[str]:
        return self.output_files

    def list_slp_references(self, namespace: str, folder: str = "") -> list[str]:
        return self.source_files

    def worker_config(self, namespace: str) -> dict:
        return {"kind": self.kind}


class FakeRunpod:
    upload_batch_size = 1
    upload_retries = 1
    upload_retry_seconds = 0

    def __init__(self):
        self.upload_started = threading.Event()
        self.allow_upload = threading.Event()
        self.second_render_started = threading.Event()
        self.upload_finished_at = 0.0
        self.second_render_started_at = 0.0

    def create_instance(self):
        return models.SlpSample(id=99, reference="pod")

    def wait_for_ssh(self, worker):
        return type("Worker", (), {"id": "worker-1"})()

    def prepare_instance(self, worker, location):
        return {"status": "ready"}

    def record_video(self, worker, sample, location):
        if sample.id == 1:
            self.second_render_started_at = time.monotonic()
            self.second_render_started.set()
        return {"sample": sample.id}

    def convert_recorded_video(self, worker, sample):
        return {"inputBytes": 2, "outputBytes": 1}

    def upload_recorded_videos(self, worker, location, samples):
        self.upload_started.set()
        self.allow_upload.wait(2)
        self.upload_finished_at = time.monotonic()
        return {"files": len(samples) * 3}

    def discard_recorded_videos(self, worker, samples):
        pass

    def delete_instance(self, worker):
        pass


class PipelineTests(unittest.TestCase):
    def test_processed_sample_requires_slp_video_and_metadata(self):
        provider = MemoryProvider(
            ["raw/0.slp", "raw/1.slp"],
            [
                "out/0/input.slp",
                "out/0/video.mp4",
                "out/0/metadata.json",
                "out/1/input.slp",
                "out/1/video.mp4",
            ],
        )
        location = models.StorageLocation(
            namespace="dataset",
            provider=provider,
            raw_slp_dir="raw",
            recording_dir="out",
        )
        producer = models.SlpProducer(Queue(), location, desired_max=2, skip_existing_processed=True)
        self.assertEqual(producer._processed_ids(), {0})

    def test_upload_does_not_block_next_render(self):
        queue = Queue()
        queue.put(models.SlpSample(id=0, reference="raw/0.slp"))
        queue.put(models.SlpSample(id=1, reference="raw/1.slp"))
        queue.put(None)
        provider = MemoryProvider([])
        location = models.StorageLocation(namespace="dataset", provider=provider)
        runpod = FakeRunpod()
        recorder = models.FrameRecorder(queue, location, runpod)
        thread = threading.Thread(target=recorder.record)
        thread.start()
        self.assertTrue(runpod.upload_started.wait(2))
        self.assertTrue(runpod.second_render_started.wait(2))
        self.assertEqual(runpod.upload_finished_at, 0.0)
        runpod.allow_upload.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertLess(runpod.second_render_started_at, runpod.upload_finished_at)
        self.assertEqual(len(recorder.results), 2)

    def test_drive_archive_index_references_members(self):
        connector = object.__new__(gdrive_module.GDriveConnector)
        connector.index_workers = 1
        connector.index_suffix = ".slp-index.jsonl"
        connector.list_files = lambda namespace, folder="": [
            "raw/loose.slp",
            "raw/shard.tar.zst",
            "raw/shard.tar.zst.slp-index.jsonl",
        ]
        connector._read_archive_index = lambda namespace, archive, index: [
            f"{archive}::files/inside.slp"
        ]
        connector._build_archive_index = lambda *args: self.fail("existing index was rebuilt")
        self.assertEqual(
            connector.list_slp_references("drive", "raw"),
            ["raw/loose.slp", "raw/shard.tar.zst::files/inside.slp"],
        )


if __name__ == "__main__":
    unittest.main()
