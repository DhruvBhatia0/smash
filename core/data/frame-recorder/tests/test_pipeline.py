from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import io
import json
import math
import os
import tarfile
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from unittest import mock


models = importlib.import_module("core.data.frame-recorder.models")
gdrive_module = importlib.import_module("core.data.frame-recorder.gdrive_connector")
daytona_module = importlib.import_module("core.data.frame-recorder.daytona_connector")
runner_module = importlib.import_module("core.data.frame-recorder.runner")


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


class FakeDrive:
    remote = "test-drive"
    config = Path("/dev/null")
    retries = 1

    def list_files(self, root: str) -> list[str]:
        return []

    def cat(self, path: str) -> bytes:
        raise AssertionError(f"unexpected Drive read: {path}")

    def upload_file(
        self, local: Path, remote_path: str, *, tps_limit: int = 8
    ) -> None:
        raise AssertionError(f"unexpected Drive upload: {local} -> {remote_path}")

    def download_manifests(self, remote_root: str, destination: Path) -> list[Path]:
        return []


class StopAfterOneWait:
    """Let CoordinatorState.reap execute exactly one sweep without sleeping."""

    def __init__(self) -> None:
        self.waits = 0

    def wait(self, timeout: float) -> bool:
        self.waits += 1
        return self.waits > 1


class PipelineTests(unittest.TestCase):
    def _state(
        self,
        spool: Path,
        *,
        prefetch: int = 8,
        max_attempts: int = 2,
        upload_batch_size: int = 3,
        upload_max_bytes: int = 100,
        upload_concurrency: int = 1,
        upload_min_batch: int = 1,
        upload_max_attempts: int = 3,
        spool_max_bytes: int = 6 * 1024**3,
        queue_root: Path | None = None,
    ):
        return daytona_module.CoordinatorState(
            drive=FakeDrive(),
            source_root="source",
            target_root="target",
            spool=spool,
            sample_limit=0,
            prefetch=prefetch,
            lease_seconds=30,
            max_attempts=max_attempts,
            upload_batch_size=upload_batch_size,
            upload_max_bytes=upload_max_bytes,
            upload_concurrency=upload_concurrency,
            upload_min_batch=upload_min_batch,
            upload_max_attempts=upload_max_attempts,
            spool_max_bytes=spool_max_bytes,
            queue_root=queue_root,
        )

    def _insert_job(
        self,
        state,
        job_id: int,
        status: str,
        *,
        attempts: int = 0,
        source_bytes: int = 1,
        result_bytes: int = 1,
        updated_at: float | None = None,
    ) -> None:
        source = state.incoming / f"{job_id:06d}.slp"
        result = state.results / f"{job_id:06d}.tar"
        source.write_bytes(b"slp")
        with state.db:
            state.db.execute(
                """
                INSERT INTO jobs(
                    id, reference, archive, member, status, attempts,
                    source_path, source_bytes, source_sha256,
                    result_path, result_bytes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    f"archive.tar.zst::games/{job_id}.slp",
                    "archive.tar.zst",
                    f"games/{job_id}.slp",
                    status,
                    attempts,
                    str(source),
                    source_bytes,
                    "0" * 64,
                    str(result),
                    result_bytes,
                    time.time() if updated_at is None else updated_at,
                ),
            )

    def _result_tar(self, root: Path, metadata: dict, *, extra_member: bool = False) -> Path:
        video = root / "video.mp4"
        metadata_path = root / "metadata.json"
        bundle = root / f"result-{time.time_ns()}.tar"
        video.write_bytes(b"not-decoded-by-coordinator")
        metadata_path.write_text(json.dumps(metadata))
        with tarfile.open(bundle, "w") as output:
            output.add(video, arcname="video.mp4", recursive=False)
            output.add(metadata_path, arcname="metadata.json", recursive=False)
            if extra_member:
                extra = root / "unexpected.txt"
                extra.write_text("unexpected")
                output.add(extra, arcname="unexpected.txt", recursive=False)
        return bundle

    def _valid_video_metadata(self) -> dict:
        return {
            "width": 252,
            "height": 208,
            "frameRate": "20/1",
            "container": "mp4",
            "codec": "h264",
            "pixelFormat": "yuv420p",
            "sourceFps": 60,
            "targetFps": 20,
            "firstSelectedSlpFrame": -39,
            "lastSourceSlpFrame": 120,
            "lastSelectedSlpFrame": 119,
            "sourceFrameStep": 3,
            "croppedTailSourceFrames": 1,
            "strictCfrEndpointCompatible": False,
            "cpuOnly": True,
        }

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

    def test_daytona_source_index_accepts_full_root_archive_and_rejects_mismatch(self):
        drive = mock.Mock()
        drive.remote = "test-drive"
        drive.config = Path("/dev/null")
        drive.retries = 1
        index = "nested/shard.tar.zst.slp-index.jsonl"
        drive.list_files.return_value = [index]
        drive.cat.return_value = (
            json.dumps(
                {
                    "archive": "/source/nested/shard.tar.zst",
                    "member": "games/match.slp",
                }
            )
            + "\n"
        ).encode()

        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(Path(temporary))
            self.addCleanup(state.db.close)
            state.drive = drive
            self.assertEqual(
                state._source_references(),
                [
                    (
                        "nested/shard.tar.zst::games/match.slp",
                        "nested/shard.tar.zst",
                        "games/match.slp",
                    )
                ],
            )
            drive.cat.assert_called_once_with(f"source/{index}")

            drive.cat.return_value = (
                json.dumps(
                    {
                        "archive": "source/nested/different.tar.zst",
                        "member": "games/match.slp",
                    }
                )
                + "\n"
            ).encode()
            with self.assertRaisesRegex(ValueError, "archive index mismatch"):
                state._source_references()

    def test_fleet_plan_sizes_one_two_and_four_process_sandboxes(self):
        replay_count = 10_000
        plans = {}
        for processes_per_sandbox in (1, 2, 4):
            connector = object.__new__(daytona_module.DaytonaConnector)
            connector.average_replay_seconds = 120.0
            connector.realtime_factor = 2.5
            connector.deadline_hours = 4.0
            connector.worker_processes = processes_per_sandbox
            plans[processes_per_sandbox] = connector.plan(replay_count)

        required_processes = math.ceil(replay_count * 120.0 / (2.5 * 4.0 * 3600))
        self.assertEqual(required_processes, 34)
        self.assertEqual(
            {cpu: plan.recommended_processes for cpu, plan in plans.items()},
            {1: 34, 2: 34, 4: 34},
        )
        self.assertEqual(
            {cpu: plan.recommended_sandboxes for cpu, plan in plans.items()},
            {1: 34, 2: 17, 4: 9},
        )
        for processes_per_sandbox, plan in plans.items():
            self.assertGreaterEqual(
                plan.recommended_sandboxes * processes_per_sandbox,
                required_processes,
            )
            self.assertLessEqual(plan.estimated_hours, plan.deadline_hours)

    def test_fleet_plan_honors_explicit_sandbox_count(self):
        connector = object.__new__(daytona_module.DaytonaConnector)
        connector.average_replay_seconds = 90.0
        connector.realtime_factor = 3.0
        connector.deadline_hours = 4.5
        connector.worker_processes = 2

        plan = connector.plan(replay_count=1_000, requested_sandboxes=7)

        self.assertEqual(plan.requested_sandboxes, 7)
        self.assertEqual(plan.processes_per_sandbox, 2)
        self.assertEqual(
            plan.estimated_hours,
            round(1_000 * 90.0 / (3.0 * 7 * 2 * 3600), 3),
        )

    def test_zero_work_fleet_plan_launches_no_workers(self):
        connector = object.__new__(daytona_module.DaytonaConnector)
        connector.average_replay_seconds = 90.0
        connector.realtime_factor = 3.0
        connector.deadline_hours = 4.5
        connector.worker_processes = 2

        plan = connector.plan(replay_count=0)

        self.assertEqual(plan.replay_count, 0)
        self.assertEqual(plan.recommended_processes, 0)
        self.assertEqual(plan.recommended_sandboxes, 0)
        self.assertEqual(plan.requested_sandboxes, 0)
        self.assertEqual(plan.estimated_hours, 0.0)

    def test_expired_lease_requeues_then_fails_at_max_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(Path(temporary), max_attempts=2)
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued")

            first = state.lease("worker-a")
            self.assertIsNotNone(first)
            with state.db:
                state.db.execute("UPDATE jobs SET lease_deadline=0 WHERE id=0")
            state.stop = StopAfterOneWait()
            state.reap()
            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(row["status"], "queued")
            self.assertEqual(row["attempts"], 1)
            self.assertIsNone(row["lease_token"])
            self.assertIsNone(row["worker_id"])
            self.assertEqual(row["error"], "lease expired")

            second = state.lease("worker-b")
            self.assertIsNotNone(second)
            self.assertNotEqual(first["leaseToken"], second["leaseToken"])
            with state.db:
                state.db.execute("UPDATE jobs SET lease_deadline=0 WHERE id=0")
            state.stop = StopAfterOneWait()
            state.reap()
            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["attempts"], 2)
            self.assertIsNone(state.lease("worker-c"))

    def test_concurrent_shared_claims_grant_once_and_increment_attempt_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_root = root / "queue"
            state = self._state(root / "spool", queue_root=queue_root)
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued")
            state._publish_pending(0)
            barrier = threading.Barrier(32)
            claims = []
            for index in range(32):
                token = f"claim-{index:02d}"
                claim = queue_root / "claims" / f"000000-{token}.json"
                claim.write_text(
                    json.dumps(
                        {
                            "id": 0,
                            "leaseToken": token,
                            "workerId": f"worker-{index:02d}",
                        }
                    )
                )
                claims.append(claim)

            def sync_claim(path: Path) -> None:
                barrier.wait()
                state._sync_filesystem_claim(path)

            with ThreadPoolExecutor(max_workers=32) as executor:
                list(executor.map(sync_claim, claims))

            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            decisions = [
                json.loads(path.read_text())
                for path in (queue_root / "grants").glob("*.json")
            ]
            granted = [decision for decision in decisions if decision["granted"]]
            self.assertEqual(row["status"], "leased")
            self.assertEqual(row["attempts"], 1)
            self.assertEqual(len(decisions), 32)
            self.assertEqual(len(granted), 1)
            self.assertEqual(row["lease_token"], granted[0]["leaseToken"])
            self.assertEqual(len(list((queue_root / "leased").glob("*.json"))), 1)
            self.assertFalse((queue_root / "pending" / "000000.json").exists())

    def test_accept_result_requires_exact_cpu_only_video_metadata(self):
        metadata = {"video": self._valid_video_metadata()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self._state(root / "spool")
            self.addCleanup(state.db.close)
            self._insert_job(state, 7, "queued")
            lease = state.lease("worker")
            bundle = self._result_tar(root, metadata)

            with bundle.open("rb") as source:
                state.accept_result(7, lease["leaseToken"], source, bundle.stat().st_size)

            row = state.db.execute("SELECT * FROM jobs WHERE id=7").fetchone()
            self.assertEqual(row["status"], "complete")
            self.assertEqual(Path(row["result_path"]).read_bytes(), bundle.read_bytes())

    def test_accept_result_rejects_stale_token_if_lease_changes_during_upload(self):
        class LeaseChangingStream(io.BytesIO):
            def __init__(self, payload: bytes, callback) -> None:
                super().__init__(payload)
                self.callback = callback

            def read(self, size: int = -1) -> bytes:
                chunk = super().read(size)
                if self.callback is not None:
                    callback, self.callback = self.callback, None
                    callback()
                return chunk

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self._state(root / "spool")
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued")
            stale_lease = state.lease("worker-old")
            bundle = self._result_tar(root, {"video": self._valid_video_metadata()})
            reassigned: dict = {}

            def reassign() -> None:
                state.fail_job(0, stale_lease["leaseToken"], "worker disconnected")
                reassigned.update(state.lease("worker-new"))

            stream = LeaseChangingStream(bundle.read_bytes(), reassign)
            with self.assertRaisesRegex(PermissionError, "lease expired while result was uploading"):
                state.accept_result(0, stale_lease["leaseToken"], stream, bundle.stat().st_size)

            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(row["status"], "leased")
            self.assertEqual(row["worker_id"], "worker-new")
            self.assertEqual(row["lease_token"], reassigned["leaseToken"])
            self.assertNotEqual(row["lease_token"], stale_lease["leaseToken"])
            self.assertFalse((state.results / "000000.tar").exists())
            self.assertEqual(list(state.results.glob("*.partial")), [])

    def test_accept_result_identical_retry_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self._state(root / "spool")
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued")
            lease = state.lease("worker")
            bundle = self._result_tar(root, {"video": self._valid_video_metadata()})
            payload = bundle.read_bytes()

            state.accept_result(0, lease["leaseToken"], io.BytesIO(payload), len(payload))
            before = dict(state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone())
            state.accept_result(0, lease["leaseToken"], io.BytesIO(payload), len(payload))
            after = dict(state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone())

            self.assertEqual(after, before)
            self.assertEqual(Path(after["result_path"]).read_bytes(), payload)
            self.assertEqual(list(state.results.glob("*.partial")), [])

    def test_accept_result_rejects_mismatched_retry_without_replacing_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self._state(root / "spool")
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued")
            lease = state.lease("worker")
            bundle = self._result_tar(root, {"video": self._valid_video_metadata()})
            original = bundle.read_bytes()
            state.accept_result(0, lease["leaseToken"], io.BytesIO(original), len(original))
            committed = dict(state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone())
            mismatched = original + b"different-retry"

            with self.assertRaises(PermissionError):
                state.accept_result(
                    0,
                    lease["leaseToken"],
                    io.BytesIO(mismatched),
                    len(mismatched),
                )

            after = dict(state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone())
            self.assertEqual(after, committed)
            self.assertEqual(Path(after["result_path"]).read_bytes(), original)
            self.assertEqual(list(state.results.glob("*.partial")), [])

    def test_shared_result_marker_and_short_tar_are_retained_until_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_root = root / "queue"
            state = self._state(root / "spool", queue_root=queue_root)
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued")
            lease = state.lease("worker")
            bundle = self._result_tar(root, {"video": self._valid_video_metadata()})
            payload = bundle.read_bytes()
            stem = state._lease_stem(0, lease["leaseToken"])
            shared_result = queue_root / "results" / f"{stem}.tar"
            marker = queue_root / "results" / f"{stem}.result.json"
            marker.write_text(
                json.dumps(
                    {
                        "id": 0,
                        "leaseToken": lease["leaseToken"],
                        "resultPath": str(shared_result),
                        "resultBytes": len(payload),
                        "resultSha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            )

            state._sync_filesystem_result(marker)
            self.assertTrue(marker.exists())
            self.assertEqual(
                state.db.execute("SELECT status FROM jobs WHERE id=0").fetchone()[0],
                "leased",
            )

            shared_result.write_bytes(payload[: len(payload) // 2])
            state._sync_filesystem_result(marker)
            self.assertTrue(marker.exists())
            self.assertTrue(shared_result.exists())
            self.assertEqual(
                state.db.execute("SELECT status FROM jobs WHERE id=0").fetchone()[0],
                "leased",
            )

            shared_result.write_bytes(payload)
            state._sync_filesystem_result(marker)
            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(row["status"], "complete")
            self.assertEqual(Path(row["result_path"]).read_bytes(), payload)
            self.assertFalse(marker.exists())
            self.assertFalse(shared_result.exists())

    def test_valid_shared_result_wins_over_simultaneous_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_root = root / "queue"
            state = self._state(root / "spool", queue_root=queue_root)
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued")
            lease = state.lease("worker")
            bundle = self._result_tar(root, {"video": self._valid_video_metadata()})
            payload = bundle.read_bytes()
            stem = state._lease_stem(0, lease["leaseToken"])
            shared_result = queue_root / "results" / f"{stem}.tar"
            result_marker = queue_root / "results" / f"{stem}.result.json"
            failure_marker = queue_root / "failures" / f"{stem}.failure.json"
            shared_result.write_bytes(payload)
            result_marker.write_text(
                json.dumps(
                    {
                        "id": 0,
                        "leaseToken": lease["leaseToken"],
                        "resultPath": str(shared_result),
                        "resultBytes": len(payload),
                        "resultSha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            )
            failure_marker.write_text(
                json.dumps(
                    {
                        "id": 0,
                        "leaseToken": lease["leaseToken"],
                        "error": "renderer reported failure after publishing result",
                    }
                )
            )
            barrier = threading.Barrier(2)

            def sync(function, path: Path) -> None:
                barrier.wait()
                function(path)

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(sync, state._sync_filesystem_result, result_marker),
                    executor.submit(sync, state._sync_filesystem_failure, failure_marker),
                ]
                for future in futures:
                    future.result()

            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["result_token"], lease["leaseToken"])
            self.assertEqual(Path(row["result_path"]).read_bytes(), payload)
            self.assertFalse(failure_marker.exists())
            self.assertFalse(result_marker.exists())

    def test_stale_shared_result_cleanup_preserves_newer_lease_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_root = root / "queue"
            state = self._state(root / "spool", queue_root=queue_root)
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued")
            current = state.lease("current-worker")
            current_token = current["leaseToken"]
            stale_token = "stale-token"
            bundle = self._result_tar(root, {"video": self._valid_video_metadata()})
            payload = bundle.read_bytes()

            def token_paths(token: str) -> list[Path]:
                stem = state._lease_stem(0, token)
                return [
                    queue_root / "claims" / f"{stem}.json",
                    queue_root / "grants" / f"{stem}.json",
                    queue_root / "leased" / f"{stem}.json",
                    queue_root / "heartbeats" / f"{stem}.json",
                    queue_root / "failures" / f"{stem}.failure.json",
                    queue_root / "results" / f"{stem}.tar",
                    queue_root / "results" / f"{stem}.result.json",
                ]

            stale_paths = token_paths(stale_token)
            current_paths = token_paths(current_token)
            for path in [*stale_paths, *current_paths]:
                path.write_text("placeholder")
            stale_result = stale_paths[-2]
            stale_marker = stale_paths[-1]
            stale_result.write_bytes(payload)
            stale_marker.write_text(
                json.dumps(
                    {
                        "id": 0,
                        "leaseToken": stale_token,
                        "resultPath": str(stale_result),
                        "resultBytes": len(payload),
                        "resultSha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            )

            state._sync_filesystem_result(stale_marker)

            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(row["status"], "leased")
            self.assertEqual(row["lease_token"], current_token)
            self.assertTrue(all(path.exists() for path in current_paths))
            self.assertTrue(all(not path.exists() for path in stale_paths))

    def test_shared_cleanup_keeps_marker_until_payload_is_confirmed_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_root = root / "queue"
            state = self._state(root / "spool", queue_root=queue_root)
            self.addCleanup(state.db.close)
            token = "lease-token"
            stem = state._lease_stem(0, token)
            payload = queue_root / "results" / f"{stem}.tar"
            marker = queue_root / "results" / f"{stem}.result.json"
            payload.write_bytes(b"result")
            marker.write_text("{}")

            with mock.patch.object(state, "_unlink_shared_paths", return_value=False):
                self.assertFalse(state._cleanup_shared_lease(0, token, include_result=True))

            self.assertTrue(payload.exists())
            self.assertTrue(marker.exists())
            self.assertTrue(state._cleanup_shared_lease(0, token, include_result=True))
            self.assertFalse(payload.exists())
            self.assertFalse(marker.exists())

    def test_startup_sweeps_only_stale_untracked_shared_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_root = root / "queue"
            state = self._state(root / "spool", queue_root=queue_root)
            self.addCleanup(state.db.close)
            results = queue_root / "results"
            stale = results / "000000-stale.tar"
            fresh = results / "000001-fresh.tar"
            tracked = results / "000002-tracked.tar"
            for path in (stale, fresh, tracked):
                path.write_bytes(b"result")
            old = time.time() - state.lease_seconds * 3
            os.utime(stale, (old, old))
            os.utime(tracked, (old, old))
            tracked.with_name("000002-tracked.result.json").write_text("{}")

            with mock.patch.object(daytona_module, "_json_line"):
                state._sweep_stale_shared_results()

            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(tracked.exists())

    def test_queue_sync_reconciles_durable_lease_before_result_after_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_root = root / "queue"
            state = self._state(root / "spool", queue_root=queue_root)
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "queued", attempts=1)
            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            token = "durable-lease"
            stem = state._lease_stem(0, token)
            lease_path = queue_root / "leased" / f"{stem}.json"
            lease_path.write_text(
                json.dumps(
                    {
                        "id": 0,
                        "reference": row["reference"],
                        "sourcePath": row["source_path"],
                        "sourceBytes": row["source_bytes"],
                        "sourceSha256": row["source_sha256"],
                        "attempts": 1,
                        "leaseToken": token,
                        "workerId": "worker-before-restart",
                        "leaseDeadline": time.time() + state.lease_seconds,
                    }
                )
            )
            bundle = self._result_tar(root, {"video": self._valid_video_metadata()})
            payload = bundle.read_bytes()
            shared_result = queue_root / "results" / f"{stem}.tar"
            result_marker = queue_root / "results" / f"{stem}.result.json"
            shared_result.write_bytes(payload)
            result_marker.write_text(
                json.dumps(
                    {
                        "id": 0,
                        "leaseToken": token,
                        "resultPath": str(shared_result),
                        "resultBytes": len(payload),
                        "resultSha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            )
            reconcile_lease = state._sync_filesystem_lease
            accept_result = state._sync_filesystem_result

            def delayed_reconcile(path: Path) -> None:
                time.sleep(0.05)
                reconcile_lease(path)

            def accept_then_stop(path: Path) -> None:
                try:
                    accept_result(path)
                finally:
                    state.stop.set()

            with mock.patch.object(
                state, "_sync_filesystem_lease", side_effect=delayed_reconcile
            ), mock.patch.object(
                state, "_sync_filesystem_result", side_effect=accept_then_stop
            ):
                state.sync_filesystem()

            committed = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(committed["status"], "complete")
            self.assertEqual(committed["result_token"], token)
            self.assertEqual(Path(committed["result_path"]).read_bytes(), payload)
            self.assertFalse(shared_result.exists())
            self.assertFalse(result_marker.exists())

    def test_result_tar_rejects_wrong_video_contract_or_extra_members(self):
        valid_video = self._valid_video_metadata()
        invalid_values = {
            "width": 255,
            "height": 207,
            "frameRate": "60/1",
            "container": "nut",
            "codec": "ffv1",
            "pixelFormat": "bgr0",
            "sourceFps": 30,
            "targetFps": 10,
            "firstSelectedSlpFrame": -40,
            "sourceFrameStep": 2,
            "cpuOnly": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self._state(root / "spool")
            self.addCleanup(state.db.close)
            for field, invalid_value in invalid_values.items():
                with self.subTest(field=field):
                    video = dict(valid_video)
                    video[field] = invalid_value
                    bundle = self._result_tar(root, {"video": video})
                    with self.assertRaisesRegex(ValueError, f"video {field}"):
                        state._validate_result(bundle, 1)

            bundle = self._result_tar(root, {"video": valid_video}, extra_member=True)
            with self.assertRaisesRegex(ValueError, "invalid result members"):
                state._validate_result(bundle, 1)

            for field, invalid_value in (
                ("croppedTailSourceFrames", 3),
                ("lastSelectedSlpFrame", 118),
            ):
                with self.subTest(field=field):
                    video = dict(valid_video)
                    video[field] = invalid_value
                    bundle = self._result_tar(root, {"video": video})
                    with self.assertRaises(ValueError):
                        state._validate_result(bundle, 1)

    def test_terminal_state_waits_for_producer_and_all_active_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(Path(temporary))
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "uploaded")

            self.assertFalse(state.is_terminal())
            state.producer_done = True
            self.assertTrue(state.is_terminal())
            self.assertEqual(state.health()["state"], "complete")

            for status in ("pending", "queued", "leased", "complete", "uploading"):
                with self.subTest(status=status), state.db:
                    state.db.execute("UPDATE jobs SET status=? WHERE id=0", (status,))
                    self.assertFalse(state.is_terminal())

            with state.db:
                state.db.execute(
                    "UPDATE jobs SET status='failed', error='render failed' WHERE id=0"
                )
            self.assertTrue(state.is_terminal())
            health = state.health()
            self.assertEqual(health["state"], "failed")
            self.assertEqual(health["counts"]["failed"], 1)
            self.assertEqual(health["errors"][0]["error"], "render failed")

    def test_rendering_done_requires_producer_and_no_renderable_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue_root = root / "queue"
            state = self._state(root / "spool", queue_root=queue_root)
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "complete")
            done = queue_root / "rendering.done"

            state._update_rendering_done()
            self.assertFalse(done.exists())
            state.producer_done = True
            state._update_rendering_done()
            self.assertTrue(done.exists())

            for status in ("pending", "queued", "leased"):
                with self.subTest(status=status), state.db:
                    state.db.execute("UPDATE jobs SET status=? WHERE id=0", (status,))
                    state._update_rendering_done()
                    self.assertFalse(done.exists())
                    state.db.execute("UPDATE jobs SET status='complete' WHERE id=0")
                    state._update_rendering_done()
                    self.assertTrue(done.exists())

            with state.db:
                state.db.execute("UPDATE jobs SET status='queued' WHERE id=0")
            state._publish_pending(0)
            self.assertFalse(done.exists())
            self.assertTrue((queue_root / "pending" / "000000.json").exists())

    def test_coordinator_restart_reconciles_durable_and_inflight_statuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            original = self._state(spool)
            try:
                statuses = ["leased", "uploading", "complete", "queued", "failed", "pending"]
                for job_id, status in enumerate(statuses):
                    self._insert_job(original, job_id, status)
                with original.db:
                    original.db.execute(
                        "UPDATE jobs SET lease_token='old-lease', lease_deadline=9999999999, "
                        "worker_id='old-worker' WHERE id=0"
                    )
                    original.db.execute(
                        "UPDATE jobs SET result_token='complete-token', "
                        "result_sha256='complete-sha' WHERE id=1"
                    )
                    original.db.execute(
                        "UPDATE jobs SET result_token='orphan-token', "
                        "result_sha256='orphan-sha' WHERE id=2"
                    )
                uploading_result = Path(
                    original.db.execute(
                        "SELECT result_path FROM jobs WHERE id=1"
                    ).fetchone()[0]
                )
                uploading_result.write_bytes(b"durable-result")
                missing_source = Path(
                    original.db.execute(
                        "SELECT source_path FROM jobs WHERE id=3"
                    ).fetchone()[0]
                )
                missing_source.unlink()
                (original.incoming / "source.partial").write_bytes(b"partial")
                (original.results / "result.partial").write_bytes(b"partial")
            finally:
                original.db.close()

            restarted = self._state(spool)
            try:
                references = [
                    (
                        f"archive.tar.zst::games/{job_id}.slp",
                        "archive.tar.zst",
                        f"games/{job_id}.slp",
                    )
                    for job_id in range(6)
                ]
                restarted._source_references = mock.Mock(return_value=references)
                restarted._uploaded_references = mock.Mock(
                    return_value={references[5][0]}
                )
                with mock.patch.object(daytona_module, "_json_line"):
                    restarted.initialize()

                rows = {
                    int(row["id"]): row
                    for row in restarted.db.execute("SELECT * FROM jobs ORDER BY id")
                }
                self.assertEqual(
                    {job_id: row["status"] for job_id, row in rows.items()},
                    {
                        0: "queued",
                        1: "complete",
                        2: "pending",
                        3: "pending",
                        4: "failed",
                        5: "uploaded",
                    },
                )
                self.assertIsNone(rows[0]["lease_token"])
                self.assertIsNone(rows[0]["lease_deadline"])
                self.assertIsNone(rows[0]["worker_id"])
                self.assertEqual(rows[1]["result_token"], "complete-token")
                self.assertEqual(rows[1]["result_sha256"], "complete-sha")
                self.assertIsNone(rows[2]["result_token"])
                self.assertIsNone(rows[2]["result_sha256"])
                self.assertEqual(list(restarted.incoming.glob("*.partial")), [])
                self.assertEqual(list(restarted.results.glob("*.partial")), [])
            finally:
                restarted.db.close()

    def test_shared_queue_restart_preserves_fresh_partial_and_readopts_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spool = root / "spool"
            queue_root = root / "queue"
            lease_token = "durable-lease"
            initial = self._state(spool, queue_root=queue_root)
            try:
                self._insert_job(initial, 0, "leased", attempts=1)
                row = initial.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
                deadline = time.time() + initial.lease_seconds
                with initial.db:
                    initial.db.execute(
                        "UPDATE jobs SET lease_token=?, lease_deadline=?, worker_id=? WHERE id=0",
                        (lease_token, deadline, "worker-before-restart"),
                    )
                stem = initial._lease_stem(0, lease_token)
                lease_path = queue_root / "leased" / f"{stem}.json"
                lease_path.write_text(
                    json.dumps(
                        {
                            "id": 0,
                            "reference": row["reference"],
                            "sourcePath": row["source_path"],
                            "sourceBytes": row["source_bytes"],
                            "sourceSha256": row["source_sha256"],
                            "attempts": 1,
                            "leaseToken": lease_token,
                            "workerId": "worker-before-restart",
                            "leaseDeadline": deadline,
                            "leaseSeconds": initial.lease_seconds,
                            "granted": True,
                        }
                    )
                )
                fresh_partial = queue_root / "results" / f".{stem}.fresh.partial"
                fresh_partial.write_bytes(b"in-flight-result")
            finally:
                initial.db.close()

            restarted = self._state(spool, queue_root=queue_root)
            try:
                references = [
                    (
                        "archive.tar.zst::games/0.slp",
                        "archive.tar.zst",
                        "games/0.slp",
                    )
                ]
                restarted._source_references = mock.Mock(return_value=references)
                restarted._uploaded_references = mock.Mock(return_value=set())
                with mock.patch.object(daytona_module, "_json_line"):
                    restarted.initialize()

                queued = restarted.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
                self.assertEqual(queued["status"], "queued")
                self.assertTrue(fresh_partial.exists())
                self.assertFalse((queue_root / "pending" / "000000.json").exists())

                restarted._sync_filesystem_lease(lease_path)
                adopted = restarted.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
                self.assertEqual(adopted["status"], "leased")
                self.assertEqual(adopted["lease_token"], lease_token)
                self.assertEqual(adopted["worker_id"], "worker-before-restart")
                self.assertEqual(adopted["attempts"], 1)
                self.assertTrue(
                    (queue_root / "grants" / f"{stem}.json").exists()
                )
                self.assertFalse((queue_root / "pending" / "000000.json").exists())
            finally:
                restarted.db.close()

    def test_upload_claim_waits_for_batch_unless_backpressure_requires_flush(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            waiting = self._state(root / "waiting", prefetch=8, upload_batch_size=3)
            pressured = self._state(root / "pressured", prefetch=2, upload_batch_size=3)
            grouped = self._state(
                root / "grouped",
                prefetch=4,
                upload_batch_size=8,
                upload_min_batch=3,
            )
            self.addCleanup(waiting.db.close)
            self.addCleanup(pressured.db.close)
            self.addCleanup(grouped.db.close)

            self._insert_job(waiting, 0, "complete")
            self._insert_job(waiting, 1, "pending")
            self.assertEqual(waiting._claim_upload_batch(), [])
            self.assertEqual(
                waiting.db.execute("SELECT status FROM jobs WHERE id=0").fetchone()[0],
                "complete",
            )

            self._insert_job(pressured, 0, "complete")
            self._insert_job(pressured, 1, "queued")
            claimed = pressured._claim_upload_batch()
            self.assertEqual([row["id"] for row in claimed], [0])
            self.assertEqual(
                pressured.db.execute("SELECT status FROM jobs WHERE id=0").fetchone()[0],
                "uploading",
            )

            self._insert_job(grouped, 0, "complete")
            self._insert_job(grouped, 1, "complete")
            self._insert_job(grouped, 2, "queued")
            self._insert_job(grouped, 3, "queued")
            self.assertEqual(grouped._claim_upload_batch(), [])
            self._insert_job(grouped, 4, "complete")
            self.assertEqual(
                [row["id"] for row in grouped._claim_upload_batch()],
                [0, 1, 4],
            )

    def test_parallel_upload_claims_are_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(
                Path(temporary),
                prefetch=8,
                upload_batch_size=2,
                upload_concurrency=4,
            )
            self.addCleanup(state.db.close)
            for job_id in range(8):
                self._insert_job(state, job_id, "complete")

            with ThreadPoolExecutor(max_workers=4) as executor:
                batches = list(executor.map(lambda _: state._claim_upload_batch(), range(4)))

            claimed = [int(row["id"]) for batch in batches for row in batch]
            self.assertEqual(sorted(claimed), list(range(8)))
            self.assertEqual(len(claimed), len(set(claimed)))
            statuses = {
                row[0]
                for row in state.db.execute("SELECT status FROM jobs ORDER BY id")
            }
            self.assertEqual(statuses, {"uploading"})

    def test_result_capacity_is_reserved_before_concurrent_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(
                Path(temporary),
                prefetch=8,
                spool_max_bytes=1000,
            )
            self.addCleanup(state.db.close)
            self._insert_job(
                state,
                0,
                "leased",
                source_bytes=200,
                result_bytes=0,
            )
            state._reserve_result_capacity(700)

            acquired = threading.Event()
            errors: Queue[BaseException] = Queue()

            def reserve_second_result():
                try:
                    state._reserve_result_capacity(200)
                    acquired.set()
                except BaseException as error:
                    errors.put(error)

            thread = threading.Thread(target=reserve_second_result)
            thread.start()
            self.assertFalse(acquired.wait(0.05))
            self.assertEqual(state._reserved_result_bytes, 700)
            state._release_result_capacity(700)
            self.assertTrue(acquired.wait(1))
            self.assertEqual(state._reserved_result_bytes, 200)
            state._release_result_capacity(200)
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive())
            self.assertTrue(errors.empty())

    def test_upload_attempt_budget_persists_across_uploader_threads(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(daytona_module.time, "time", return_value=100.0) as clock:
                state = self._state(
                    Path(temporary),
                    prefetch=1,
                    upload_batch_size=1,
                    upload_concurrency=8,
                    upload_max_attempts=3,
                )
                self.addCleanup(state.db.close)
                self._insert_job(state, 0, "complete")

                for attempt in range(1, 4):
                    rows = state._claim_upload_batch()
                    self.assertEqual(int(rows[0]["upload_attempts"]), attempt)
                    exhausted = state._record_upload_failure(
                        rows, RuntimeError("Drive failed")
                    )
                    self.assertEqual(exhausted, attempt == 3)
                    status = state.db.execute(
                        "SELECT status FROM jobs WHERE id=0"
                    ).fetchone()[0]
                    self.assertEqual(status, "failed" if attempt == 3 else "complete")
                    if attempt < 3:
                        self.assertEqual(state._claim_upload_batch(), [])
                        clock.return_value += daytona_module.UPLOAD_RETRY_SECONDS + 0.001

    def test_upload_failure_backoff_is_global_and_persists_across_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary)
            with mock.patch.object(daytona_module.time, "time", return_value=100.0) as clock:
                state = self._state(
                    spool,
                    prefetch=4,
                    upload_batch_size=2,
                    upload_concurrency=4,
                )
                for job_id in range(4):
                    self._insert_job(state, job_id, "complete")

                rows = state._claim_upload_batch()
                self.assertEqual([int(row["id"]) for row in rows], [0, 1])
                self.assertFalse(
                    state._record_upload_failure(rows, RuntimeError("Drive failed"))
                )
                self.assertEqual(state._claim_upload_batch(), [])
                attempts = list(
                    state.db.execute(
                        "SELECT id, upload_attempts FROM jobs ORDER BY id"
                    )
                )
                self.assertEqual(
                    [(int(row["id"]), int(row["upload_attempts"])) for row in attempts],
                    [(0, 1), (1, 1), (2, 0), (3, 0)],
                )
                deadlines = {
                    float(row[0])
                    for row in state.db.execute(
                        "SELECT upload_not_before FROM jobs "
                        "WHERE upload_not_before IS NOT NULL"
                    )
                }
                self.assertEqual(deadlines, {105.0})
                state.db.close()

                restarted = self._state(
                    spool,
                    prefetch=4,
                    upload_batch_size=2,
                    upload_concurrency=4,
                )
                try:
                    clock.return_value = 104.999
                    self.assertEqual(restarted._claim_upload_batch(), [])
                    clock.return_value = 105.001
                    claimed = restarted._claim_upload_batch()
                    self.assertEqual([int(row["id"]) for row in claimed], [0, 1])
                    self.assertEqual(
                        [int(row["upload_attempts"]) for row in claimed], [2, 2]
                    )
                finally:
                    restarted.db.close()

    def test_drive_query_rate_limit_refunds_attempt_and_stops_all_uploaders(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(
                Path(temporary),
                prefetch=1,
                upload_batch_size=1,
                upload_concurrency=8,
                upload_max_attempts=3,
            )
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "complete")
            rows = state._claim_upload_batch()
            self.assertEqual(int(rows[0]["upload_attempts"]), 1)
            error = RuntimeError(
                "Quota exceeded for quota metric 'Queries': rateLimitExceeded"
            )

            with (
                mock.patch.object(daytona_module.time, "monotonic", return_value=100.0),
                mock.patch.object(daytona_module.time, "time", return_value=1000.0),
            ):
                retry_after = state._record_drive_rate_limit(rows, error)
                self.assertEqual(state._claim_upload_batch(), [])

            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(row["status"], "complete")
            self.assertEqual(int(row["upload_attempts"]), 0)
            self.assertEqual(float(row["upload_not_before"]), 1060.0)
            self.assertEqual(retry_after, 60.0)
            self.assertTrue(state._is_drive_rate_limit(error))
            self.assertFalse(state._is_drive_rate_limit(RuntimeError("connection reset")))

            state.stop.set()
            self.assertFalse(state._wait_for_drive_backoff())

    def test_stream_archive_surfaces_rclone_rate_limit_after_broken_pipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(Path(temporary))
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "complete")
            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            rclone = mock.Mock(
                stdin=io.BytesIO(),
                stderr=io.BytesIO(b"googleapi: Error 403: rateLimitExceeded\n"),
            )
            rclone.wait.return_value = 1
            zstd = mock.Mock(
                stdin=io.BytesIO(),
                stderr=io.BytesIO(b"zstd: Write error: Broken pipe\n"),
            )
            zstd.wait.return_value = 1
            archive = mock.MagicMock()
            archive.__enter__.return_value.add.side_effect = BrokenPipeError("broken pipe")

            with (
                mock.patch.object(
                    daytona_module.subprocess, "Popen", side_effect=[rclone, zstd]
                ),
                mock.patch.object(daytona_module.tarfile, "open", return_value=archive),
                self.assertRaises(RuntimeError) as raised,
            ):
                state._stream_result_archive([row], "target/archive.tar.zst")

            self.assertIn("rateLimitExceeded", str(raised.exception))
            self.assertTrue(state._is_drive_rate_limit(raised.exception))
            rclone.kill.assert_called_once_with()
            zstd.kill.assert_called_once_with()

    def test_stream_archive_prefers_rclone_error_when_children_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(Path(temporary))
            self.addCleanup(state.db.close)
            rclone = mock.Mock(
                stdin=io.BytesIO(),
                stderr=io.BytesIO(b"googleapi: Error 403: rateLimitExceeded\n"),
            )
            rclone.wait.return_value = 1
            zstd = mock.Mock(
                stdin=io.BytesIO(),
                stderr=io.BytesIO(b"zstd: Write error: Broken pipe\n"),
            )
            zstd.wait.return_value = 1
            archive = mock.MagicMock()

            with (
                mock.patch.object(
                    daytona_module.subprocess, "Popen", side_effect=[rclone, zstd]
                ),
                mock.patch.object(daytona_module.tarfile, "open", return_value=archive),
                self.assertRaises(RuntimeError) as raised,
            ):
                state._stream_result_archive([], "target/archive.tar.zst")

            self.assertIn("rateLimitExceeded", str(raised.exception))
            self.assertTrue(state._is_drive_rate_limit(raised.exception))

    def test_committed_batch_cleans_local_files_before_releasing_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(
                Path(temporary),
                prefetch=1,
                upload_batch_size=1,
                upload_concurrency=8,
            )
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "complete")
            row = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            Path(row["result_path"]).write_bytes(b"result")
            rows = state._claim_upload_batch()
            state._stream_result_archive = mock.Mock()
            state.drive.upload_file = mock.Mock()

            state._upload_rows(rows)

            committed = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(committed["status"], "uploaded")
            self.assertIsNone(committed["source_path"])
            self.assertIsNone(committed["result_path"])
            self.assertFalse(Path(row["source_path"]).exists())
            self.assertFalse(Path(row["result_path"]).exists())
            state.drive.upload_file.assert_called_once()
            self.assertEqual(state.drive.upload_file.call_args.kwargs["tps_limit"], 1)

    def test_initialize_sweeps_files_already_committed_by_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(Path(temporary))
            self.addCleanup(state.db.close)
            self._insert_job(state, 0, "complete")
            before = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            Path(before["result_path"]).write_bytes(b"result")
            reference = str(before["reference"])
            state._source_references = mock.Mock(
                return_value=[(reference, str(before["archive"]), str(before["member"]))]
            )
            state._uploaded_references = mock.Mock(return_value={reference})

            with mock.patch.object(daytona_module, "_json_line"):
                state.initialize()

            after = state.db.execute("SELECT * FROM jobs WHERE id=0").fetchone()
            self.assertEqual(after["status"], "uploaded")
            self.assertIsNone(after["source_path"])
            self.assertIsNone(after["result_path"])
            self.assertFalse(Path(before["source_path"]).exists())
            self.assertFalse(Path(before["result_path"]).exists())

    def test_uploaded_references_are_loaded_by_one_bulk_manifest_sync(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = self._state(Path(temporary))
            self.addCleanup(state.db.close)

            def download(remote_root: str, destination: Path) -> list[Path]:
                self.assertEqual(remote_root, "target/batches")
                destination.mkdir(parents=True, exist_ok=True)
                first = destination / "batch-a.manifest.jsonl"
                second = destination / "batch-b.manifest.jsonl"
                first.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "sourceReference": "archive::one.slp",
                        }
                    )
                    + "\n"
                )
                second.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "sourceReference": "archive::two.slp",
                        }
                    )
                    + "\n"
                )
                return [first, second]

            state.drive.download_manifests = mock.Mock(side_effect=download)
            state.drive.cat = mock.Mock(side_effect=AssertionError("sequential cat forbidden"))

            self.assertEqual(
                state._uploaded_references(),
                {"archive::one.slp", "archive::two.slp"},
            )
            state.drive.download_manifests.assert_called_once()
            state.drive.cat.assert_not_called()

    def test_upload_claim_flushes_on_batch_bytes_and_final_drain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            by_count = self._state(root / "count", upload_batch_size=3)
            by_bytes = self._state(root / "bytes", upload_batch_size=3, upload_max_bytes=100)
            final = self._state(root / "final", upload_batch_size=3)
            self.addCleanup(by_count.db.close)
            self.addCleanup(by_bytes.db.close)
            self.addCleanup(final.db.close)

            for job_id in range(3):
                self._insert_job(by_count, job_id, "complete")
            self.assertEqual(
                [row["id"] for row in by_count._claim_upload_batch()],
                [0, 1, 2],
            )

            self._insert_job(by_bytes, 0, "complete", source_bytes=40, result_bytes=60)
            self._insert_job(by_bytes, 1, "complete", source_bytes=1, result_bytes=1)
            self.assertEqual(
                [row["id"] for row in by_bytes._claim_upload_batch()],
                [0],
            )
            self.assertEqual(
                by_bytes.db.execute("SELECT status FROM jobs WHERE id=1").fetchone()[0],
                "complete",
            )

            self._insert_job(final, 0, "complete")
            final.producer_done = True
            self.assertEqual(
                [row["id"] for row in final._claim_upload_batch()],
                [0],
            )

    def test_runner_build_and_run_surface_is_daytona_only(self):
        tree = ast.parse(inspect.getsource(runner_module))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        self.assertFalse(
            any("runpod" in module.lower() for module in imported_modules),
            imported_modules,
        )

        daytona = mock.Mock()
        daytona.run.return_value = {"state": "complete"}
        with mock.patch.dict(
            os.environ,
            {
                "SMASH_STORAGE_PROVIDER": "gdrive",
                "SMASH_SAMPLE_LIMIT": "12",
                "SMASH_WORKER_COUNT": "34",
            },
            clear=True,
        ), mock.patch.object(runner_module, "DaytonaConnector", return_value=daytona):
            runner = runner_module.build_runner()

        self.assertEqual(runner.run(), {"state": "complete"})
        daytona.run.assert_called_once_with(sample_limit=12, worker_count=34)
        self.assertFalse(hasattr(runner, "runpod"))

    def test_coordinator_defaults_use_rate_safe_parallel_archives(self):
        args = daytona_module.build_parser().parse_args(
            [
                "coordinator",
                "--remote",
                "drive",
                "--source-root",
                "source",
                "--target-root",
                "target",
                "--config",
                "/tmp/rclone.conf",
                "--spool",
                "/tmp/spool",
            ]
        )
        self.assertEqual(args.prefetch, 512)
        self.assertEqual(args.upload_batch_size, 64)
        self.assertEqual(args.upload_min_batch, 32)
        self.assertEqual(args.upload_concurrency, 4)
        self.assertEqual(args.upload_tps_limit, 1)

    def test_raw_packet_pts_bounds_preserve_a_head_gap(self):
        completed = mock.Mock(stdout="0.016667,\n0.033333,\n1.000000,\n")
        with mock.patch.object(daytona_module, "_run", return_value=completed):
            self.assertEqual(
                daytona_module._packet_pts_bounds(Path("capture.avi")),
                (0.016667, 1.0),
            )

    def test_runner_rejects_non_drive_provider_without_constructing_daytona(self):
        with mock.patch.dict(
            os.environ, {"SMASH_STORAGE_PROVIDER": "hf"}, clear=True
        ), mock.patch.object(runner_module, "DaytonaConnector") as constructor:
            with self.assertRaisesRegex(ValueError, "requires Google Drive"):
                runner_module.build_runner()
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
