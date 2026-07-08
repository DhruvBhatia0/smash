from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from frame_queue.hf_storage import (  # noqa: E402
    HfDatasetStore,
    HfStorageConfig,
    HfStorageError,
    LocalFileCollector,
    ProcessedFramePublisher,
    join_repo_path,
    normalize_repo_path,
)
from frame_queue.jobs import FrameRenderJob, write_json  # noqa: E402


class HfStorageTests(unittest.TestCase):
    def test_repo_paths_are_normalized_and_cannot_escape(self) -> None:
        self.assertEqual(join_repo_path("/raw/slp/", "foo", "bar.slp"), "raw/slp/foo/bar.slp")
        self.assertEqual(normalize_repo_path("raw//slp"), "raw/slp")
        with self.assertRaises(HfStorageError):
            normalize_repo_path("../private")

    def test_tree_stats_respects_ignore_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "raw-frames").mkdir()
            (root / "raw-frames" / "frame.png").write_bytes(b"abc")
            (root / "result.json").write_text("{}\n")

            stats = LocalFileCollector.tree_stats(root, ignore_patterns=["raw-frames/**"])

            self.assertEqual(stats.file_count, 1)
            self.assertEqual(stats.byte_count, 3)

    def test_expected_email_mismatch_fails_closed(self) -> None:
        store = HfDatasetStore(
            HfStorageConfig(
                repo_id="user/dataset",
                token="redacted",
                expected_email="expected@example.com",
            )
        )
        store._identity_result = {
            "status": "verified",
            "email": "actual@example.com",
            "name": "actual-user",
        }

        with self.assertRaisesRegex(HfStorageError, "HF token email mismatch"):
            store.verify_expected_identity()

    def test_expected_email_match_passes(self) -> None:
        store = HfDatasetStore(
            HfStorageConfig(
                repo_id="user/dataset",
                token="redacted",
                expected_email="expected@example.com",
            )
        )
        store._identity_result = {
            "status": "verified",
            "email": "expected@example.com",
            "name": "expected-user",
        }

        identity = store.verify_expected_identity()

        self.assertEqual(identity["name"], "expected-user")

    def test_processed_cleanup_prunes_bulky_files_but_keeps_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slp = root / "sample.slp"
            slp.write_bytes(b"slp")
            job = FrameRenderJob.from_slp(
                slp_path=slp,
                root_dir=root,
                run_id="run",
                start_frame=None,
                end_frame=None,
            )
            job.ensure_dirs()
            (job.raw_frame_dir / "frame.png").write_bytes(b"png")
            write_json(job.result_path, {"status": "processed"})

            publisher = ProcessedFramePublisher(
                store=HfDatasetStore(
                    HfStorageConfig(repo_id="user/dataset", token="redacted")
                ),
                path_prefix="processed/frame-queues",
                delete_local_after_upload=True,
            )
            cleanup = publisher.cleanup_job(job)

            self.assertEqual(cleanup["status"], "pruned")
            self.assertTrue(job.result_path.exists())
            self.assertFalse(job.raw_frame_dir.exists())


if __name__ == "__main__":
    unittest.main()
