from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).parent
sys.path.insert(0, str(MODULE_ROOT))
SPEC = importlib.util.spec_from_file_location("daytona_connector", MODULE_ROOT / "daytona_connector.py")
assert SPEC and SPEC.loader
daytona_connector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = daytona_connector
SPEC.loader.exec_module(daytona_connector)


class ResumeReconciliationTest(unittest.TestCase):
    def test_preserves_uncommitted_results_and_ready_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("sources", "ready", "results", "done", "failures", "workers",
                         "upload-queue", "upload-acks"):
                (root / name).mkdir()
            jobs = [{"id": job_id, "reference": f"archive::job-{job_id}"}
                    for job_id in range(3)]

            committed_source = root / "sources/000000.slp"
            committed_source.write_bytes(b"committed")
            committed_result = root / "results/000000.tar"
            committed_result.write_bytes(b"old")
            (root / "results/000000.json").write_text(json.dumps({
                **jobs[0], "sourcePath": str(committed_source), "sourceBytes": committed_source.stat().st_size,
                "resultPath": str(committed_result), "resultBytes": committed_result.stat().st_size,
            }))

            result_source = root / "sources/000001.slp"
            result_source.write_bytes(b"source-one")
            result_tar = root / "results/000001.tar"
            result_tar.write_bytes(b"result-one")
            (root / "results/000001.json").write_text(json.dumps({
                **jobs[1], "sourcePath": str(result_source), "sourceBytes": result_source.stat().st_size,
                "resultPath": str(result_tar), "resultBytes": result_tar.stat().st_size,
            }))
            (root / "ready/000001.json").write_text(json.dumps({
                **jobs[1], "sourcePath": str(result_source), "sourceBytes": result_source.stat().st_size,
            }))

            ready_source = root / "sources/000002.slp"
            ready_source.write_bytes(b"source-two")
            (root / "ready/000002.json").write_text(json.dumps({
                **jobs[2], "sourcePath": str(ready_source), "sourceBytes": ready_source.stat().st_size,
            }))
            (root / "done/000000.json").write_text("{}")
            (root / "failures/worker.json").write_text('{"error":"old failure"}')
            (root / "workers/0000.done").touch()
            (root / "upload-queue/000.json").write_text("{}")
            (root / "upload-acks/000.json").write_text("{}")
            for name in ("stop", "producer.done", "fleet.json"):
                (root / name).touch()

            summary = daytona_connector._prepare_resume(root, jobs, {jobs[0]["reference"]})

            self.assertEqual(summary["preservedResults"], 1)
            self.assertEqual(summary["preservedReady"], 1)
            self.assertTrue((root / "results/000001.json").is_file())
            self.assertTrue(result_tar.is_file())
            self.assertTrue(result_source.is_file())
            self.assertFalse((root / "ready/000001.json").exists())
            self.assertTrue((root / "ready/000002.json").is_file())
            self.assertTrue(ready_source.is_file())
            self.assertFalse((root / "results/000000.json").exists())
            self.assertFalse(committed_source.exists())
            for name in ("done", "failures", "workers", "upload-queue", "upload-acks"):
                self.assertEqual(list((root / name).glob("*")), [])
            for name in ("stop", "producer.done", "fleet.json"):
                self.assertFalse((root / name).exists())
            recovery = list((root / "diagnostics/recoveries").glob("*.json"))
            self.assertEqual(len(recovery), 1)
            self.assertEqual(json.loads(recovery[0].read_text())["previousFailures"][0]["file"],
                             "worker.json")


class SupervisorStateTest(unittest.TestCase):
    def test_transient_daytona_failure_is_a_missed_poll(self) -> None:
        connector = object.__new__(daytona_connector.DaytonaConnector)
        connector._exec = lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="request timeout"
        )
        coordinator = daytona_connector.Sandbox("id", "coordinator", 2, 4, 10, 0, "started")

        self.assertIsNone(connector._state(coordinator))

    def test_valid_state_is_returned(self) -> None:
        connector = object.__new__(daytona_connector.DaytonaConnector)
        connector._exec = lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"state":"running","uploaded":855}', stderr=""
        )
        coordinator = daytona_connector.Sandbox("id", "coordinator", 2, 4, 10, 0, "started")

        self.assertEqual(connector._state(coordinator), {"state": "running", "uploaded": 855})


if __name__ == "__main__":
    unittest.main()
