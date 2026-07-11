#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


VIDEO_ROOT = "slp_with_video"
REPORT_NAME = "migration-report.json"
CONVERSION_LOG_NAME = "conversions.jsonl"


def log(event: str, **fields):
    print(json.dumps({"event": event, "time": round(time.time(), 3), **fields}), flush=True)


def write_json(path: Path, value: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def inventory(repo: str, token: str) -> dict:
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{repo}?blobs=true",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)
    siblings = data.get("siblings", [])
    videos = [row for row in siblings if row["rfilename"].endswith("/video.avi")]
    mp4s = [row for row in siblings if row["rfilename"].endswith("/video.mp4")]
    input_slps = [row for row in siblings if row["rfilename"].endswith("/input.slp")]
    return {
        "sha": data["sha"],
        "private": data["private"],
        "usedStorage": data.get("usedStorage"),
        "files": len(siblings),
        "bytes": sum(int(row.get("size") or 0) for row in siblings),
        "aviCount": len(videos),
        "aviBytes": sum(int(row.get("size") or 0) for row in videos),
        "mp4Count": len(mp4s),
        "inputSlpCount": len(input_slps),
        "videoPaths": [row["rfilename"] for row in videos],
    }


def probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,nb_read_frames,avg_frame_rate,width,height:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    value = json.loads(completed.stdout)
    stream = value["streams"][0]
    frames = stream.get("nb_read_frames")
    if not frames or frames == "N/A":
        raise RuntimeError(f"ffprobe could not count frames in {path}")
    return {
        "codec": stream["codec_name"],
        "frames": int(frames),
        "frameRate": stream["avg_frame_rate"],
        "width": stream["width"],
        "height": stream["height"],
        "duration": float(value["format"]["duration"]),
        "bytes": int(value["format"]["size"]),
    }


class Migration:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.token = os.environ["HF_TOKEN"]
        self.work_dir = Path(args.work_dir)
        self.dataset_dir = self.work_dir / "dataset"
        self.spotcheck_dir = self.work_dir / "spotchecks"
        self.report_path = self.work_dir / REPORT_NAME
        self.conversion_log_path = self.work_dir / CONVERSION_LOG_NAME
        self.log_lock = threading.Lock()
        self.api = HfApi(token=self.token)

    def prepare(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        source = inventory(self.args.repo, self.token)
        report = self._load_report()
        if report and report["source"]["sha"] != source["sha"]:
            raise RuntimeError("HF changed since this migration was prepared; use a fresh work directory")
        if not report:
            report = {
                "repo": self.args.repo,
                "status": "downloading",
                "source": source,
                "settings": {
                    "codec": "libx264",
                    "crf": self.args.crf,
                    "preset": self.args.preset,
                    "encodeWorkers": self.args.encode_workers,
                    "ffmpegThreads": self.args.ffmpeg_threads,
                },
                "converted": 0,
                "failed": 0,
            }
            write_json(self.report_path, report)
        self._require_disk(source["bytes"])

        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        log("download_started", repo=self.args.repo, revision=source["sha"], bytes=source["bytes"])
        snapshot_download(
            repo_id=self.args.repo,
            repo_type="dataset",
            revision=source["sha"],
            token=self.token,
            local_dir=self.dataset_dir,
            max_workers=self.args.download_workers,
        )
        log("download_completed", repo=self.args.repo)

        expected_paths = [Path(path) for path in report["source"]["videoPaths"]]
        missing = [str(path) for path in expected_paths if not (self.dataset_dir / path).exists()]
        if missing:
            raise RuntimeError(f"snapshot is missing {len(missing)} AVI files; first={missing[0]}")
        for relative in expected_paths:
            if not (self.dataset_dir / relative.parent / "input.slp").exists():
                raise RuntimeError(f"missing paired input.slp for {relative}")

        retained = self._spotcheck_paths(expected_paths)
        report["status"] = "converting"
        report["retainedSpotcheckAvis"] = [str(path) for path in sorted(retained)]
        write_json(self.report_path, report)
        previous = self._completed_rows()
        converted = []
        failures = []

        with ThreadPoolExecutor(max_workers=self.args.encode_workers) as executor:
            futures = {
                executor.submit(self._convert_one, relative, relative in retained, previous.get(str(relative))): relative
                for relative in expected_paths
            }
            for index, future in enumerate(as_completed(futures), start=1):
                relative = futures[future]
                try:
                    row = future.result()
                    converted.append(row)
                    self._append_conversion(row)
                    log(
                        "converted",
                        completed=index,
                        total=len(futures),
                        path=str(relative),
                        ratio=row["ratio"],
                    )
                except Exception as error:
                    failures.append({"path": str(relative), "error": str(error)})
                    log("conversion_failed", path=str(relative), error=str(error))
                if index % 10 == 0:
                    report["converted"] = len(converted)
                    report["failed"] = len(failures)
                    write_json(self.report_path, report)

        report["converted"] = len(converted)
        report["failed"] = len(failures)
        report["failures"] = failures
        if failures or len(converted) != len(expected_paths):
            report["status"] = "conversion_failed"
            write_json(self.report_path, report)
            raise RuntimeError(f"conversion failed for {len(failures)} of {len(expected_paths)} videos")

        spotchecks = [self._spotcheck(relative) for relative in sorted(retained)]
        minimum_ssim = min(row["ssim"] for row in spotchecks)
        report.update(
            {
                "status": "prepared" if minimum_ssim >= self.args.minimum_ssim else "spotcheck_failed",
                "inputVideoBytes": sum(row["inputBytes"] for row in converted),
                "outputVideoBytes": sum(row["outputBytes"] for row in converted),
                "spotchecks": spotchecks,
                "minimumSsim": minimum_ssim,
            }
        )
        write_json(self.report_path, report)
        if report["status"] != "prepared":
            raise RuntimeError(f"minimum SSIM {minimum_ssim:.6f} is below {self.args.minimum_ssim}")
        log(
            "prepared",
            videos=report["converted"],
            inputBytes=report["inputVideoBytes"],
            outputBytes=report["outputVideoBytes"],
            minimumSsim=minimum_ssim,
        )

    def cutover(self):
        report = self._load_report()
        if not report or report.get("status") not in {"prepared", "cutover"}:
            raise RuntimeError("a clean prepared migration report is required before cutover")
        if report["converted"] != report["source"]["aviCount"] or report["failed"]:
            raise RuntimeError("conversion report is incomplete")
        if report["minimumSsim"] < self.args.minimum_ssim:
            raise RuntimeError("spotcheck threshold is not satisfied")

        phase = report.get("cutoverPhase", "ready")
        if phase == "ready":
            remote = inventory(self.args.repo, self.token)
            if remote["sha"] != report["source"]["sha"]:
                raise RuntimeError("HF changed after download; refusing destructive cutover")
            report["status"] = "cutover"
            report["cutoverPhase"] = "deleting_avi"
            write_json(self.report_path, report)
            self.api.delete_files(
                repo_id=self.args.repo,
                repo_type="dataset",
                delete_patterns=[f"{VIDEO_ROOT}/*/video.avi"],
                parent_commit=remote["sha"],
                commit_message="Replace raw AVI recordings with frame-complete H.264 MP4",
            )
            report["cutoverPhase"] = "avi_deleted"
            write_json(self.report_path, report)
            phase = "avi_deleted"

        if phase == "avi_deleted":
            self.api.super_squash_history(
                repo_id=self.args.repo,
                repo_type="dataset",
                commit_message="Remove superseded raw video history",
            )
            report["cutoverPhase"] = "history_squashed"
            write_json(self.report_path, report)
            phase = "history_squashed"

        if phase in {"history_squashed", "uploading"}:
            report["cutoverPhase"] = "uploading"
            write_json(self.report_path, report)
            self.api.upload_large_folder(
                repo_id=self.args.repo,
                repo_type="dataset",
                folder_path=self.dataset_dir,
                allow_patterns=[f"{VIDEO_ROOT}/*/video.mp4"],
                num_workers=self.args.upload_workers,
                print_report_every=30,
            )
            report["cutoverPhase"] = "uploaded"
            write_json(self.report_path, report)

        remote = inventory(self.args.repo, self.token)
        expected = report["source"]["aviCount"]
        if remote["aviCount"] != 0:
            raise RuntimeError(f"remote still contains {remote['aviCount']} AVI files")
        if remote["mp4Count"] != expected or remote["inputSlpCount"] != expected:
            raise RuntimeError(
                f"remote pair mismatch: mp4={remote['mp4Count']} slp={remote['inputSlpCount']} expected={expected}"
            )
        report["status"] = "complete"
        report["cutoverPhase"] = "verified"
        report["destination"] = remote
        write_json(self.report_path, report)
        for relative in report.get("retainedSpotcheckAvis", []):
            (self.dataset_dir / relative).unlink(missing_ok=True)
        log("cutover_complete", videos=expected, bytes=remote["bytes"])

    def status(self):
        report = self._load_report()
        if report and "source" in report:
            report["source"] = {key: value for key, value in report["source"].items() if key != "videoPaths"}
        print(json.dumps(report, indent=2, sort_keys=True))

    def _convert_one(self, relative: Path, retain_source: bool, previous: dict | None) -> dict:
        source = self.dataset_dir / relative
        target = source.with_name("video.mp4")
        temporary = source.with_name("video.partial.mp4")
        if previous and target.exists():
            output = probe(target)
            if output["frames"] != previous["frames"] or output["codec"] != "h264":
                raise RuntimeError(f"invalid resumed output: {target}")
            if source.exists() and not retain_source:
                source.unlink()
            return previous
        if not source.exists():
            raise FileNotFoundError(source)

        input_probe = probe(source)
        temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vsync",
            "0",
            "-c:v",
            "libx264",
            "-preset",
            self.args.preset,
            "-crf",
            str(self.args.crf),
            "-g",
            "120",
            "-bf",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(self.args.ffmpeg_threads),
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(completed.stderr.strip())
        output_probe = probe(temporary)
        if output_probe["codec"] != "h264":
            raise RuntimeError(f"expected h264 output, got {output_probe['codec']}")
        if output_probe["frames"] != input_probe["frames"]:
            raise RuntimeError(f"frame count changed: {input_probe['frames']} -> {output_probe['frames']}")
        if output_probe["bytes"] >= input_probe["bytes"]:
            raise RuntimeError(f"output is not smaller: {input_probe['bytes']} -> {output_probe['bytes']}")
        temporary.replace(target)
        if not retain_source:
            source.unlink()
        return {
            "source": str(relative),
            "target": str(relative.with_name("video.mp4")),
            "frames": output_probe["frames"],
            "frameRate": output_probe["frameRate"],
            "width": output_probe["width"],
            "height": output_probe["height"],
            "inputBytes": input_probe["bytes"],
            "outputBytes": output_probe["bytes"],
            "ratio": round(input_probe["bytes"] / output_probe["bytes"], 3),
        }

    def _spotcheck(self, relative: Path) -> dict:
        source = self.dataset_dir / relative
        target = source.with_name("video.mp4")
        source_probe = probe(source)
        ssim = self._ssim(source, target)
        images = []
        sample_dir = self.spotcheck_dir / relative.parent.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        for percentage in (10, 50, 90):
            timestamp = source_probe["duration"] * percentage / 100
            output = sample_dir / f"{percentage:02d}.png"
            command = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                str(timestamp),
                "-i",
                str(source),
                "-ss",
                str(timestamp),
                "-i",
                str(target),
                "-filter_complex",
                "[0:v][1:v]hstack=inputs=2",
                "-frames:v",
                "1",
                str(output),
            ]
            subprocess.run(command, check=True)
            images.append(str(output.relative_to(self.work_dir)))
        return {"source": str(relative), "ssim": ssim, "images": images}

    def _ssim(self, source: Path, target: Path) -> float:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "info",
                "-i",
                str(target),
                "-i",
                str(source),
                "-lavfi",
                "[0:v]setpts=PTS-STARTPTS,format=yuv420p[dist];"
                "[1:v]setpts=PTS-STARTPTS,format=yuv420p[ref];[dist][ref]ssim",
                "-f",
                "null",
                "-",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-2000:])
        matches = re.findall(r"All:([0-9.]+)", completed.stderr)
        if not matches:
            raise RuntimeError("ffmpeg did not report SSIM")
        return float(matches[-1])

    def _spotcheck_paths(self, paths: list[Path]) -> set[Path]:
        count = min(self.args.spotcheck_count, len(paths))
        if count == 0:
            raise RuntimeError("at least one spotcheck is required")
        if count == 1:
            return {paths[len(paths) // 2]}
        return {paths[round(index * (len(paths) - 1) / (count - 1))] for index in range(count)}

    def _append_conversion(self, row: dict):
        with self.log_lock:
            with self.conversion_log_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _completed_rows(self) -> dict[str, dict]:
        if not self.conversion_log_path.exists():
            return {}
        rows = {}
        for line in self.conversion_log_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["source"]] = row
        return rows

    def _require_disk(self, source_bytes: int):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        current_bytes = sum(path.stat().st_size for path in self.work_dir.rglob("*") if path.is_file())
        available = shutil.disk_usage(self.work_dir).free + current_bytes
        required = math.ceil(source_bytes * 1.10)
        if available < required:
            raise RuntimeError(f"insufficient disk: available={available} required={required}")

    def _load_report(self) -> dict | None:
        return json.loads(self.report_path.read_text()) if self.report_path.exists() else None


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("command", choices=("prepare", "cutover", "status"))
    value.add_argument("--repo", required=True)
    value.add_argument("--work-dir", default="/workspace/hf-video-migration")
    value.add_argument("--download-workers", type=int, default=32)
    value.add_argument("--encode-workers", type=int, default=8)
    value.add_argument("--ffmpeg-threads", type=int, default=4)
    value.add_argument("--upload-workers", type=int, default=16)
    value.add_argument("--crf", type=int, default=18)
    value.add_argument("--preset", default="fast")
    value.add_argument("--spotcheck-count", type=int, default=5)
    value.add_argument("--minimum-ssim", type=float, default=0.93)
    return value


def main():
    args = parser().parse_args()
    migration = Migration(args)
    getattr(migration, args.command)()


if __name__ == "__main__":
    main()
