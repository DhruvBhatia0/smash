from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def safe_slug(value: str, fallback: str = "replay") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or fallback


def stable_path_token(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class QueueStop:
    reason: str = "producer-finished"


@dataclass(frozen=True)
class FrameRenderJob:
    job_id: str
    slp_path: Path
    relative_slp_path: str
    root_dir: Path
    run_id: str
    start_frame: int | None
    end_frame: int | None

    @classmethod
    def from_slp(
        cls,
        *,
        slp_path: Path,
        root_dir: Path,
        run_id: str,
        start_frame: int | None,
        end_frame: int | None,
    ) -> "FrameRenderJob":
        resolved = slp_path.resolve()
        try:
            relative = str(resolved.relative_to(root_dir.resolve()))
        except ValueError:
            relative = str(resolved)

        base = safe_slug(resolved.stem)
        job_id = f"{base}-{stable_path_token(resolved)}"
        return cls(
            job_id=job_id,
            slp_path=resolved,
            relative_slp_path=relative,
            root_dir=root_dir.resolve(),
            run_id=run_id,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    @property
    def command_id(self) -> str:
        return self.job_id

    @property
    def replay_id(self) -> str:
        return self.slp_path.stem

    @property
    def run_dir(self) -> Path:
        return self.root_dir / "processed-frame-queues" / self.run_id

    @property
    def job_dir(self) -> Path:
        return self.run_dir / "jobs" / self.job_id

    @property
    def logs_dir(self) -> Path:
        return self.job_dir / "logs"

    @property
    def playback_json_path(self) -> Path:
        return self.job_dir / "playback.json"

    @property
    def remote_playback_json_path(self) -> Path:
        return self.job_dir / "remote-playback.json"

    @property
    def state_dir(self) -> Path:
        return self.job_dir / "state"

    @property
    def raw_frame_dir(self) -> Path:
        return self.job_dir / "raw-frames"

    @property
    def aligned_frame_dir(self) -> Path:
        return self.job_dir / "aligned-frames"

    @property
    def image_rows_path(self) -> Path:
        return self.job_dir / "image-rows.jsonl"

    @property
    def result_path(self) -> Path:
        return self.job_dir / "result.json"

    def ensure_dirs(self) -> None:
        for path in (
            self.job_dir,
            self.logs_dir,
            self.state_dir,
            self.raw_frame_dir,
            self.aligned_frame_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def playback_config(self, replay_path: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "replay": replay_path or str(self.slp_path),
            "commandId": self.command_id,
        }
        if self.start_frame is not None:
            payload["startFrame"] = self.start_frame
        if self.end_frame is not None:
            payload["endFrame"] = self.end_frame
        return payload

    def to_json(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "runId": self.run_id,
            "slpPath": str(self.slp_path),
            "relativeSlpPath": self.relative_slp_path,
            "commandId": self.command_id,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "paths": {
                "jobDir": str(self.job_dir),
                "playbackJson": str(self.playback_json_path),
                "stateDir": str(self.state_dir),
                "rawFrameDir": str(self.raw_frame_dir),
                "alignedFrameDir": str(self.aligned_frame_dir),
                "imageRows": str(self.image_rows_path),
                "result": str(self.result_path),
            },
        }

