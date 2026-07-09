from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi

from runner import FOLDERS, LocalBattlefieldFoxRunner, META_COLUMNS, SOURCE_HF, default_output_hf, log


class CheckpointUploader:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source_dir = Path(args.source_dir)
        self.stage_dir = Path(args.stage_dir)
        self.state_path = self.stage_dir / "uploaded.json"
        self.uploaded = set(self.load_state())
        runner_args = argparse.Namespace(
            source_hf=args.source_hf,
            output_hf=args.output_hf,
            folders=args.folders,
            workers=1,
            max_per_folder=0,
            progress_every=500,
            retries=args.retries,
            download_matches=True,
            download_first=False,
            local_source_dir=str(self.source_dir),
            stage_dir=str(self.stage_dir),
            dry_run=False,
            private=args.private,
        )
        self.runner = LocalBattlefieldFoxRunner(runner_args)
        HfApi(token=self.runner.token).create_repo(
            self.runner.output.repo,
            repo_type="dataset",
            token=self.runner.token,
            private=args.private,
            exist_ok=True,
        )

    def run(self):
        last_upload = 0.0
        while True:
            paths = self.new_paths()
            stage_summary_exists = Path(self.args.final_stage_dir, "summary.json").exists()
            should_upload = paths and (
                len(paths) >= self.args.batch_size or time.monotonic() - last_upload >= self.args.interval
            )
            if should_upload:
                self.upload_batch(paths[: self.args.batch_size])
                last_upload = time.monotonic()
            elif stage_summary_exists and not paths:
                log("checkpoint_uploader_done", uploaded=len(self.uploaded))
                return
            else:
                log("checkpoint_waiting", uploaded=len(self.uploaded), pending=len(paths))
                time.sleep(self.args.poll)

    def new_paths(self) -> list[Path]:
        paths = []
        for path in self.source_dir.rglob("*.slp"):
            if ".cache" not in path.parts:
                rel = path.relative_to(self.source_dir).as_posix()
                if rel not in self.uploaded:
                    paths.append(path)
        return sorted(paths)

    def upload_batch(self, paths: list[Path]):
        batch_name = f"batch_{int(time.time())}_{len(self.uploaded):08d}"
        batch_dir = self.stage_dir / batch_name
        rows = []
        for local_path in paths:
            source_path = local_path.relative_to(self.source_dir).as_posix()
            target_path = self.runner.output.join(f"slp/{source_path}")
            staged_path = batch_dir / "slp" / source_path
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            self.link_or_copy(local_path, staged_path)
            rows.append(self.runner.meta_row(source_path, target_path, self.runner.slp_meta(source_path)))

        metadata_path = batch_dir / "metadata" / "checkpoints" / f"{batch_name}.csv"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=META_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        summary_path = batch_dir / "summary.checkpoint.json"
        summary_path.write_text(
            json.dumps(
                {
                    "checkpoint": batch_name,
                    "batch_count": len(paths),
                    "uploaded_count": len(self.uploaded) + len(paths),
                    "source_dir": str(self.source_dir),
                },
                indent=2,
            )
        )

        log("checkpoint_upload_started", checkpoint=batch_name, count=len(paths))
        self.runner.upload_staged_folder(batch_dir / "slp", "slp")
        self.runner.upload_staged_folder(batch_dir / "metadata", "metadata")
        self.runner.upload_file(summary_path, "summary.checkpoint.json")
        self.uploaded.update(path.relative_to(self.source_dir).as_posix() for path in paths)
        self.save_state()
        shutil.rmtree(batch_dir)
        log("checkpoint_upload_done", checkpoint=batch_name, uploaded=len(self.uploaded))

    def load_state(self) -> list[str]:
        if not self.state_path.exists():
            return []
        return json.loads(self.state_path.read_text())

    def save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(sorted(self.uploaded)))

    def link_or_copy(self, source: Path, target: Path):
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source-hf", default=SOURCE_HF)
    arg_parser.add_argument("--output-hf", default=default_output_hf())
    arg_parser.add_argument("--folders", nargs="*", default=FOLDERS)
    arg_parser.add_argument("--source-dir", default="/workspace/slippi-matched-source")
    arg_parser.add_argument("--stage-dir", default="/workspace/battlefield-fox-checkpoints")
    arg_parser.add_argument("--final-stage-dir", default="/workspace/battlefield-fox-stage")
    arg_parser.add_argument("--batch-size", type=int, default=1000)
    arg_parser.add_argument("--interval", type=int, default=180)
    arg_parser.add_argument("--poll", type=int, default=30)
    arg_parser.add_argument("--retries", type=int, default=80)
    arg_parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    return arg_parser


def main() -> int:
    args = parser().parse_args()
    if not args.output_hf:
        raise SystemExit("--output-hf or SMASH_FILTER_HF_OUTPUT is required")
    CheckpointUploader(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
