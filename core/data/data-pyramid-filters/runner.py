from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from models import Character, FilterSource, HfLocation, Map, Rank, SlpMeta


SOURCE_HF = "erickfm/slippi-public-dataset-v3.7"
FOLDERS = [
    "FOX",
    "FALCO",
    "CPTFALCON",
    "MARTH",
    "ZELDA_SHEIK",
    "PEACH",
    "JIGGLYPUFF",
    "ICE_CLIMBERS",
    "SAMUS",
    "YOSHI",
    "PIKACHU",
    "LUIGI",
    "GANONDORF",
    "DOC",
    "ROY",
]
META_COLUMNS = [
    "source_path",
    "target_path",
    "map",
    "map_id",
    "character1",
    "character1_id",
    "character2",
    "character2_id",
    "match_duration_s",
    "char1_winner",
    "rank",
]


def log(event: str, **fields):
    print(json.dumps({"event": event, "time": round(time.time(), 3), **fields}), flush=True)


class BattlefieldFoxRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.source = HfLocation.parse(args.source_hf)
        self.output = HfLocation.parse(args.output_hf)
        self.filter = FilterSource()
        self.upload_lock = threading.Lock()
        self.existing_lock = threading.Lock()
        self.existing = self.existing_outputs() if not args.dry_run and not args.local_source_dir and not args.download_matches else set()

    def run(self) -> dict:
        if not self.args.dry_run:
            HfApi(token=self.token).create_repo(
                self.output.repo,
                repo_type="dataset",
                token=self.token,
                private=self.args.private,
                exist_ok=True,
            )

        log(
            "run_started",
            source=self.source.repo,
            output=self.output.repo,
            outputPrefix=self.output.prefix,
            folders=self.args.folders,
            workers=self.args.workers,
            dryRun=self.args.dry_run,
        )
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = [pool.submit(self.run_folder, folder) for folder in self.args.folders]
            summaries = [future.result() for future in as_completed(futures)]

        summary = {
            "scanned": sum(row["scanned"] for row in summaries),
            "matched": sum(row["matched"] for row in summaries),
            "copied": sum(row["copied"] for row in summaries),
            "folders": sorted(summaries, key=lambda row: row["folder"]),
        }
        if not self.args.dry_run:
            self.upload_json("summary.json", summary)
        log("run_finished", **{key: summary[key] for key in ("scanned", "matched", "copied")})
        return summary

    def run_folder(self, folder: str) -> dict:
        api = HfApi(token=self.token)
        scanned = matched = copied = 0
        rows = []
        started = time.monotonic()

        for source_path in self.slp_paths(api, folder):
            scanned += 1
            meta = self.slp_meta(source_path)
            if scanned % self.args.progress_every == 0:
                log("folder_progress", folder=folder, scanned=scanned, matched=matched, copied=copied)
            if meta.map != Map.BATTLEFIELD or Character.FOX not in {meta.character1, meta.character2}:
                continue

            matched += 1
            target_path = self.output.join(f"slp/{source_path}")
            rows.append(self.meta_row(source_path, target_path, meta))

            if not self.args.dry_run:
                copied += 1 if self.copy_slp(api, source_path, target_path) else 0

            if self.args.max_per_folder and matched >= self.args.max_per_folder:
                break

        if rows and not self.args.dry_run:
            self.upload_csv(f"metadata/{folder}.csv", rows)

        summary = {
            "folder": folder,
            "scanned": scanned,
            "matched": matched,
            "copied": copied,
            "seconds": round(time.monotonic() - started, 3),
        }
        log("folder_done", **summary)
        return summary

    def slp_paths(self, api: HfApi, folder: str):
        prefix = self.source.join(folder)
        for item in api.list_repo_tree(
            self.source.repo,
            repo_type="dataset",
            token=self.token,
            path_in_repo=prefix,
            recursive=True,
        ):
            path = getattr(item, "path", "")
            if path.lower().endswith(".slp"):
                yield path

    def slp_meta(self, source_path: str) -> SlpMeta:
        characters = self.characters_from_path(source_path)
        return SlpMeta(
            map=self.filter._stage_from_path(source_path) or Map.UNKNOWN,
            character1=characters[0] if len(characters) > 0 else Character.UNKNOWN,
            character2=characters[1] if len(characters) > 1 else Character.UNKNOWN,
            match_duration_s=-1,
            char1_winner=None,
            rank=Rank.UNKNOWN,
        )

    def characters_from_path(self, source_path: str) -> list[Character]:
        name = PurePosixPath(source_path).name
        player_text = name.rsplit("(", 1)[0]
        player_text = player_text.split(" ", 1)[1] if " " in player_text else ""
        players = []
        for part in player_text.split("+"):
            character = self.filter._character_from_text(part)
            if character is not None and character not in players:
                players.append(character)
        return players[:2]

    def copy_slp(self, api: HfApi, source_path: str, target_path: str):
        with self.existing_lock:
            if target_path in self.existing:
                return False

        with tempfile.TemporaryDirectory(prefix="battlefield-fox-") as temp_dir:
            local_path = self.retry(
                "download",
                source_path,
                lambda: hf_hub_download(
                    repo_id=self.source.repo,
                    repo_type="dataset",
                    token=self.token,
                    filename=source_path,
                    local_dir=temp_dir,
                ),
            )
            with self.upload_lock:
                self.retry(
                    "upload",
                    target_path,
                    lambda: api.upload_file(
                        repo_id=self.output.repo,
                        repo_type="dataset",
                        token=self.token,
                        path_or_fileobj=local_path,
                        path_in_repo=target_path,
                    ),
                )
                with self.existing_lock:
                    self.existing.add(target_path)
        return True

    def existing_outputs(self) -> set[str]:
        prefix = self.output.join("slp")
        api = HfApi(token=self.token)

        def list_paths():
            return {
                getattr(item, "path", "")
                for item in api.list_repo_tree(
                    self.output.repo,
                    repo_type="dataset",
                    token=self.token,
                    path_in_repo=prefix,
                    recursive=True,
                )
                if getattr(item, "path", "").lower().endswith(".slp")
            }

        try:
            return self.retry("list_existing", prefix, list_paths)
        except Exception as error:
            log("list_existing_failed", prefix=prefix, error=str(error)[:300])
            return set()

    def retry(self, action: str, path: str, fn):
        for attempt in range(1, self.args.retries + 1):
            try:
                return fn()
            except Exception as error:
                if attempt == self.args.retries:
                    raise
                log("retry", action=action, path=path, attempt=attempt, error=str(error)[:300])
                time.sleep(self.retry_delay(error, attempt))

    def retry_delay(self, error: Exception, attempt: int) -> float:
        match = re.search(r"Retry after (\\d+) seconds", str(error))
        if match:
            return int(match.group(1)) + 5
        return min(60, 2**attempt)

    def meta_row(self, source_path: str, target_path: str, meta: SlpMeta) -> dict:
        return {
            "source_path": source_path,
            "target_path": target_path,
            "map": meta.map.name,
            "map_id": int(meta.map),
            "character1": meta.character1.name,
            "character1_id": int(meta.character1),
            "character2": meta.character2.name,
            "character2_id": int(meta.character2),
            "match_duration_s": meta.match_duration_s,
            "char1_winner": "" if meta.char1_winner is None else meta.char1_winner,
            "rank": meta.rank.value,
        }

    def upload_csv(self, path: str, rows: list[dict]):
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames=META_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        self.upload_bytes(path, handle.getvalue().encode())

    def upload_json(self, path: str, payload: dict):
        self.upload_bytes(path, json.dumps(payload, indent=2).encode())

    def upload_bytes(self, path: str, payload: bytes):
        with self.upload_lock:
            self.retry(
                "upload",
                self.output.join(path),
                lambda: HfApi(token=self.token).upload_file(
                    repo_id=self.output.repo,
                    repo_type="dataset",
                    token=self.token,
                    path_or_fileobj=payload,
                    path_in_repo=self.output.join(path),
                ),
            )


class LocalBattlefieldFoxRunner(BattlefieldFoxRunner):
    def run(self) -> dict:
        source_dir = Path(self.args.local_source_dir)
        stage_dir = Path(self.args.stage_dir)
        if self.args.download_first:
            self.download_source(source_dir)

        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        (stage_dir / "slp").mkdir(parents=True)
        (stage_dir / "metadata").mkdir(parents=True)

        summaries = [self.stage_folder(folder, source_dir, stage_dir) for folder in self.args.folders]
        summary = {
            "scanned": sum(row["scanned"] for row in summaries),
            "matched": sum(row["matched"] for row in summaries),
            "copied": sum(row["copied"] for row in summaries),
            "folders": summaries,
        }
        (stage_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        if not self.args.dry_run:
            HfApi(token=self.token).create_repo(
                self.output.repo,
                repo_type="dataset",
                token=self.token,
                private=self.args.private,
                exist_ok=True,
            )
            for folder in self.args.folders:
                self.upload_staged_folder(stage_dir / "slp" / self.source.join(folder), f"slp/{self.source.join(folder)}")
            self.upload_staged_folder(stage_dir / "metadata", "metadata")
            self.upload_file(stage_dir / "summary.json", "summary.json")

        log("run_finished", **{key: summary[key] for key in ("scanned", "matched", "copied")})
        return summary

    def download_source(self, source_dir: Path):
        patterns = [f"{self.source.join(folder)}/**" for folder in self.args.folders]
        log("source_download_started", source=self.source.repo, localDir=str(source_dir), patterns=patterns)
        snapshot_download(
            repo_id=self.source.repo,
            repo_type="dataset",
            token=self.token,
            allow_patterns=patterns,
            local_dir=str(source_dir),
            max_workers=self.args.workers,
        )
        log("source_download_done", localDir=str(source_dir))

    def stage_folder(self, folder: str, source_dir: Path, stage_dir: Path) -> dict:
        scanned = matched = copied = 0
        rows = []
        folder_dir = source_dir / self.source.join(folder)
        started = time.monotonic()
        for local_path in folder_dir.rglob("*.slp"):
            scanned += 1
            source_path = local_path.relative_to(source_dir).as_posix()
            meta = self.slp_meta(source_path)
            if scanned % self.args.progress_every == 0:
                log("folder_progress", folder=folder, scanned=scanned, matched=matched, copied=copied)
            if meta.map != Map.BATTLEFIELD or Character.FOX not in {meta.character1, meta.character2}:
                continue

            matched += 1
            staged_path = stage_dir / "slp" / source_path
            target_path = self.output.join(f"slp/{source_path}")
            rows.append(self.meta_row(source_path, target_path, meta))
            if not staged_path.exists():
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                self.link_or_copy(local_path, staged_path)
                copied += 1
            if self.args.max_per_folder and matched >= self.args.max_per_folder:
                break

        self.write_csv(stage_dir / "metadata" / f"{folder}.csv", rows)
        summary = {
            "folder": folder,
            "scanned": scanned,
            "matched": matched,
            "copied": copied,
            "seconds": round(time.monotonic() - started, 3),
        }
        log("folder_done", **summary)
        return summary

    def link_or_copy(self, source: Path, target: Path):
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def write_csv(self, path: Path, rows: list[dict]):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=META_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def upload_staged_folder(self, local_path: Path, path: str):
        if not local_path.exists():
            return
        self.retry(
            "upload_folder",
            self.output.join(path),
            lambda: HfApi(token=self.token).upload_folder(
                repo_id=self.output.repo,
                repo_type="dataset",
                token=self.token,
                folder_path=str(local_path),
                path_in_repo=self.output.join(path),
            ),
        )

    def upload_file(self, local_path: Path, path: str):
        self.retry(
            "upload",
            self.output.join(path),
            lambda: HfApi(token=self.token).upload_file(
                repo_id=self.output.repo,
                repo_type="dataset",
                token=self.token,
                path_or_fileobj=str(local_path),
                path_in_repo=self.output.join(path),
            ),
        )


class MatchedBattlefieldFoxRunner(LocalBattlefieldFoxRunner):
    def run(self) -> dict:
        if not self.args.local_source_dir:
            self.args.local_source_dir = str(Path(self.args.stage_dir).with_name("slippi-matched-source"))
        self.args.download_first = True
        return super().run()

    def download_source(self, source_dir: Path):
        patterns = [f"{self.source.join(folder)}/*Fox*(BF).slp" for folder in self.args.folders]
        log("matched_source_download_started", source=self.source.repo, localDir=str(source_dir), patterns=patterns)
        self.retry(
            "snapshot_download",
            str(source_dir),
            lambda: snapshot_download(
                repo_id=self.source.repo,
                repo_type="dataset",
                token=self.token,
                allow_patterns=patterns,
                local_dir=str(source_dir),
                max_workers=self.args.workers,
            ),
        )
        log("matched_source_download_done", localDir=str(source_dir))


def default_output_hf() -> str:
    if os.environ.get("SMASH_FILTER_HF_OUTPUT"):
        return os.environ["SMASH_FILTER_HF_OUTPUT"]
    if os.environ.get("SMASH_HF_REPO"):
        root = os.environ.get("SMASH_HF_ROOT", "battlefield-fox")
        return f"{os.environ['SMASH_HF_REPO']}/{root}"
    return ""


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--source-hf", default=SOURCE_HF)
    arg_parser.add_argument("--output-hf", default=default_output_hf())
    arg_parser.add_argument("--folders", nargs="*", default=FOLDERS)
    arg_parser.add_argument("--workers", type=int, default=15)
    arg_parser.add_argument("--max-per-folder", type=int, default=0)
    arg_parser.add_argument("--progress-every", type=int, default=500)
    arg_parser.add_argument("--retries", type=int, default=5)
    arg_parser.add_argument("--download-matches", action=argparse.BooleanOptionalAction, default=False)
    arg_parser.add_argument("--download-first", action=argparse.BooleanOptionalAction, default=False)
    arg_parser.add_argument("--local-source-dir", default="")
    arg_parser.add_argument("--stage-dir", default="/workspace/battlefield-fox-stage")
    arg_parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    arg_parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    return arg_parser


def main() -> int:
    args = parser().parse_args()
    if not args.output_hf:
        raise SystemExit("--output-hf or SMASH_FILTER_HF_OUTPUT is required")
    if args.download_first and not args.local_source_dir:
        raise SystemExit("--local-source-dir is required with --download-first")
    if args.download_matches:
        runner = MatchedBattlefieldFoxRunner(args)
    elif args.local_source_dir:
        runner = LocalBattlefieldFoxRunner(args)
    else:
        runner = BattlefieldFoxRunner(args)
    print(json.dumps(runner.run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
