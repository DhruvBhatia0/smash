"""Materialize frozen-codec shards and stream them through a bounded worker queue."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import zlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from core.training.codec.data import committed_archives, holdout_archive

from .cache import DINO_MEAN, DINO_STD, _load_encoder
from .data import SlpVideoDataset


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return rank, world_size, local_rank


def _archive_key(name: str) -> str:
    return name.removesuffix(".tar.zst")


class _ArchiveWriter:
    def __init__(
        self, root: Path, archive: str, expected: int, shard_size: int, rank: int
    ) -> None:
        self.archive, self.expected, self.shard_size = archive, expected, shard_size
        self.folder = root / ".partial" / f"{_archive_key(archive)}-rank{rank}"
        shutil.rmtree(self.folder, ignore_errors=True)
        self.folder.mkdir(parents=True)
        self.latents: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.samples: list[str] = []
        self.shards: list[dict[str, Any]] = []
        self.count = 0

    def add(self, latent: Tensor, actions: Tensor, sample: str) -> None:
        self.latents.append(latent)
        self.actions.append(actions)
        self.samples.append(sample)
        self.count += 1
        if len(self.latents) == self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.latents:
            return
        name = f"shard-{len(self.shards):04d}.pt"
        temporary = self.folder / f"{name}.partial"
        torch.save(
            {
                "latents": torch.stack(self.latents),
                "actions": torch.stack(self.actions),
                "samples": self.samples,
            },
            temporary,
        )
        temporary.replace(self.folder / name)
        self.shards.append({"file": name, "count": len(self.latents)})
        self.latents, self.actions, self.samples = [], [], []

    def finish(self, root: Path) -> None:
        self.flush()
        if self.count != self.expected:
            raise RuntimeError(
                f"{self.archive}: encoded {self.count:,}, expected {self.expected:,}"
            )
        (self.folder / "complete.json").write_text(
            json.dumps(
                {
                    "archive": self.archive,
                    "count": self.count,
                    "shards": self.shards,
                },
                indent=2,
            )
            + "\n"
        )
        destination = root / "archives" / _archive_key(self.archive)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.folder.replace(destination)


@torch.inference_mode()
def materialize(
    data_dir: Path,
    codec_checkpoint: Path,
    output: Path,
    *,
    batch_size: int = 18,
    workers: int = 6,
    prefetch_factor: int = 4,
    shard_size: int = 256,
) -> None:
    rank, world_size, local_rank = _distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    archives = committed_archives(
        data_dir, frames=40, stride_frames=40, require_index=True
    )
    names = [archive.path.name for archive in archives]
    if len(names) != len(set(names)):
        raise ValueError("Archive filenames must be unique")
    expected = {archive.path.name: archive.clips or 0 for archive in archives}

    if rank == 0:
        shutil.rmtree(output / ".partial", ignore_errors=True)
        (output / ".partial").mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    completed = {
        archive.path.name
        for archive in archives
        if (output / "archives" / _archive_key(archive.path.name) / "complete.json").is_file()
    }
    pending = [archive for archive in archives if archive.path.name not in completed]
    dataset = SlpVideoDataset(
        pending,
        frames=40,
        stride_frames=40,
        size=(208, 252),
        rank=rank,
        world_size=world_size,
    )
    worker_options = (
        {"persistent_workers": True, "prefetch_factor": prefetch_factor}
        if workers
        else {}
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        **worker_options,
    )
    encoder, latent_mean, latent_std = _load_encoder(codec_checkpoint, device)
    mean = torch.tensor(DINO_MEAN, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(DINO_STD, device=device).view(1, 1, 3, 1, 1)
    writers: dict[str, _ArchiveWriter] = {}
    encoded = 0

    for video, actions, samples, archive_names in loader:
        video = video.to(device, non_blocking=True).float().div_(255)
        video = F.pad(
            (video - mean) / std,
            (0, -video.shape[-1] % 32, 0, -video.shape[-2] % 32, 0, 0),
            mode="replicate",
        )
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            latents = (encoder(video) - latent_mean) / latent_std
        latents = latents.to(device="cpu", dtype=torch.bfloat16)
        actions = actions.to(dtype=torch.float16)
        for latent, action, sample, archive in zip(
            latents, actions, samples, archive_names, strict=True
        ):
            writer = writers.get(archive)
            if writer is None:
                writer = writers[archive] = _ArchiveWriter(
                    output, archive, expected[archive], shard_size, rank
                )
            writer.add(latent, action, sample)
            if writer.count == writer.expected:
                writer.finish(output)
                del writers[archive]
                print(
                    json.dumps(
                        {"rank": rank, "archive": archive, "clips": writer.count}
                    ),
                    flush=True,
                )
        encoded += len(video)
        if encoded % (10 * batch_size) == 0:
            print(json.dumps({"rank": rank, "encoded": encoded}), flush=True)

    if writers:
        raise RuntimeError(
            f"Rank {rank} ended with incomplete archives: {sorted(writers)}"
        )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        _write_manifest(archives, codec_checkpoint, output)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _write_manifest(archives, codec_checkpoint: Path, output: Path) -> None:
    _, evaluation = holdout_archive(archives)
    shards: list[dict[str, Any]] = []
    for archive in archives:
        key = _archive_key(archive.path.name)
        complete_path = output / "archives" / key / "complete.json"
        if not complete_path.is_file():
            raise RuntimeError(f"Missing completed cache for {archive.path.name}")
        complete = json.loads(complete_path.read_text())
        if complete["count"] != archive.clips:
            raise RuntimeError(f"Stale completed cache for {archive.path.name}")
        split = "eval" if archive == evaluation else "train"
        shards.extend(
            {
                "path": f"archives/{key}/{item['file']}",
                "count": item["count"],
                "archive": archive.path.name,
                "split": split,
            }
            for item in complete["shards"]
        )
    payload = {
        "schema_version": 1,
        "codec_checkpoint": codec_checkpoint.name,
        "eval_archive": evaluation.path.name,
        "train_samples": sum(item["count"] for item in shards if item["split"] == "train"),
        "eval_samples": sum(item["count"] for item in shards if item["split"] == "eval"),
        "latent_shape": [20, 32, 7, 8],
        "action_shape": [39, 3, 2, 56],
        "shards": shards,
    }
    temporary = output / "manifest.json.partial"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    print(json.dumps({key: value for key, value in payload.items() if key != "shards"}), flush=True)


class LatentShardDataset(IterableDataset[tuple[Tensor, Tensor]]):
    """Cover every shard once per epoch, with only standard DDP-style tail padding."""

    def __init__(
        self,
        root: Path,
        shards: list[dict[str, Any]],
        *,
        rank: int,
        world_size: int,
        workers: int,
        batch_size: int,
        epoch: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.root, self.rank, self.epoch, self.seed = root, rank, epoch, seed
        self.workers = max(workers, 1)
        entries = shards.copy()
        random.Random(seed + epoch).shuffle(entries)
        slots = world_size * self.workers
        self.plans: list[list[dict[str, Any]]] = [[] for _ in range(slots)]
        totals = [0] * slots
        for entry in entries:
            slot = min(range(slots), key=lambda index: (totals[index], index))
            self.plans[slot].append(entry)
            totals[slot] += int(entry["count"])
        alignment = batch_size // math.gcd(batch_size, self.workers)
        self.target = math.ceil(max(totals) / alignment) * alignment
        self.rank_samples = self.workers * self.target
        self.padding_samples = slots * self.target - sum(totals)

    def __len__(self) -> int:
        return self.rank_samples

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        if worker and worker.num_workers != self.workers:
            raise RuntimeError("DataLoader worker count differs from the shard plan")
        plan = self.plans[self.rank * self.workers + worker_id]
        emitted = 0
        cycle = 0
        while emitted < self.target:
            for entry in plan:
                payload = torch.load(
                    self.root / entry["path"], map_location="cpu", mmap=True, weights_only=False
                )
                count = int(entry["count"])
                generator = torch.Generator().manual_seed(
                    self.seed
                    + 1_000_003 * self.epoch
                    + 10_007 * cycle
                    + zlib.crc32(entry["path"].encode())
                )
                for index in torch.randperm(count, generator=generator).tolist():
                    yield payload["latents"][index], payload["actions"][index]
                    emitted += 1
                    if emitted == self.target:
                        return
            cycle += 1


def load_evaluation(
    root: Path, shards: list[dict[str, Any]], samples: int
) -> tuple[Tensor, Tensor]:
    latents, actions, remaining = [], [], samples
    for entry in shards:
        payload = torch.load(
            root / entry["path"], map_location="cpu", mmap=True, weights_only=False
        )
        take = min(remaining, int(entry["count"]))
        latents.append(payload["latents"][:take].clone())
        actions.append(payload["actions"][:take].clone())
        remaining -= take
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Requested {samples} eval clips but only found {samples - remaining}")
    return torch.cat(latents), torch.cat(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("codec_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=256)
    materialize(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
