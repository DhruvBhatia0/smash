#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from frame_queue.pipeline import FrameQueuePipeline


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_NODE = (
    "/Users/dhruv/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
)
DEFAULT_ISO = "/Users/dhruv/Downloads/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"
DEFAULT_RUNPOD_SSH_PRIVATE_KEY = Path.home() / ".ssh" / "id_ed25519"
DEFAULT_RUNPOD_SSH_PUBLIC_KEY = Path.home() / ".ssh" / "id_ed25519.pub"


def int_or_none(value: str) -> int | None:
    if value.lower() in {"none", "null", ""}:
        return None
    return int(value)


def existing_path_default(path: Path) -> str:
    return str(path) if path.exists() else ""


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Process a corpus of .slp files through a bounded producer/consumer frame queue.",
    )
    arg_parser.add_argument(
        "inputs",
        nargs="*",
        default=["replays/downloaded"],
        help="One or more .slp files or directories containing .slp files.",
    )
    arg_parser.add_argument(
        "--runtime",
        choices=["plan", "local-macos", "docker", "runpod"],
        default="plan",
        help="Consumer runtime. RunPod is the first scalable target; plan is safe for tests.",
    )
    arg_parser.add_argument("--run-id", default=None)
    arg_parser.add_argument("--queue-size", type=int, default=1000)
    arg_parser.add_argument("--consumers", type=int, default=1)
    arg_parser.add_argument("--max-jobs", type=int, default=None)
    arg_parser.add_argument("--start-frame", type=int_or_none, default=-123)
    arg_parser.add_argument("--end-frame", type=int_or_none, default=900)
    arg_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    arg_parser.add_argument("--plan-only", action=argparse.BooleanOptionalAction, default=False)
    arg_parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    arg_parser.add_argument("--extract-state", action=argparse.BooleanOptionalAction, default=True)
    arg_parser.add_argument("--align-frames", action=argparse.BooleanOptionalAction, default=True)
    arg_parser.add_argument("--attach-images", action=argparse.BooleanOptionalAction, default=True)
    arg_parser.add_argument("--node-bin", default=DEFAULT_NODE)
    arg_parser.add_argument("--timeout-seconds", type=int, default=90)
    arg_parser.add_argument("--iso", default=DEFAULT_ISO)
    arg_parser.add_argument("--playback-app", default=None)
    arg_parser.add_argument(
        "--video-backend",
        default=os.environ.get("SMASH_VIDEO_BACKEND", "OGL"),
        help="Dolphin video backend passed to the renderer image.",
    )
    arg_parser.add_argument(
        "--dolphin-cpu-core",
        type=int,
        default=int(os.environ.get("SMASH_DOLPHIN_CPU_CORE", "1")),
        help="Dolphin CPU core. Use 0 for interpreter when validating amd64 Docker on Apple Silicon.",
    )
    arg_parser.add_argument(
        "--dolphin-audio-backend",
        default=os.environ.get("SMASH_DOLPHIN_AUDIO_BACKEND", "Null"),
        help="Dolphin DSP audio backend passed to the renderer image.",
    )
    arg_parser.add_argument(
        "--allow-parallel-local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow multiple local macOS Dolphin instances. Off by default because the renderer is fragile.",
    )
    arg_parser.add_argument("--docker-image", default=os.environ.get("SMASH_DOCKER_IMAGE", ""))
    arg_parser.add_argument(
        "--renderer-command",
        default="/opt/slippi-renderer/render-replay.sh",
        help="Command provided by the Docker/RunPod renderer image.",
    )
    arg_parser.add_argument("--runpod-image", default=os.environ.get("SMASH_RUNPOD_IMAGE", ""))
    arg_parser.add_argument(
        "--runpod-cpu-flavor-id",
        action="append",
        default=None,
        help="RunPod CPU flavor family. Repeatable. Defaults to cpu3c.",
    )
    arg_parser.add_argument("--runpod-vcpu-count", type=int, default=1)
    arg_parser.add_argument("--runpod-container-disk-gb", type=int, default=10)
    arg_parser.add_argument("--runpod-volume-gb", type=int_or_none, default=None)
    arg_parser.add_argument("--runpod-cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    arg_parser.add_argument("--runpod-remote-iso", default="/workspace/iso/melee.iso")
    arg_parser.add_argument(
        "--runpod-ssh-private-key",
        default=os.environ.get(
            "RUNPOD_SSH_PRIVATE_KEY",
            existing_path_default(DEFAULT_RUNPOD_SSH_PRIVATE_KEY),
        ),
        help="Private key used for SSH/rsync into RunPod workers.",
    )
    arg_parser.add_argument(
        "--runpod-public-key",
        default=os.environ.get("RUNPOD_PUBLIC_KEY", ""),
        help="Public key text injected into the RunPod worker authorized_keys.",
    )
    arg_parser.add_argument(
        "--runpod-public-key-file",
        default=os.environ.get(
            "RUNPOD_PUBLIC_KEY_FILE",
            existing_path_default(DEFAULT_RUNPOD_SSH_PUBLIC_KEY),
        ),
        help="Public key file injected into the RunPod worker authorized_keys.",
    )
    arg_parser.add_argument("--runpod-wait-timeout-seconds", type=int, default=600)
    arg_parser.add_argument(
        "--upload-processed-to-hf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload each processed job directory to a Hugging Face dataset as it finishes.",
    )
    arg_parser.add_argument(
        "--hf-processed-repo",
        default=os.environ.get("SMASH_HF_PROCESSED_REPO", ""),
        help="Target Hugging Face dataset repo for processed frame output.",
    )
    arg_parser.add_argument(
        "--hf-processed-prefix",
        default=os.environ.get("SMASH_HF_PROCESSED_PREFIX", "processed/frame-queues"),
    )
    arg_parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    arg_parser.add_argument("--hf-private", action=argparse.BooleanOptionalAction, default=True)
    arg_parser.add_argument("--hf-create-repo", action=argparse.BooleanOptionalAction, default=True)
    arg_parser.add_argument("--hf-dry-run", action=argparse.BooleanOptionalAction, default=False)
    arg_parser.add_argument(
        "--hf-include-raw-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include raw PNG frame dumps in processed job uploads.",
    )
    arg_parser.add_argument(
        "--delete-local-after-hf-upload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete bulky per-job output directories after successful HF upload.",
    )
    return arg_parser


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = parser().parse_args(argv)
    if args.runtime == "plan":
        args.plan_only = True
        args.dry_run = True
    if args.plan_only or args.dry_run:
        args.plan_only = True
        args.dry_run = True
        args.hf_dry_run = True
    if args.runtime == "local-macos" and args.consumers > 1 and not args.allow_parallel_local:
        raise SystemExit(
            "local-macos runtime is limited to one consumer unless --allow-parallel-local is set"
        )
    if args.consumers < 1:
        raise SystemExit("--consumers must be >= 1")
    if args.queue_size < 1:
        raise SystemExit("--queue-size must be >= 1")
    if not args.runpod_cpu_flavor_id:
        args.runpod_cpu_flavor_id = ["cpu3c"]
    if not args.runpod_public_key and args.runpod_public_key_file:
        public_key_path = Path(args.runpod_public_key_file).expanduser()
        if public_key_path.exists():
            args.runpod_public_key = public_key_path.read_text().strip()

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    input_paths = [Path(value).expanduser() for value in args.inputs]
    input_paths = [path if path.is_absolute() else ROOT_DIR / path for path in input_paths]

    pipeline = FrameQueuePipeline(
        root_dir=ROOT_DIR,
        run_id=run_id,
        input_paths=input_paths,
        args=args,
    )
    result = pipeline.run().to_json()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
