#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

from frame_queue.runtimes import RunPodRestClient


DEFAULT_PREFIX = "smash-frame-worker-"


class RunPodCpuPodCleaner:
    def __init__(self, *, client: RunPodRestClient, prefix: str | None, confirm: bool):
        self.client = client
        self.prefix = prefix
        self.confirm = confirm

    def is_cpu_pod(self, pod: dict) -> bool:
        compute_type = pod.get("computeType") or pod.get("compute_type")
        return compute_type == "CPU" or bool(pod.get("cpuFlavorId") or pod.get("cpuFlavorIds"))

    def matches_prefix(self, pod: dict) -> bool:
        if not self.prefix:
            return True
        name = str(pod.get("name") or "")
        return name.startswith(self.prefix)

    def run(self) -> dict:
        pods = self.client.list_pods()
        candidates = [
            pod for pod in pods if self.is_cpu_pod(pod) and self.matches_prefix(pod)
        ]
        deleted = []
        for pod in candidates:
            pod_id = pod.get("id")
            if not pod_id:
                continue
            if self.confirm:
                self.client.delete_pod(str(pod_id))
                deleted.append(str(pod_id))
        return {
            "confirm": self.confirm,
            "prefix": self.prefix,
            "matchedCpuPods": len(candidates),
            "deletedCpuPods": len(deleted),
            "deletedPodIds": deleted,
            "note": "Run with --confirm to delete. Use --all-cpu to ignore the default smash-frame-worker prefix.",
        }


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Delete RunPod CPU Pods for the frame queue experiment.",
    )
    arg_parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Only delete CPU Pods whose names start with this prefix.",
    )
    arg_parser.add_argument(
        "--all-cpu",
        action="store_true",
        help="Delete all CPU Pods in the account, not just frame queue worker pods.",
    )
    arg_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete matched CPU Pods. Without this, the script is a dry run.",
    )
    return arg_parser


def main(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    cleaner = RunPodCpuPodCleaner(
        client=RunPodRestClient(api_key=api_key),
        prefix=None if args.all_cpu else args.prefix,
        confirm=args.confirm,
    )
    print(json.dumps(cleaner.run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
