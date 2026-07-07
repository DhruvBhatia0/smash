from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    def to_json(self) -> dict:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "returncode": self.returncode,
            "stdout": self.stdout[-4000:],
            "stderr": self.stderr[-4000:],
            "elapsedSeconds": round(self.elapsed_seconds, 3),
        }


class CommandRunner:
    def __init__(self, *, cwd: Path):
        self.cwd = cwd

    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        check: bool = True,
        timeout: int | None = None,
    ) -> CommandResult:
        started = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=str(self.cwd),
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        result = CommandResult(
            command=command,
            cwd=str(self.cwd),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=time.monotonic() - started,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit {result.returncode}: {' '.join(command)}\n"
                f"stdout:\n{result.stdout[-2000:]}\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )
        return result

