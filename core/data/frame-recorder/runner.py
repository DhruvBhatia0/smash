import json
import os

from .daytona_connector import DaytonaConnector


class DatasetRunner:
    """Thin entrypoint around the Daytona coordinator/fleet launcher."""

    def __init__(
        self,
        daytona: DaytonaConnector,
        desired_max: int,
        worker_count: int,
    ) -> None:
        self.daytona = daytona
        self.desired_max = desired_max
        self.worker_count = worker_count

    def run(self) -> dict:
        return self.daytona.run(
            sample_limit=self.desired_max,
            worker_count=self.worker_count,
        )


def build_runner() -> DatasetRunner:
    """Build the CPU-only Daytona pipeline from environment variables."""
    provider = os.environ.get("SMASH_STORAGE_PROVIDER", "gdrive").lower()
    if provider not in {"gdrive", "google_drive"}:
        raise ValueError("the Daytona frame pipeline currently requires Google Drive storage")
    return DatasetRunner(
        daytona=DaytonaConnector(),
        desired_max=int(os.environ.get("SMASH_SAMPLE_LIMIT", "0")),
        worker_count=int(os.environ.get("SMASH_WORKER_COUNT", "0")),
    )


def main() -> None:
    print(json.dumps(build_runner().run(), indent=2))


if __name__ == "__main__":
    main()
