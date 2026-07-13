import json
import os

from .daytona_connector import DaytonaConnector


def main() -> None:
    provider = os.environ.get("SMASH_STORAGE_PROVIDER", "gdrive").lower()
    if provider not in {"gdrive", "google_drive"}:
        raise ValueError("Daytona frame capture requires Google Drive storage")
    result = DaytonaConnector().run(
        sample_limit=int(os.environ.get("SMASH_SAMPLE_LIMIT", "0")),
        worker_count=int(os.environ.get("SMASH_WORKER_COUNT", "0")),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
