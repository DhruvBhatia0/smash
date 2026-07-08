import os
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


class HfConnector:
    def __init__(self, token: str | None = None, expected_email: str | None = None):
        """Verify HF access and fail if the token belongs to the wrong account."""
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.expected_email = expected_email or os.environ.get("SMASH_HF_EXPECTED_EMAIL")
        self.retries = int(os.environ.get("SMASH_HF_RETRIES", "3"))
        self.api = HfApi(token=self.token)
        self.identity = self.whoami()

    def whoami(self) -> dict:
        """Return the authenticated HF user, with no token material included."""
        identity = self.api.whoami(token=self.token)
        email = identity.get("email")
        if self.expected_email and email != self.expected_email:
            raise ValueError(f"HF token email mismatch: expected {self.expected_email}, got {email}")
        return {
            "name": identity.get("name"),
            "email": email,
            "fullname": identity.get("fullname"),
        }

    def create_repo(self, repo: str, private: bool = True) -> str:
        """Create the dataset repo if it does not already exist."""
        return str(self.api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True))

    def list_files(self, repo: str, folder: str = "") -> list[str]:
        """List files under one logical HF folder."""
        prefix = folder.strip("/")
        if not prefix:
            return sorted(self.api.list_repo_files(repo, repo_type="dataset", token=self.token))
        return sorted(
            item.path
            for item in self.api.list_repo_tree(
                repo,
                repo_type="dataset",
                token=self.token,
                path_in_repo=prefix,
                recursive=True,
            )
            if getattr(item, "path", "").startswith(f"{prefix}/")
        )

    def upload_file(self, repo: str, local_path: str, hf_path: str) -> str:
        """Upload one local file to one HF path."""
        return str(
            self._retry(
                lambda: self.api.upload_file(
                    repo_id=repo,
                    repo_type="dataset",
                    token=self.token,
                    path_or_fileobj=local_path,
                    path_in_repo=hf_path.strip("/"),
                )
            )
        )

    def upload_folder(self, repo: str, local_dir: str, hf_path: str) -> str:
        """Upload one local directory to one HF folder."""
        return str(
            self._retry(
                lambda: self.api.upload_folder(
                    repo_id=repo,
                    repo_type="dataset",
                    token=self.token,
                    folder_path=local_dir,
                    path_in_repo=hf_path.strip("/"),
                )
            )
        )

    def download_file(self, repo: str, hf_path: str, local_path: str) -> str:
        """Download one HF file to one local path."""
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = self._retry(
            lambda: hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                token=self.token,
                filename=hf_path.strip("/"),
                local_dir=str(target.parent),
            )
        )
        if Path(downloaded).resolve() != target.resolve():
            shutil.copyfile(downloaded, target)
        return str(target)

    def exists(self, repo: str, hf_path: str) -> bool:
        """Return whether one HF path already exists."""
        return hf_path.strip("/") in set(self.list_files(repo))

    def _retry(self, action):
        """Retry short HF races without hiding persistent failures."""
        for attempt in range(self.retries):
            try:
                return action()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2**attempt)
