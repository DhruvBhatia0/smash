from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from random import Random
import json
import os
import re
import urllib.parse
import urllib.request


REPO = "DhruvBhatia0/smash-battlefield-fox"
REVISION = "2c94351b82c2c65a31fb39fe52a34ff905b6abcf"
SEED = 20260712
TARGET = 100
ROOT = Path(os.environ.get("AUDIT_ROOT", "/home/daytona/smash-scenario-audit"))
OUT = ROOT / "replays"


def api_pages(url: str):
    while url:
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request) as response:
            yield from json.load(response)
            link = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>; rel="next"', link)
        url = match.group(1) if match else None


def download(candidate):
    remote_path = candidate["path"]
    url = (
        f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
        + urllib.parse.quote(remote_path, safe="/")
    )
    target = OUT / candidate["target"]
    request = urllib.request.Request(url)
    digest = sha256()
    size = 0
    with urllib.request.urlopen(request) as response, target.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return {
        "file": target.name,
        "sourcePath": remote_path,
        "lfsOid": candidate["lfsOid"],
        "sha256": digest.hexdigest(),
        "bytes": size,
    }


OUT.mkdir(parents=True, exist_ok=True)
tree_url = (
    f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/slp_with_video"
    "?recursive=true&expand=false&limit=1000"
)
entries = list(api_pages(tree_url))
slps = [
    entry
    for entry in entries
    if entry.get("type") == "file" and entry["path"].endswith("/input.slp")
]

# The dataset contains mirrored copies of some matches. LFS object identity is the cheap,
# authoritative pre-download content hash for this pinned revision.
by_oid = {}
for entry in slps:
    oid = entry.get("lfs", {}).get("oid")
    if not oid:
        continue
    by_oid.setdefault(oid, entry)
candidates = list(by_oid.values())
Random(SEED).shuffle(candidates)
selected = []
for index, entry in enumerate(candidates[:TARGET]):
    selected.append(
        {
            "path": entry["path"],
            "lfsOid": entry["lfs"]["oid"],
            "declaredBytes": entry["size"],
            "target": f"{index:03d}.slp",
        }
    )
if len(selected) != TARGET:
    raise RuntimeError(f"Only found {len(selected)} unique SLPs")

rows = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(download, entry) for entry in selected]
    for completed, future in enumerate(as_completed(futures), 1):
        rows.append(future.result())
        if completed % 10 == 0:
            print(f"downloaded {completed}/{TARGET}", flush=True)
rows.sort(key=lambda row: row["file"])

manifest = {
    "repo": REPO,
    "revision": REVISION,
    "seed": SEED,
    "treeEntries": len(entries),
    "slpPaths": len(slps),
    "uniqueLfsObjects": len(candidates),
    "sampleMethod": "deterministic seeded shuffle over unique LFS object ids",
    "replays": rows,
}
(ROOT / "sample-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {
            "replays": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "manifest": str(ROOT / "sample-manifest.json"),
        },
        indent=2,
    )
)
