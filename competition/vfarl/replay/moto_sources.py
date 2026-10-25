"""Acquire immutable Moto training sources for the reviewed CPU replay batch.

This downloads source data only. It does not execute downloaded installation
scripts, modify official verification, or count any source outcome as local success.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "competition/vfarl/replay/artifacts/moto-cpu"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquire(record):
    source = ROOT / record["source_task_path"]
    if digest(source) != record["source_task_sha256"]:
        raise ValueError("Task bytes differ from reviewed inventory")
    wrapper = json.loads(source.read_text())
    task = wrapper["row"]
    commit = record["execution_commit"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Execution requires an immutable complete commit")
    repo = task["repo"]
    folder = OUT / record["instance_id"]
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "source.tgz"
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    if not archive.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "vfarl-public-source-research"})
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
        temporary = archive.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(archive)
    target = folder / "source"
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        prefix = members[0].name.split("/")[0]
        if not prefix.endswith(commit):
            raise ValueError("Codeload root does not match the requested full commit")
        if not target.exists():
            target.mkdir()
            for member in members:
                parts = Path(member.name).parts
                if not parts or parts[0] != prefix or ".." in parts:
                    raise ValueError("Unexpected archive member")
                if len(parts) == 1:
                    continue
                member.name = str(Path(*parts[1:]))
                tar.extract(member, target, filter="data")
    result = {
        "instance_id": record["instance_id"], "source_task": str(source),
        "source_task_sha256": digest(source), "source_repo": repo,
        "execution_commit": commit, "source_url": url,
        "archive": str(archive), "archive_sha256": digest(archive),
        "archive_root": prefix, "workspace_source": str(target),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_code_executed_by_acquisition": False,
    }
    (folder / "source-provenance.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((ROOT / "competition/vfarl/pilot96/catalog.json").read_text())
    records = [t for t in catalog["tasks"] if t["canonical_repo"].lower() == "getmoto/moto"]
    # Bounded public acquisition; all selected task bodies are allowed train rows.
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(acquire, records))
    (OUT / "sources.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({"immutable_sources_acquired": len(results), "directory": str(OUT)}))
