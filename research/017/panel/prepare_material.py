#!/usr/bin/env python3
"""Prepare frozen P017 evaluation sources; never acquire teacher messages.

The fetch mode projects only task/criterion columns from the pinned parquet,
decodes frozen rows, and separates agent-visible statements from host sources.
The snapshots mode fetches each exact base commit without executing repo code.
Environment installation and upstream pre/post execution are later CPU work.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FREEZE = HERE / "confirmation-identity-freeze.json"
COLUMNS = [
    "instance_id", "repo", "base_commit", "license_name", "image_name",
    "problem_statement", "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS",
    "FAIL_TO_FAIL", "PASS_TO_FAIL", "test_cmd", "eval_cmd", "environment",
    "version", "environment_setup_commit", "install_config", "requirements",
    "dataset_name", "created_at",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def selected():
    freeze = json.loads(FREEZE.read_text())
    items = freeze["primary"]
    if len(items) != 24 or len({x["instance_id"] for x in items}) != 24:
        raise ValueError("Expected frozen 24 primary identities")
    metadata = ROOT / freeze["source"]["metadata_path"]
    if digest(metadata) != freeze["source"]["metadata_sha256"]:
        raise ValueError("Source metadata differs from freeze")
    return freeze, items


def fetch():
    import pyarrow as pa
    import pyarrow.parquet as pq

    freeze, items = selected()
    source = freeze["source"]
    # Reuse the existing strict HTTP range transport. Unlike P015 selected(),
    # this never calls the rows endpoint, which would also return messages.
    path = ROOT / "research/015/data-preparation/acquire_lego.py"
    spec = importlib.util.spec_from_file_location("p015_range_transport", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = module.RangeReader(source["source_url"], source["file_bytes"])
    parquet = pq.ParquetFile(stream)
    missing = sorted(set(COLUMNS) - set(parquet.schema_arrow.names))
    if missing or parquet.metadata.num_rows != source["metadata_rows"]:
        raise ValueError("Pinned source schema/count mismatch: " + str(missing))
    by_offset = {item["row_index"]: item for item in items}
    acquired, agent_tasks, offset = [], [], 0
    for group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group).num_rows
        offsets = sorted(x for x in by_offset if offset <= x < offset + count)
        if offsets:
            # Parquet column pages may contain neighboring rows, but only
            # frozen rows are decoded to Python or exported. messages/meta and
            # teacher hints are never part of the column projection.
            table = parquet.read_row_group(group, columns=COLUMNS, use_threads=False)
            rows = table.take(pa.array([x - offset for x in offsets])).to_pylist()
            for absolute, row in zip(offsets, rows):
                item = by_offset[absolute]
                if (row["instance_id"] != item["instance_id"]
                        or row["repo"].lower() != item["repo"]
                        or row["base_commit"] != item["base_commit"]):
                    raise ValueError("Frozen row mismatch: " + item["instance_id"])
                if not row["problem_statement"] or not row["patch"]:
                    raise ValueError("Missing source statement/reference: " + item["instance_id"])
                row["p017_source"] = {
                    "dataset": source["dataset"], "revision": source["revision"],
                    "split": source["split"], "row_index": absolute,
                    "file_sha256": source["file_sha256"],
                    "freeze_sha256": digest(FREEZE), "teacher_messages_acquired": False,
                }
                out = HERE / "host-source" / (item["instance_id"] + ".json")
                save(out, row)
                acquired.append({"instance_id": item["instance_id"],
                                 "path": str(out.relative_to(ROOT)),
                                 "bytes": out.stat().st_size, "sha256": digest(out)})
                agent_tasks.append({"iid": item["instance_id"],
                                    "instance_id": item["instance_id"],
                                    "repo": item["repo"], "base_commit": item["base_commit"],
                                    "problem_statement": row["problem_statement"],
                                    "split": "dev"})
                print("source", item["instance_id"], out.stat().st_size, flush=True)
        offset += count
    if len(acquired) != len(items):
        raise ValueError("Source acquisition incomplete")
    rank = {x["instance_id"]: i for i, x in enumerate(items)}
    acquired.sort(key=lambda x: rank[x["instance_id"]])
    agent_tasks.sort(key=lambda x: rank[x["instance_id"]])
    save(HERE / "agent-tasks.json", agent_tasks)
    save(HERE / "source-acquisition.json", {
        "freeze_sha256": digest(FREEZE), "source": source,
        "columns_read": COLUMNS, "projected_bytes_read": stream.bytes_read,
        "frozen_rows_exported": len(acquired), "teacher_messages_acquired": False,
        "shared_storage_note": "HTTP range reads return shared parquet column pages; only frozen rows are decoded/exported.",
        "range_transport_path": str(path.relative_to(ROOT)),
        "range_transport_sha256": digest(path), "files": acquired,
        "cpu_environment_executed": False,
    })


def snapshots():
    _, items = selected()
    identities = json.loads((HERE / "repository-identities.json").read_text())
    canonical = {r["requested_repo"]: r["full_name"] for r in identities["repositories"]}
    source_files = json.loads((HERE / "source-acquisition.json").read_text())["files"]
    sources = {x["instance_id"]: x for x in source_files}
    records = []
    for item in items:
        iid = item["instance_id"]
        source = ROOT / sources[iid]["path"]
        if digest(source) != sources[iid]["sha256"]:
            raise ValueError("Source row changed: " + iid)
        dest = HERE / "snapshots" / (iid + ".tar.gz")
        repo = HERE / "workspaces" / iid / "repo"
        if not (repo / ".git").exists():
            repo.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              text=True, capture_output=True)
        if head.returncode or head.stdout.strip() != item["base_commit"]:
            url = "https://github.com/" + canonical[item["repo"]] + ".git"
            subprocess.run(["git", "fetch", "--depth=1", "--no-tags", url,
                            item["base_commit"]], cwd=repo, check=True, timeout=300)
            subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"],
                           cwd=repo, check=True, timeout=120)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if actual != item["base_commit"]:
            raise ValueError("Snapshot base mismatch: " + iid)
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
            raise ValueError("Snapshot checkout has local changes: " + iid)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "w:gz") as archive:
            archive.add(repo, arcname="repo")
        records.append({"instance_id": iid, "repo": item["repo"],
                        "canonical_repo": canonical[item["repo"]],
                        "base_commit": actual, "snapshot": str(dest.relative_to(ROOT)),
                        "bytes": dest.stat().st_size, "sha256": digest(dest)})
        save(HERE / "snapshot-acquisition.json", records)
        print("snapshot", iid, dest.stat().st_size, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fetch", "snapshots"])
    args = parser.parse_args()
    {"fetch": fetch, "snapshots": snapshots}[args.mode]()
