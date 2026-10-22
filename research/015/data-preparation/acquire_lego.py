#!/usr/bin/env python3
"""Acquire pinned SWE-Lego metadata, then only explicitly assigned train rows.

Metadata uses Parquet column projection over HTTP ranges. Trajectories are
requested from the dataset server by exact ID only after train assignment.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

DOCROOT = Path(__file__).resolve().parent
ROOT = DOCROOT.parent / "data"
DATASET = "Lego-X/SWE-Lego-Real-Data"
REVISION = "879c96b91dfa321b4d1c621d149a9200f8a1d118"
SOURCE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/resolved-00000-of-00001.parquet"


def get_json(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


class RangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.pos + size)
        if stop <= self.pos:
            return b""
        request = urllib.request.Request(self.url + f"?p015_range={self.pos}-{stop - 1}", headers={"Range": f"bytes={self.pos}-{stop - 1}"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206:
                        raise RuntimeError("Server did not honor column-range request; refusing full download")
                    body = response.read()
                    if len(body) != stop - self.pos:
                        raise RuntimeError("Incorrect range length")
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        self.pos = stop
        self.bytes_read += len(body)
        return body


def metadata():
    import pyarrow.parquet as pq
    files = json.loads((ROOT / "source-files.json").read_text())
    source = next(x for x in files if x["path"] == "data/resolved-00000-of-00001.parquet")
    stream = RangeReader(SOURCE, source["size"])
    parquet = pq.ParquetFile(stream)
    columns = ["instance_id", "repo", "base_commit", "resolved", "license_name", "image_name"]
    rows = parquet.read(columns=columns, use_threads=False).to_pylist()
    result = {"source_dataset": DATASET, "revision": REVISION, "columns_read": columns,
              "rows": [dict(row_index=i, **row) for i, row in enumerate(rows)],
              "projected_bytes_read": stream.bytes_read, "source_file_bytes": source["size"],
              "source_sha256": source["lfs"]["oid"], "trajectory_content_read": False}
    (ROOT / "source-metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    print("metadata_rows", len(rows))


def selected():
    policy = json.loads((DOCROOT / "train-identity-freeze.json").read_text())
    if not policy.get("selected"):
        raise RuntimeError("Freeze train selections before requesting trajectory content")
    current = get_json(f"https://huggingface.co/api/datasets/{DATASET}")["sha"]
    if current != REVISION:
        raise RuntimeError("Dataset server main no longer matches pinned metadata revision")
    out = ROOT / "source-rows"
    out.mkdir(exist_ok=True)
    allowed = set(policy["allowed_train_repositories"])
    denied = set(policy["excluded_repositories"])
    for item in policy["selected"]:
        if item["repo"].lower() not in allowed or item["repo"].lower() in denied:
            raise RuntimeError("Selection falls outside frozen train repositories")
        name = item["instance_id"]
        dest = out / (name + ".json")
        if dest.exists():
            print("cached", name, flush=True)
            continue
        where = '"instance_id"=' + "'" + name.replace("'", "''") + "'"
        query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": "resolved", "offset": item["row_index"], "length": 1})
        result = get_json("https://datasets-server.huggingface.co/rows?" + query)
        rows = [row["row"] for row in result.get("rows", [])]
        if len(rows) != 1 or rows[0]["instance_id"] != name or rows[0]["base_commit"] != item["base_commit"]:
            raise RuntimeError("Exact-row lookup disagrees with pinned metadata: " + name)
        result["p015_source"] = {"dataset": DATASET, "revision": REVISION, "split": "resolved", "metadata_row_index": item["row_index"]}
        payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        dest.write_text(payload)
        print("saved", name, len(payload.encode()), hashlib.sha256(payload.encode()).hexdigest(), flush=True)
    if get_json(f"https://huggingface.co/api/datasets/{DATASET}")["sha"] != REVISION:
        raise RuntimeError("Dataset revision changed during selected-row acquisition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["metadata", "selected"])
    args = parser.parse_args()
    (metadata if args.mode == "metadata" else selected)()
