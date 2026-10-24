"""Acquire only fixed-revision public train identity columns for task joins."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import requests
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
DATASET = "SWE-bench/SWE-smith-py"
LOCK = threading.Lock()
RECEIPTS = []


def get(url, start=None, end=None):
    headers = {} if start is None else {"Range": "bytes=%s-%s" % (start, end)}
    if start is not None:
        url += "?vf96_range=%s-%s" % (start, end)
    for retry in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            if start is not None:
                if response.status_code != 206 or len(response.content) != end-start+1:
                    raise RuntimeError("Source failed to honor byte range")
            with LOCK:
                RECEIPTS.append({"url": url, "bytes": len(response.content),
                                 "sha256": hashlib.sha256(response.content).hexdigest(),
                                 "content_range": response.headers.get("Content-Range")})
            return response.content
        except (requests.RequestException, RuntimeError):
            if retry == 2:
                raise
            time.sleep(retry + 1)


class RangeReader(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        return self.pos

    def read(self, size=-1):
        size = min(self.size-self.pos, size if size >= 0 else self.size-self.pos)
        if size <= 0:
            return b""
        result = get(self.url, self.pos, self.pos+size-1)
        self.pos += len(result)
        return result


def load_file(file):
    url = "https://huggingface.co/datasets/%s/resolve/%s/%s" % (DATASET, REV, file["path"])
    with RangeReader(url, file["size"]) as reader:
        rows = pq.ParquetFile(reader).read(columns=["instance_id", "repo"]).to_pylist()
    return file["path"], rows


if __name__ == "__main__":
    tree = json.loads((OUT / "smith-tree.json").read_text())
    files = sorted([f for f in tree if f["path"].endswith(".parquet")], key=lambda f: f["path"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        groups = list(executor.map(load_file, files))
    offset, records = 0, []
    for file, rows in groups:
        for local_index, row in enumerate(rows):
            records.append(dict(row, source=DATASET, revision=REV, split="train",
                                row_idx=offset+local_index, file=file, file_row_idx=local_index))
        offset += len(rows)
    target = OUT / "smith-task-index.jsonl"
    target.write_text("".join(json.dumps(r)+"\n" for r in records))
    manifest = {"source": DATASET, "revision": REV, "split": "train",
                "columns": ["instance_id", "repo"], "rows": len(records),
                "payload_bytes": sum(r["bytes"] for r in RECEIPTS),
                "index_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "requests": RECEIPTS}
    (OUT / "smith-index-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "requests"}), flush=True)
