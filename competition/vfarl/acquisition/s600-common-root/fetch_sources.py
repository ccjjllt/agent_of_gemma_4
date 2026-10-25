"""Cache fixed public train rows after identity-only index validation.

This acquires source material only. A cached row is neither a reviewed repair
family nor an admitted tool trajectory. It never reads another dataset split.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REVISION = "77cab9055d42ab4a5c25c89a8f937096db13558e"
DATASET = "SWE-bench/SWE-smith-py"
INDEX = ROOT / "competition/vfarl/acquisition/general/smith-task-index.jsonl"
EXCLUDED = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def fetch_page(job):
    mirror, offset, length, indexed = job
    name = mirror.split("/", 1)[1] + "-" + str(offset)
    target = HERE / "artifacts/pages" / (name + ".json")
    receipt_path = target.with_suffix(".receipt.json")
    if target.exists() and receipt_path.exists():
        raw = target.read_bytes()
        receipt = json.loads(receipt_path.read_text())
        if receipt["sha256"] != sha(raw) or receipt["revision"] != REVISION:
            raise ValueError("Existing source cache differs from its fixed receipt")
        payload = json.loads(raw)
    else:
        query = urllib.parse.urlencode({"dataset": DATASET, "config": "default",
            "split": "train", "revision": REVISION, "offset": offset, "length": length})
        url = "https://datasets-server.huggingface.co/rows?" + query
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    raw = response.read()
                    actual_revision = response.headers.get("X-Revision")
                if actual_revision != REVISION:
                    raise ValueError("Dataset server did not bind the requested immutable revision")
                payload = json.loads(raw)
                receipt = {"url": url, "dataset": DATASET, "split": "train",
                    "revision": REVISION, "response_revision": actual_revision,
                    "offset": offset, "length": length, "sha256": sha(raw), "bytes": len(raw)}
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        # Validate every identity before persisting or examining task bodies.
        for item in payload["rows"]:
            expected = indexed[item["row_idx"]]
            if (item["row"]["instance_id"], item["row"]["repo"]) != (
                    expected["instance_id"], expected["repo"]):
                raise ValueError("Source body does not join the fixed identity-only train index")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        save(receipt_path, receipt)
    if len(payload["rows"]) != length:
        raise ValueError("Requested exact train range is incomplete")
    for item in payload["rows"]:
        expected = indexed[item["row_idx"]]
        if item["row"]["repo"] != mirror or item["row"]["instance_id"] != expected["instance_id"]:
            raise ValueError("Cached page identity is outside its approved train repository range")
    return {"path": str(target.relative_to(ROOT)), **receipt,
            "task_count": len(payload["rows"]), "mechanism_reviewed": False,
            "official_tool_replayed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirrors", nargs="+", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in INDEX.read_text().splitlines()]
    indexed = {row["row_idx"]: row for row in rows}
    jobs = []
    for mirror in args.mirrors:
        if not mirror.startswith("swesmith/"):
            raise ValueError("Only explicit Smith mirror identities are supported")
        canonical = mirror.split("/", 1)[1].rsplit(".", 1)[0].replace("__", "/").lower()
        if canonical in EXCLUDED:
            raise ValueError("Competition repository excluded")
        subset = [r for r in rows if r["repo"] == mirror]
        if not subset:
            raise ValueError("Repository is absent from the pinned train index")
        offsets = sorted(r["row_idx"] for r in subset)
        if offsets != list(range(offsets[0], offsets[-1]+1)):
            raise ValueError("Noncontiguous source range requires explicit partitioning")
        for offset in range(offsets[0], offsets[-1]+1, 100):
            jobs.append((mirror, offset, min(100, offsets[-1]+1-offset), indexed))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch_page, jobs))
    save(HERE / "source-pages.json", {"dataset": DATASET, "revision": REVISION,
        "physical_split": "train", "identity_index": str(INDEX.relative_to(ROOT)),
        "identity_index_sha256": sha(INDEX.read_bytes()), "pages": results,
        "cached_rows": sum(r["task_count"] for r in results),
        "source_only_not_training_ready": True})
    print(json.dumps({"cached_train_rows": sum(r["task_count"] for r in results),
                      "pages": len(results), "mechanism_reviewed": False}))


if __name__ == "__main__":
    main()
