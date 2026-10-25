"""Fetch Smith-py train pages for F3-new mining."""
import json, hashlib, urllib.parse, urllib.request, time
from pathlib import Path
ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
HERE = ROOT / 'competition/vfarl/acquisition/f3-new'
REVISION = "77cab9055d42ab4a5c25c89a8f937096db13558e"
DATASET = "SWE-bench/SWE-smith-py"
INDEX_PATH = ROOT / 'competition/vfarl/acquisition/general/smith-task-index.jsonl'

def sha(b): return hashlib.sha256(b).hexdigest()

rows = [json.loads(l) for l in INDEX_PATH.read_text().splitlines()]
indexed = {r['row_idx']: r for r in rows}

# candidate repos most likely to have cache/invalidation
CANDIDATES = [
 "swesmith/seperman__deepdiff.ed252022",
 "swesmith/iterative__dvc.1d6ea681",
 "swesmith/conan-io__conan.86f29e13",
 "swesmith/dask__dask.5f61e423",
 "swesmith/davidhalter__parso.338a5760",
 "swesmith/encode__starlette.db5063c2",
 "swesmith/agronholm__typeguard.b6a7e438",
 "swesmith/cloudpipe__cloudpickle.6220b0ce",
 "swesmith/tornadoweb__tornado.d5ac65c1",
 "swesmith/getmoto__moto.694ce1f4",
 "swesmith/jd__tenacity.0d40e76f",
 "swesmith/pallets__click.fde47b4b",
 "swesmith/andialbrecht__sqlparse.e57923b3",
 "swesmith/tobymao__sqlglot.036601ba",
 "swesmith/pygments__pygments.27649ebb",
]

def fetch_page(mirror, offset, length):
    name = mirror.split("/",1)[1] + "-" + str(offset)
    target = HERE / "artifacts/pages" / (name + ".json")
    receipt_path = target.with_suffix(".receipt.json")
    if target.exists() and receipt_path.exists():
        raw = target.read_bytes()
        receipt = json.loads(receipt_path.read_text())
        if receipt["sha256"] != sha(raw):
            raise ValueError("cache mismatch "+str(target))
        payload = json.loads(raw)
        print(f"CACHED {name} rows={len(payload['rows'])}")
        return payload
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "default",
        "split": "train", "revision": REVISION, "offset": offset, "length": length})
    url = "https://datasets-server.huggingface.co/rows?" + query
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
                actual_revision = resp.headers.get("X-Revision")
            if actual_revision != REVISION:
                raise ValueError("revision mismatch")
            payload = json.loads(raw)
            receipt = {"url": url, "dataset": DATASET, "split": "train",
                "revision": REVISION, "response_revision": actual_revision,
                "offset": offset, "length": length, "sha256": sha(raw), "bytes": len(raw)}
            break
        except Exception as e:
            print(f"fetch {name} attempt {attempt} failed: {e}")
            if attempt==4: raise
            time.sleep(5*(attempt+1))
    for item in payload["rows"]:
        expected = indexed[item["row_idx"]]
        if (item["row"]["instance_id"], item["row"]["repo"]) != (expected["instance_id"], expected["repo"]):
            raise ValueError("identity mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print(f"FETCHED {name} rows={len(payload['rows'])} bytes={len(raw)}")
    return payload

if __name__ == "__main__":
    import sys
    only = sys.argv[1:] or CANDIDATES
    for mirror in only:
        subset = [r for r in rows if r["repo"]==mirror]
        if not subset:
            print("absent",mirror); continue
        offsets = sorted(r["row_idx"] for r in subset)
        print(f"{mirror} total={len(subset)} range={offsets[0]}-{offsets[-1]}")
        # fetch all pages for this mirror (contiguous? check)
        # Smith index is contiguous per repo? assume yes, fetch in 100 chunks
        start, end = offsets[0], offsets[-1]
        for off in range(start, end+1, 100):
            length = min(100, end+1-off)
            fetch_page(mirror, off, length)
