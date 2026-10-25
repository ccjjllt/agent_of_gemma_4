"""F5-fresh extension: fetch uncached Smith-py train pages (same pinned revision).

Same source/revision/policy as scan_fresh.py. New files only under
acquisition/f5-new/f5-fresh/pages-fresh/. Old f1/f2/f5 pages untouched.
"""
import glob
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path("/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4")
HERE = ROOT / "competition/vfarl/acquisition/f5-new/f5-fresh"
PAGES = HERE / "pages-fresh"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
BASE = ("https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-smith-py"
        "&config=default&split=train&revision=" + REV)


def cached_offsets():
    offs = set()
    for pat in ["acquisition/f1-new/pages/train-*.json",
                "acquisition/f2-new/pages/train-*.json",
                "acquisition/f5-new/pages/train-*.json"]:
        for p in glob.glob(str(ROOT / "competition/vfarl" / pat)):
            m = re.search(r"train-(\d+)", p)
            if m:
                offs.add(int(m.group(1)))
    return offs


def fetch_page(offset, length=100):
    path = PAGES / f"train-{offset}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = BASE + f"&offset={offset}&length={length}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "vfarl-f5fresh"}),
                    timeout=120) as r:
                body = r.read()
            rev = r.headers.get("X-Revision", "")
            blob = json.loads(body)
            PAGES.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps(blob, ensure_ascii=False).encode())
            (PAGES / f"train-{offset}.receipt.json").write_text(json.dumps(
                {"url": url, "offset": offset, "length": length, "split": "train",
                 "revision": REV, "response_X_Revision": rev,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "rows": len(blob.get("rows", []))}, indent=2) + "\n")
            time.sleep(2)
            return blob
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"fetch offset={offset} attempt={attempt} error={e!r} sleep={wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"fetch failed offset={offset}")


if __name__ == "__main__":
    have = cached_offsets()
    for p in PAGES.glob("train-*.json"):
        if "receipt" not in p.name:
            m = re.search(r"train-(\d+)", p.name)
            if m:
                have.add(int(m.group(1)))
    missing = sorted(set(range(0, 50908, 100)) - have)
    print(f"cached={len(have)} missing={len(missing)}", flush=True)
    done = 0
    for off in missing:
        fetch_page(off)
        done += 1
        if done % 10 == 0:
            print(f"fetched {done}/{len(missing)}", flush=True)
    print(f"extension fetch complete: {done} new pages", flush=True)
