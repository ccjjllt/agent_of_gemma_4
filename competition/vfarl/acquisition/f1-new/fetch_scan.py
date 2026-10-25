"""F1-new batch: fetch pinned train pages and heuristic-scan for F1 candidates.

Reads-only HF datasets-server at pinned revision. Saves pages + receipts.
Scans patches for parameter/config propagation signals.
"""
import hashlib, json, re, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
BASE = ("https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-smith-py"
        "&config=default&split=train&revision=" + REV)

TARGETS = {
    # repo_suffix: (start_row, end_row)
    "oauthlib__oauthlib.1fd52536": (0, 939),
    "joke2k__faker.8b401a7d": (6420, 6779),
    "rsalmei__alive-progress.35853799": (10645, 11015),
    "kurtmckee__feedparser.cad965a3": (19126, 19722),
    "mido__mido.a0158ff9": (29508, 29868),
    "davidhalter__parso.338a5760": (30522, 31132),
    "mozilla__bleach.73871d76": (47255, 47737),
    "arrow-py__arrow.1d70d009": (26747, 27079),
    "tkrajina__gpxpy.09fc46b3": (26068, 26746),
    "alecthomas__voluptuous.a7a55f83": (33771, 34018),
    "pygments__pygments.27649ebb": (3228, 4657),
    "r1chardj0n3s__parse.30da9e4f": (40837, 40926),
    "pyasn1__pyasn1.0f07d724": (16887, 17652),
    "facelessuser__soupsieve.a8080d97": (34725, 35009),
    "rubik__radon.54b88e58": (20842, 21046),
    "keleshev__schema.24a30457": (9475, 9562),
    "erikrose__parsimonious.0d3f5f93": (37084, 37178),
    "dbader__schedule.82a43db1": (36920, 36999),
    "pallets__markupsafe.620c06c9": (30484, 30521),
    "jaraco__inflect.c079a96a": (9021, 9153),
    "chardet__chardet.9630f238": (11016, 11230),
    "seatgeek__thefuzz.8a05a3ee": (2852, 2883),
    "cookiecutter__cookiecutter.b4451231": (28576, 28642),
    "john-kurkowski__tldextract.3d1bf184": (36730, 36829),
    "Mimino666__langdetect.a1598f1a": (27341, 27458),
    "vi3k6i5__flashtext.b316c7e9": (40927, 40972),
    "borntyping__python-colorlog.dfa10f59": (34680, 34707),
    "termcolor__termcolor.3a42086f": (36713, 36729),
    "pwaller__pyfiglet.f8c5f35b": (47244, 47254),
    "gruns__icecream.f76fef56": (34708, 34724),
    "kennethreitz__records.5941ab27": (16853, 16886),
    "tobymao__sqlglot.036601ba": (19919, 20708),
    "jawah__charset_normalizer.1fdd6463": (21397, 21583),
    "madzak__python-json-logger.5f85723f": (2884, 2921),
    "cknd__stackprinter.219fcc52": (2608, 2744),
    "tweepy__tweepy.91a41c6e": (21584, 21885),
    "mozillazg__python-pinyin.e42dede5": (33177, 33497),
    "pyutils__line_profiler.a646bf0f": (33498, 33720),
    "django__channels.a144b4b8": (28464, 28575),
    "django__daphne.32ac73e1": (33721, 33770),
    "burnash__gspread.a8be3b96": (36830, 36919),
    "pyca__pyopenssl.04766a49": (29078, 29378),
    "weaveworks__grafanalib.5c3b17ed": (36591, 36712),
    "alanjds__drf-nested-routers.6144169d": (43703, 43737),
    "buriy__python-readability.40256f40": (21055, 21095),
    "django-money__django-money.835c1ab8": (31809, 32064),
    "getnikola__nikola.0f4c230e": (28643, 29077),
}

F1_POS = [
    r"\*\*kwargs", r"\*args", r"dict\(.*\*\*",
    r"\.get\(", r"getattr\(", r"environ", r"config\[",
    r"super\(\)\.__init__", r"options\.", r"params", r"settings",
    r"forward", r"propagat", r"pass_through", r"defaults\.update",
    r"kwargs\.update", r"kwargs\.pop", r"kwargs\[",
]
F1_NEG = [
    r"invert", r"not \w+:$", r"==.*!=", r"True.*False|False.*True",
]


def fetch_page(offset, length=100):
    path = PAGES / f"train-{offset}.json"
    url = BASE + f"&offset={offset}&length={length}"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            rev = r.headers.get("X-Revision", "")
            break
        except Exception as e:
            print(f"fetch retry {attempt} {offset}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    blob = json.loads(body)
    path.write_bytes(json.dumps(blob, ensure_ascii=False).encode())
    (PAGES / f"train-{offset}.receipt.json").write_text(json.dumps(
        {"url": url, "offset": offset, "length": length, "split": "train",
         "revision": REV, "response_X_Revision": rev,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "rows": len(blob.get("rows", []))}, indent=2) + "\n")
    return blob


def score(row):
    patch = row.get("patch", "")
    iid = row["instance_id"]
    if ".pr_" in iid:
        return None  # synthetic func-level mutations only for this batch
    if "func_" not in iid and "lm_rewrite" not in iid and "combine_file" not in iid:
        return None
    f2p = row.get("FAIL_TO_PASS") or []
    p2p = row.get("PASS_TO_PASS") or []
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    if not f2p or not p2p or set(f2p) & set(p2p):
        return None
    pos = sum(1 for pat in F1_POS if re.search(pat, patch))
    neg = sum(1 for pat in F1_NEG if re.search(pat, patch))
    # call-site arg add/drop lines
    call_lines = [l for l in patch.splitlines()
                  if (l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
                  and re.search(r"\w+\(.*[=,].*\)", l)]
    return {"score": pos * 2 - neg + min(len(call_lines), 6),
            "pos": pos, "neg": neg, "call_lines": len(call_lines),
            "f2p": len(f2p), "p2p": len(p2p),
            "empty_stmt": not (row.get("problem_statement") or "").strip()}


def main():
    only = sys.argv[1:] or list(TARGETS)
    PAGES.mkdir(parents=True, exist_ok=True)
    index = {}
    for l in (ROOT / "competition/vfarl/acquisition/general/smith-task-index.jsonl").read_text().splitlines():
        r = json.loads(l)
        index[r["instance_id"]] = r
    import pickle
    used = set(pickle.load(open("/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/used_ids.pkl", "rb")))
    out = []
    for repo_key in only:
        start, end = TARGETS[repo_key]
        off = start - (start % 100)
        while off <= end:
            blob = fetch_page(off)
            for entry in blob.get("rows", []):
                row = entry["row"]
                if not (start <= entry["row_idx"] <= end):
                    continue
                if row["instance_id"] in used:
                    continue
                s = score(row)
                if s is None:
                    continue
                out.append({"instance_id": row["instance_id"], "row_idx": entry["row_idx"],
                            "repo": row["repo"], **s})
            off += 100
    out.sort(key=lambda d: -d["score"])
    (HERE / "scan-hits.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"scanned_hits={len(out)}")
    for d in out[:80]:
        print(d["score"], d["instance_id"], f'F2P={d["f2p"]} P2P={d["p2p"]} calls={d["call_lines"]} empty={d["empty_stmt"]}')


if __name__ == "__main__":
    main()
