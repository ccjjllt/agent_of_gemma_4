"""Structural triage for F1-new candidates (read-only, no replay).

Screens (all must pass before prepare):
  S1: every FAIL_TO_PASS/PASS_TO_PASS node lives in a *.py file
      (hermetic pytest.ini never collects .md/.rst; official matcher would reject).
  S2: no xfail/skip/skipif-marked test shares its base name with any required
      node (official matcher merges parametrized siblings by base name, so one
      skipped sibling poisons every required sibling; see bleach 2w3h76l9).
  S3: no required node is itself marked skip/xfail.

Usage: python3 triage.py <instance_id>...  (uses cached pages + upstream planes)
Upstream plane: replay/artifacts/f1-new/upstream/<canon>.<prefix> if present,
else a scratch download under /tmp/f1tri (read-only inspection).
"""
import ast
import hashlib
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f1-new"


def load_rows():
    rows = {}
    for page in sorted((HERE / "pages").glob("train-*.json")):
        if page.name.endswith(".receipt.json"):
            continue
        for e in json.loads(page.read_text()).get("rows", []):
            rows[e["row"]["instance_id"]] = e["row"]
    return rows


def req_lists(row):
    f = row["FAIL_TO_PASS"]
    p = row["PASS_TO_PASS"]
    if isinstance(f, str):
        f = json.loads(f)
    if isinstance(p, str):
        p = json.loads(p)
    return f, p


def upstream_dir(canonical_repo, prefix):
    dest = ART / "upstream" / (canonical_repo.replace("/", "__") + "." + prefix)
    if dest.is_dir():
        return dest
    dest = Path("/tmp/f1tri") / (canonical_repo.replace("/", "__") + "." + prefix)
    if dest.is_dir():
        return dest
    # scratch download (inspection only)
    for repo_cand in (canonical_repo,):
        url = f"https://codeload.github.com/{repo_cand}/tar.gz/{prefix}"
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            break
        except Exception as e:  # noqa: BLE001
            print(f"download failed {url}: {e}")
            return None
    dest.mkdir(parents=True, exist_ok=True)
    tg = dest.with_suffix(".tgz")
    tg.write_bytes(data)
    with tarfile.open(tg) as b:
        top = b.getmembers()[0].name.split("/")[0]
        for m in b.getmembers():
            parts = Path(m.name).parts
            if not parts or parts[0] != top or ".." in parts:
                raise ValueError("unsafe member")
            rel = Path(*parts[1:])
            if not rel.parts:
                continue
            t = dest / rel
            if m.isdir():
                t.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_bytes(b.extractfile(m).read())
    return dest


def _matcher_poison(skip_base: str, req: str) -> bool:
    """Mirror the official JUnit matcher (prefix branches included).

    A skipped/error sibling poisons a required node not only on exact base
    equality but also when it extends the required base with _suffix, [params]
    or digits (plus the 68-char rule).
    """
    req_short = req.split("::")[-1]
    req_base = req_short.split("[", 1)[0]
    for cand in (skip_base,):
        short_node = cand.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        if cand in (req, req_short, req_base):
            return True
        if short_node in (req, req_short, req_base):
            return True
        if short_node.startswith(f"{req_base}_") or short_node.startswith(f"{req_base}["):
            return True
        if len(req_base) >= 68 and short_node.startswith(req_base):
            return True
        if short_node.startswith(req_base) and short_node[len(req_base):].isdigit():
            return True
    return False


def marked_bases(path: Path):
    """Return {base_name: kinds} for tests carrying skip/xfail marks.

    Covers @pytest.mark.(xfail|skip|skipif) on defs and any pytest.param(
    ..., marks=...) inside parametrize decorators (a single marked param
    poisons the whole base family under the official matcher).
    """
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return {}
    out = {}

    def has_mark(decos):
        for d in decos:
            s = ast.dump(d)
            if "xfail" in s or "skip" in s:
                return True
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            kinds = set()
            if has_mark(node.decorator_list):
                kinds.add("def-mark")
            for d in node.decorator_list:
                s = ast.dump(d)
                if "pytest.param" in s and "mark" in s:
                    kinds.add("param-mark")
            if kinds:
                out.setdefault(node.name, set()).update(kinds)
    # module-level importorskip poisons the whole file
    text = path.read_text(errors="replace")
    if "importorskip" in text:
        out.setdefault("<module-importorskip>", set()).add("module")
    return out


def triage(iid, canonical_repo):
    rows = load_rows()
    row = rows.get(iid)
    if row is None:
        return {"instance_id": iid, "verdict": "no-row"}
    f2p, p2p = req_lists(row)
    if not f2p or not p2p or set(f2p) & set(p2p):
        return {"instance_id": iid, "verdict": "reject-requirements"}
    files = [x.split("::")[0] for x in f2p] + [x.split("::")[0] for x in p2p]
    nonpy = sorted({x for x in files if not x.endswith(".py")})
    if nonpy:
        return {"instance_id": iid, "verdict": "reject-S1-nonpy", "files": nonpy}
    prefix = row["repo"].rsplit(".", 1)[-1]
    up = upstream_dir(canonical_repo, prefix)
    if up is None:
        return {"instance_id": iid, "verdict": "no-upstream"}
    req_bases = {x.split("::")[-1].split("[", 1)[0] for x in f2p + p2p}
    f2p_bases = {x.split("::")[-1].split("[", 1)[0] for x in f2p}
    poison, selfmarked = {}, {}
    for f in sorted(set(files)):
        marks = marked_bases(up / f)
        if "<module-importorskip>" in marks:
            poison[f] = ["<module-importorskip>"]
        for b in marks:
            hit = sorted(r for r in (f2p + p2p) if _matcher_poison(b, r))
            if hit:
                poison.setdefault(f, []).append({b: hit[:4]})
        selfhit = sorted(b for b in marks if b in f2p_bases)
        if selfhit:
            selfmarked[f] = selfhit
    if poison or selfmarked:
        return {"instance_id": iid, "verdict": "reject-S2-poison",
                "poison": poison, "f2p_marked": selfmarked}
    return {"instance_id": iid, "verdict": "pass",
            "f2p": len(f2p), "p2p": len(p2p), "files": sorted(set(files))}


if __name__ == "__main__":
    canon = sys.argv[1]
    for iid in sys.argv[2:]:
        print(json.dumps(triage(iid, canon), ensure_ascii=False))
