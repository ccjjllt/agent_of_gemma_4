"""Cheap pre-verification before official replay (scratch only, no replay dirs).

For each (iid, profile, canonical_repo):
  upstream: run target files with profile python, require exit 0 and no
    skipped/error node sharing a base name with any required node.
  buggy: resolve mirror ref, download scratch tgz, run F2P nodes, require
    every F2P node fails/errors (none passes, none skipped-missing).

Prints PASS/FAIL per candidate. Does not write acquisition outputs.
"""
import hashlib
import json
import subprocess
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPLAY = ROOT / "competition/vfarl/replay"


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


def base(n):
    return n.split("::")[-1].split("[", 1)[0]


def matcher_poison(skip_name, skip_class, req):
    """Full official-matcher poison semantics for one skipped/error node."""
    req_short = req.split("::")[-1]
    req_base = req_short.split("[", 1)[0]
    cands = {skip_name}
    if "[" in skip_name:
        cands.add(skip_name.split("[", 1)[0])
    if skip_class:
        cands.add(f"{skip_class}::{skip_name}")
        cands.add(f"{skip_class}.{skip_name}")
        cands.add(f"{skip_class.replace('.', '/')}.py::{skip_name}")
    for node_name in cands:
        if node_name in (req, req_short, req_base):
            return True
        short_node = node_name.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
        if short_node in (req, req_short, req_base):
            return True
        if short_node.startswith(f"{req_base}_") or short_node.startswith(f"{req_base}["):
            return True
        if len(req_base) >= 68 and short_node.startswith(req_base):
            return True
        if short_node.startswith(req_base) and short_node[len(req_base):].isdigit():
            return True
    return False


def run_py(py, args, cwd, extra_env=None):
    import os
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra_env or {})
    return subprocess.run([str(py)] + args, cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=600)


def upstream_tree(canonical_repo, prefix):
    dest = REPLAY / "artifacts/f1-new/upstream" / (canonical_repo.replace("/", "__") + "." + prefix)
    if dest.is_dir():
        return dest
    dest = Path("/tmp/f1tri") / (canonical_repo.replace("/", "__") + "." + prefix)
    return dest if dest.is_dir() else None


def buggy_tree(repo, iid, targets, upstream):
    dest = Path("/tmp/f1bug") / (iid + "-pre")
    if dest.is_dir():
        return dest
    out = subprocess.check_output(
        ["git", "ls-remote", f"https://github.com/{repo}.git", f"refs/heads/{iid}"],
        text=True)
    commit = out.strip().split()[0]
    tg = Path("/tmp/f1bug") / (iid + ".tgz")
    if not tg.exists():
        with urllib.request.urlopen(
                f"https://codeload.github.com/{repo}/tar.gz/{commit}", timeout=180) as r:
            tg.write_bytes(r.read())
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tg) as b:
        top = b.getmembers()[0].name.split("/")[0]
        for m in b.getmembers():
            parts = Path(m.name).parts
            if not parts or parts[0] != top or ".." in parts:
                raise ValueError("unsafe")
            rel = Path(*parts[1:])
            if not rel.parts:
                continue
            t = dest / rel
            if m.isdir():
                t.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_bytes(b.extractfile(m).read())
    # Mirror the acquisition rule: missing original verification files are
    # restored from the pinned upstream snapshot before any pytest runs.
    for t in targets:
        src = upstream / t
        dst = dest / t
        if not dst.exists() and src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    # Scratch fidelity for extension-backed suites: reuse the already-built
    # pristine .so for F2P probing (the real prepare rebuilds from source).
    for so in upstream.rglob("*.so"):
        rel = so.relative_to(upstream)
        dst = dest / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(so.read_bytes())
    return dest


def junit_cases(xml_path):
    root = ET.parse(xml_path).getroot()
    out = []
    for tc in root.findall(".//testcase"):
        outcome = next((k for k in ("failure", "error", "skipped")
                        if tc.find(k) is not None), "passed")
        out.append((tc.get("classname", ""), tc.get("name", ""), outcome))
    return out


def precheck(iid, profile, canonical_repo, source_layout=".", extra_env=None, python=None):
    rows = load_rows()
    row = rows[iid]
    f2p, p2p = req_lists(row)
    req = f2p + p2p
    req_bases = {base(n) for n in req}
    f2p_bases = {base(n) for n in f2p}
    prefix = row["repo"].rsplit(".", 1)[-1]
    py = Path(python) if python else REPLAY / profile / "bin/python"
    if python:
        # Task interpreter carries no packages; the pinned profile supplies
        # test-only dependencies through the module search path.
        ppenv = {"PYTHONPATH": str(REPLAY / profile / "lib/python3.9/site-packages")}
        ppenv.update(extra_env or {})
    up = upstream_tree(canonical_repo, prefix)
    if up is None:
        return {"iid": iid, "verdict": "no-upstream"}
    targets = sorted({n.split("::")[0] for n in req})
    if not python:
        ppenv = {"PYTHONPATH": str(up / source_layout)} if source_layout != "." else {}
        ppenv.update(extra_env or {})
    xml = Path("/tmp/f1bug") / (iid + "-up.xml")
    cmd = ["-m", "pytest"] + targets + ["-q", "-p", "no:cacheprovider",
                                        "-o", "addopts=",
                                        "-W", "ignore::DeprecationWarning",
                                        "-W", "ignore::UserWarning",
                                        f"--junitxml={xml}"]
    r = run_py(py, cmd, up, ppenv)
    if r.returncode != 0:
        return {"iid": iid, "verdict": "FAIL-upstream-exit",
                "code": r.returncode, "tail": (r.stdout + r.stderr)[-1500:]}
    poison = sorted({f"{c}::{n}" for c, n, o in junit_cases(xml)
                     if o in ("skipped", "error", "failure")
                     and any(matcher_poison(n, c, r) for r in req)})
    if poison:
        return {"iid": iid, "verdict": "FAIL-upstream-poison", "nodes": poison[:10]}
    # buggy side: every F2P must fail or error
    targets = sorted({n.split("::")[0] for n in req})
    bug = buggy_tree(row["repo"], iid, targets, up)
    import os as _os
    if python:
        _pp = [str(REPLAY / profile / "lib/python3.9/site-packages")]
        if source_layout != ".":
            _pp.append(str(bug / source_layout))
        ppenv_bug = {"PYTHONPATH": _os.pathsep.join(_pp)}
        ppenv_bug.update(extra_env or {})
    else:
        ppenv_bug = {"PYTHONPATH": str(bug / source_layout)} if True else ppenv
    xmlb = Path("/tmp/f1bug") / (iid + "-bug.xml")
    rb = run_py(py, ["-m", "pytest"] + f2p + ["-q", "-p", "no:cacheprovider",
                                             "-o", "addopts=",
                                             "-W", "ignore::DeprecationWarning",
                                             "-W", "ignore::UserWarning",
                                             f"--junitxml={xmlb}"], bug, ppenv_bug)
    try:
        cases = junit_cases(xmlb)
    except Exception as e:  # noqa: BLE001
        return {"iid": iid, "verdict": "FAIL-buggy-noxml",
                "tail": (rb.stdout + rb.stderr)[-1500:]}
    by_base = {}
    for c, n, o in cases:
        by_base.setdefault(base(n), set()).add(o)
    bad = []
    for n in f2p:
        b = base(n)
        outcomes = by_base.get(b, set())
        if "failure" not in outcomes and "error" not in outcomes:
            bad.append((n, sorted(outcomes) or ["missing"]))
    if bad:
        return {"iid": iid, "verdict": "FAIL-buggy-f2p-pass-or-missing",
                "nodes": bad[:8]}
    return {"iid": iid, "verdict": "PASS"}


if __name__ == "__main__":
    spec = sys.argv[1]
    # spec file: json list of {iid, profile, canonical_repo, source_layout?}
    for item in json.loads(Path(spec).read_text()):
        try:
            print(json.dumps(precheck(
                item["iid"], item["profile"], item["canonical_repo"],
                item.get("source_layout", "."), item.get("env"),
                item.get("python")), ensure_ascii=False), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"iid": item["iid"], "verdict": f"ERROR: {type(e).__name__}: {e}"}), flush=True)
