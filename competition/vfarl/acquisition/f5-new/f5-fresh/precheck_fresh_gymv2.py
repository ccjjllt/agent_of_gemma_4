"""F5-fresh Gym/V2 skip precheck: fixed-tree full-file run, real official matcher.

Builds a temp fixed tree (buggy workspace + reviewed gold production edit +
official test_patch), runs the full required file scope with replay/.venv
pytest, collects the JUnit report, checks every required F2P+P2P ID with
swegemma 0.2.11 _node_matches_requirement. Only CLEAN tasks are dualed.
Also records full-suite failures on the fixed tree (any required failure
blocks eligibility just as a collision does).
"""
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4")
HERE = ROOT / "competition/vfarl/acquisition/f5-new/f5-fresh"
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-fresh"
VENV_PY = REPLAY / ".venv/bin/python"

sys.path.insert(0, str(REPLAY / ".venv/lib/python3.12/site-packages"))
sys.path.insert(0, str(HERE))
from collect_fresh_gymv2 import forward_edits, prod_patch_of


def load_contracts():
    return [c for c in json.loads((HERE / "selected-contracts.json").read_text())["tasks"]
            if c.get("pool") in ("gym", "v2")]


def main(keys=None):
    from swegemma.harness.verification import _node_matches_requirement
    HERE.joinpath("reports").mkdir(parents=True, exist_ok=True)
    contracts = load_contracts()
    if keys:
        contracts = [c for c in contracts if c["key"] in keys]
    for c in contracts:
        key = c["key"]
        iid = c["instance_id"]
        task_doc = json.loads((HERE / (iid + ".task.json")).read_text())
        row = task_doc["row"]
        required = list(row["FAIL_TO_PASS"]) + list(row["PASS_TO_PASS"])
        targets = sorted({n.split("::", 1)[0] for n in required})
        workspace = ART / (key + "-source")
        assert workspace.exists(), workspace
        rel = c["relative_path"]
        edits = forward_edits(prod_patch_of(row, rel, key))
        assert len(edits) == 1, key
        old, new = edits[0]
        with tempfile.TemporaryDirectory() as td:
            fixed = Path(td) / "fixed"
            shutil.copytree(workspace, fixed, symlinks=True,
                            ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.pyc",
                                                          ".pytest_cache", ".git"))
            cur = (fixed / rel).read_text()
            if cur.count(old) != 1:
                raise RuntimeError(f"precheck edit not unique: {key}")
            (fixed / rel).write_text(cur.replace(old, new, 1))
            tp = row.get("test_patch", "")
            if tp.strip():
                tp_file = Path(td) / "test.patch"
                tp_file.write_text(tp)
                ap = subprocess.run(["git", "apply", "--whitespace=nowarn",
                                     str(tp_file)], cwd=str(fixed),
                                    capture_output=True, text=True, timeout=120)
                if ap.returncode != 0:
                    # record but continue: official verifier may apply differently
                    print(f"precheck test_patch git-apply note {key}: {ap.stderr[:300]}", flush=True)
            junit = Path(td) / "precheck.xml"
            layout = c.get("source_layout", ".")
            env = None
            if True:
                import os
                env = dict(os.environ)
                # Never write bytecode/caches into the artifact workspaces.
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                if layout and layout != ".":
                    env["PYTHONPATH"] = str(fixed / layout) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            cmd = [str(VENV_PY), "-m", "pytest"] + targets + ["-q", f"--junitxml={junit}", "-p", "no:cacheprovider"]
            print(f"precheck run {key} targets={targets}", flush=True)
            try:
                proc = subprocess.run(cmd, cwd=str(fixed), capture_output=True, text=True,
                                      timeout=600, env=env)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"precheck timeout for {key}")
            print(proc.stdout[-1500:], flush=True)
            print(proc.stderr[-1500:], flush=True)
            if not junit.exists():
                raise RuntimeError(f"precheck JUnit missing for {key}")
            xml = ET.parse(str(junit)).getroot()
            suites = [xml] if xml.tag == "testsuite" else xml.findall(".//testsuite")
            tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
            failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
            errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
            skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
            cases = xml.findall(".//testcase")
            skipped_nodes, failed_nodes = [], []
            for tc in cases:
                cn, nm = tc.get("classname", ""), tc.get("name", "")
                node = f"{cn}::{nm}" if cn else nm
                if tc.find("skipped") is not None:
                    skipped_nodes.append(node)
                if tc.find("failure") is not None or tc.find("error") is not None:
                    failed_nodes.append(node)
            collisions = []
            for req in required:
                req_short = req.rsplit("::", 1)[-1]
                req_base = req_short.split("[")[0]
                for node in skipped_nodes:
                    if _node_matches_requirement(node, req, req_short, req_base):
                        collisions.append({"requirement": req, "skipped": node})
            # required failures on the FIXED tree (official matcher)
            req_failed = []
            for req in required:
                req_short = req.rsplit("::", 1)[-1]
                req_base = req_short.split("[")[0]
                for node in failed_nodes:
                    if _node_matches_requirement(node, req, req_short, req_base):
                        req_failed.append({"requirement": req, "failed": node})
                        break
            verdict = "CLEAN" if not collisions and not req_failed else "BLOCKED"
            out = {
                "key": key, "instance_id": iid,
                "precheck_kind": "fixed_tree_full_scope_upstream_skip_prefix_official_matcher",
                "scope_files": targets, "scope_env": "replay/.venv",
                "fixed_run": f"{tests} tests, {failures} failures, {errors} errors, {skipped} skipped",
                "required_count": len(required),
                "skipped_nodes": skipped_nodes,
                "failed_nodes": failed_nodes[:40],
                "prefix_collisions": collisions,
                "required_failed_on_fixed_tree": req_failed,
                "decision": "CLEAN_dual_permitted" if verdict == "CLEAN" else "BLOCKED_no_dual_per_precheck_first_rule",
                "date": "2026-10-25",
            }
            (HERE / "reports" / f"f5-fresh-{key}-precheck.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"key": key, "verdict": verdict, "tests": tests,
                              "failures": failures, "errors": errors,
                              "skipped": len(skipped_nodes),
                              "collisions": len(collisions),
                              "req_failed": len(req_failed)}), flush=True)
            if verdict != "CLEAN":
                raise SystemExit(f"precheck BLOCKED for {key}, no dual per rule")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
