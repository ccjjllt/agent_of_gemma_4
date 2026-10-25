"""F5-fresh skip precheck: upstream full-file run, real official matcher.

Only CLEAN tasks are dualed. Uses replay/.venv python + pytest on the
upstream workspace, collects skipped JUnit cases, checks each required
F2P+P2P ID with swegemma 0.2.11 _node_matches_requirement.
"""
import json
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


def load_contracts():
    return [c for c in json.loads((HERE / "selected-contracts.json").read_text())["tasks"]
            if c.get("pool", "smith") == "smith"]


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
        # upstream workspace from acquired-fresh
        acq = {t["instance_id"]: t for t in json.loads((HERE / "acquired-fresh.json").read_text())["tasks"]}[iid]
        # upstream dir is ART/scrapy__scrapy-upstream (shared)
        upstream = ROOT / acq["upstream_dir"]
        assert upstream.exists(), upstream
        with tempfile.TemporaryDirectory() as td:
            junit = Path(td) / "precheck.xml"
            import os
            layout = c.get("source_layout", ".")
            # Never write bytecode/caches into the artifact workspaces.
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            if layout and layout != ".":
                env["PYTHONPATH"] = str(upstream / layout) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            cmd = [str(VENV_PY), "-m", "pytest"] + targets + ["-q", f"--junitxml={junit}", "-p", "no:cacheprovider"]
            print(f"precheck run {key} targets={targets}", flush=True)
            proc = subprocess.run(cmd, cwd=str(upstream), capture_output=True, text=True, timeout=300, env=env)
            print(proc.stdout[-2000:], flush=True)
            print(proc.stderr[-2000:], flush=True)
            if not junit.exists():
                raise RuntimeError(f"precheck JUnit missing for {key}")
            xml = ET.parse(str(junit)).getroot()
            suites = [xml] if xml.tag == "testsuite" else xml.findall(".//testsuite")
            tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
            failures = sum(int(s.attrib.get("failures", 0)) for s in suites)
            errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
            skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
            cases = xml.findall(".//testcase")
            skipped_nodes = []
            failed_nodes = []
            for tc in cases:
                if tc.find("skipped") is not None:
                    cn = tc.get("classname", "")
                    nm = tc.get("name", "")
                    skipped_nodes.append(f"{cn}::{nm}" if cn else nm)
                if tc.find("failure") is not None or tc.find("error") is not None:
                    cn = tc.get("classname", "")
                    nm = tc.get("name", "")
                    failed_nodes.append(f"{cn}::{nm}" if cn else nm)
            # check collisions with real matcher
            collisions = []
            for req in required:
                req_short = req.rsplit("::", 1)[-1]
                req_base = req_short.split("[")[0]
                for node in skipped_nodes:
                    if _node_matches_requirement(node, req, req_short, req_base):
                        collisions.append({"requirement": req, "skipped": node})
            # required failures on the FIXED upstream tree doom the dual too
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
                "precheck_kind": "full_suite_upstream_skip_prefix_official_matcher",
                "scope_files": targets,
                "scope_env": "replay/.venv",
                "upstream_run": f"{tests} tests, {failures} failures, {errors} errors, {skipped} skipped",
                "required_count": len(required),
                "skipped_nodes": skipped_nodes,
                "failed_nodes": failed_nodes[:40],
                "prefix_collisions": collisions,
                "required_failed_on_fixed_tree": req_failed,
                "decision": "CLEAN_dual_permitted" if verdict == "CLEAN" else "BLOCKED_no_dual_per_precheck_first_rule",
                "date": "2026-10-25",
            }
            (HERE / "reports" / f"f5-fresh-{key}-precheck.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"key": key, "verdict": verdict, "tests": tests, "failures": failures, "errors": errors, "skipped": len(skipped_nodes), "collisions": len(collisions), "req_failed": len(req_failed)}), flush=True)
            if verdict != "CLEAN":
                raise SystemExit(f"precheck BLOCKED for {key}, no dual per rule")


if __name__ == "__main__":
    import sys as _sys
    main(_sys.argv[1:] or None)
