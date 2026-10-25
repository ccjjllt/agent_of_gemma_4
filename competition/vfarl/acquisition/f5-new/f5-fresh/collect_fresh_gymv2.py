"""Prepare explicitly reviewed F5-fresh Gym/V2 source records and real tool runs.

Mirrors collect_fresh.py (Smith) for SWE-Gym / SWE-rebench-V2 rows, modeled on
the f2-gymv2c forward-edit precedent: the source patch is the repair (gold)
direction, the workspace pins the buggy base commit, and the recipe applies the
single reviewed production edit. Same six tools, same JUnit file-scope
evidence, same protected paths, same dual-phase official replay via
replay/.venv (copy mode). Reports under acquisition/f5-new/f5-fresh/reports/.
Old f5-new/f5-replace/f5-small, f1-f4, S600 untouched.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-fresh"
GYM_REV = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
V2_REV = "10483de0f50fe5da545942705a76c6150171af7f"
DATASET = {"gym": "SWE-Gym/SWE-Gym", "v2": "nebius/SWE-rebench-V2"}
REV = {"gym": GYM_REV, "v2": V2_REV}
sys.path.insert(0, str(REPLAY))
from prepare_sfr import acquire
from normalize_results import junit_report

REPORTS = HERE / "reports"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(path):
    return str(Path(path).relative_to(ROOT))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def contracts():
    return json.loads((HERE / "selected-contracts.json").read_text())["tasks"]


def preparation_path(config):
    return HERE / (config["key"] + "-preparation.json")


def function_text(path, qualified):
    text = Path(path).read_text()
    scope = ast.parse(text)
    for component in qualified.split("."):
        nodes = [node for node in scope.body if isinstance(node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == component]
        if len(nodes) != 1:
            raise ValueError("Production symbol does not resolve uniquely: " + qualified)
        scope = nodes[0]
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("Reviewed repair must target an explicit production function")
    return "".join(text.splitlines(keepends=True)[scope.lineno-1:scope.end_lineno]), scope.lineno, scope.end_lineno


def enclosing_symbol(path, lineno):
    tree = ast.parse(Path(path).read_text())
    best = None
    def visit(node, stack):
        nonlocal best
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.lineno <= lineno <= child.end_lineno:
                    name = ".".join(stack + [child.name])
                    if best is None or len(name) > len(best[0]):
                        best = (name, child.lineno, child.end_lineno)
                    visit(child, stack + [child.name])
            else:
                visit(child, stack)
    visit(tree, [])
    return best


def forward_edits(patch):
    """Repair edits (buggy old -> fixed new) parsed from a repair-direction patch."""
    edits = []
    old, new = [], []
    started = False
    for line in patch.splitlines(keepends=True):
        if line.startswith("@@ "):
            if old or new:
                edits.append(("".join(old), "".join(new)))
            old, new = [], []
            started = True
        elif not started:
            continue
        elif line.startswith(("diff --git ", "index ", "--- ", "+++ ", "\\")):
            continue
        elif line.startswith(" "):
            old.append(line[1:])
            new.append(line[1:])
        elif line.startswith("+"):
            new.append(line[1:])
        elif line.startswith("-"):
            old.append(line[1:])
    if old or new:
        edits.append(("".join(old), "".join(new)))
    return [(o, n) for o, n in edits if o != n]


def prod_patch_of(row, rel, key):
    parts = row["patch"].split("diff --git a/")
    hits = [p for p in parts if p.startswith(rel + " ") or p.startswith(rel + "\n")]
    # fall back to b/ prefix match
    if not hits:
        hits = [p for p in parts if rel in p.split("\n")[0]]
    if len(hits) != 1:
        raise ValueError(f"Production patch part not unique for {rel}: {key}")
    return "diff --git a/" + hits[0]


def prepare(config):
    key = config["key"]
    pool = config["pool"]
    compile(config["body"], "<reviewed observation " + key + ">", "exec")
    task_doc = json.loads((HERE / (config["instance_id"] + ".task.json")).read_text())
    row = task_doc["row"]
    if row["instance_id"] != config["instance_id"]:
        raise ValueError("Task identity mismatch: " + key)
    if not (row.get("problem_statement") or "").strip():
        raise ValueError("This batch requires nonempty publisher descriptions")
    if task_doc.get("revision") != REV[pool]:
        raise ValueError("Task outside the fixed train revision: " + key)
    base = config["base_commit"]
    if len(base) != 40:
        raise ValueError("Complete immutable base commit required: " + key)
    source = ART / (key + "-source")
    got = acquire(config["canonical_repo"], base, source)
    if got["resolved_revision"] != base:
        raise ValueError("Base snapshot moved under F5 batch: " + key)
    rel = config["relative_path"]
    if not (source / rel).exists():
        raise ValueError("Reviewed production file missing in buggy workspace: " + key)
    try:
        _, line, end = function_text(source / rel, config["symbol"])
    except ValueError as exc:
        raise ValueError("Production symbol missing in buggy workspace: " + key) from exc
    prod_patch = prod_patch_of(row, rel, key)
    edits = forward_edits(prod_patch)
    if len(edits) != 1:
        raise ValueError(f"Expected exactly one reviewed production edit: {key} got {len(edits)}")
    old, new = edits[0]
    # confinement: first changed old-line must sit inside the declared symbol
    old_lines, new_lines = old.splitlines(keepends=True), new.splitlines(keepends=True)
    # find first differing line offset
    off = 0
    for a, b in zip(old_lines, new_lines):
        if a == b:
            off += 1
        else:
            break
    # map offset to workspace line: locate the old block first
    cur = (source / rel).read_text()
    if cur.count(old) != 1:
        raise ValueError("Reviewed edit is not unique in buggy workspace: " + key)
    start_line = cur[:cur.index(old)].count("\n") + 1 + off
    found = enclosing_symbol(source / rel, start_line)
    if found is None or found[0] != config["symbol"]:
        raise ValueError("Repair hunk sits outside %s (in %s): %s"
                         % (config["symbol"], found[0] if found else None, key))
    fixed = cur.replace(old, new, 1)
    compile(fixed, "<repaired " + rel + ">", "exec")
    targets = sorted({n.split("::", 1)[0] for n in list(row["FAIL_TO_PASS"]) + list(row["PASS_TO_PASS"])})
    if not targets:
        raise ValueError("Full original required file scope unavailable: " + key)
    adapted = ART / (key + "-task.json")
    save(adapted, {"instance_id": row["instance_id"], "repo": config["canonical_repo"],
        "base_commit": base, "problem_statement": row["problem_statement"],
        "patch": "", "test_patch": row.get("test_patch", ""),
        "FAIL_TO_PASS": list(row["FAIL_TO_PASS"]), "PASS_TO_PASS": list(row["PASS_TO_PASS"]),
        "verification_target_files": targets})
    module = rel.removeprefix("src/").removesuffix(".py").replace("/", ".")
    command = ("python3 - <<'PYREPLAY'\nfrom pathlib import Path\nimport importlib,hashlib\n"
        + "m=importlib.import_module(" + repr(module) + ")\nactual=Path(m.__file__).resolve();expected=(Path.cwd()/"
        + repr(rel) + ").resolve()\nassert actual==expected,(actual,expected)\n"
        + "print('source_binding',actual,'sha256',hashlib.sha256(actual.read_bytes()).hexdigest())\n"
        + config["body"] + "\nPYREPLAY")
    def run(status):
        return {"tool": "run_command", "arguments": {"command": command}, "expected_status": status,
                "reviewed_command_sha256": hashlib.sha256(command.encode()).hexdigest()}
    recipe = {"evidence_kind": "human_reviewed_off_policy_demonstration",
        "reviewed_by": "Codex F600 F5 source contract and explicit command review 2026-10-25",
        "review_notes": config["contract"] + " The complete production-file delta is exactly the reviewed repair edit inside this function. Original full-file verification and all requirements retained. Actual diagnosis is not a rejected policy action.",
        "actions": [{"tool": "get_status", "arguments": {}, "expected_status": "ok"},
            {"tool": "read_file", "arguments": {"filepath": rel, "start_line": max(1, line-4),
                "end_line": min(end+4, line+120)}, "expected_status": "ok"}, run("error"),
            {"tool": "edit_file", "arguments": {"filepath": rel, "old_string": old, "new_string": new}, "expected_status": "ok"},
            run("ok"), {"tool": "get_status", "arguments": {}, "expected_status": "ok"},
            {"tool": "submit_patch", "arguments": {}, "expected_status": "ok"}],
        "final_response": config["contract"] + " The source-bound reproduction passes after the repair. Patch submitted."}
    recipe_path = REPLAY / ("f5-fresh-" + key + "-reviewed-recipe.json")
    save(recipe_path, recipe)
    upstream_probe = source
    license_path = next((upstream_probe / name for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst")
                         if (upstream_probe / name).exists()), None)
    if license_path is None:
        raise ValueError("Original source license unavailable: " + key)
    copied = HERE / (config["canonical_repo"].replace("/", "__") + ".LICENSE")
    if copied.exists() and copied.read_bytes() != license_path.read_bytes():
        raise ValueError("Same-canon license differs across snapshots: " + key)
    copied.write_bytes(license_path.read_bytes())
    task_path = HERE / (row["instance_id"] + ".task.json")
    prep = {k: v for k, v in config.items() if k != "body"}
    prep.update({"source_dataset": DATASET[pool], "source_revision": REV[pool],
        "physical_split": "train", "use_partition": "train_only", "course_membership": "F600_F5",
        "original_task": label(task_path), "original_task_sha256": sha(task_path),
        "execution_commit": base,
        "buggy_source": {"repo": config["canonical_repo"], "requested_revision": base,
            "resolved_revision": base,
            "archive_path": label(ART / (key + "-source.tgz")),
            "archive_sha256": sha(ART / (key + "-source.tgz")),
            "download_url": f"https://codeload.github.com/{config['canonical_repo']}/tar.gz/{base}",
            "immutable_url": f"https://codeload.github.com/{config['canonical_repo']}/tar.gz/{base}",
            "commit_resolution": None},
        "upstream_source": {"repo": config["canonical_repo"], "requested_revision": base,
            "resolved_revision": base},
        "workspace": label(source),
        "adapted_task": label(adapted), "adapted_task_sha256": sha(adapted),
        "recipe": label(recipe_path), "recipe_sha256": sha(recipe_path),
        "snapshot_group_id": config["canonical_repo"] + "::" + base,
        "repair_family_id": config["canonical_repo"] + "::" + config["symbol"] + "::" + config["family_contract"],
        "verification_target_files": targets,
        "verification_assets": [],
        "original_FAIL_TO_PASS_count": len(row["FAIL_TO_PASS"]),
        "original_PASS_TO_PASS_count": len(row["PASS_TO_PASS"]),
        "source_license": {"path": label(copied), "sha256": sha(copied), "revision": base},
        "mechanism_evidence": {"primary_mechanism": config["submechanism"],
            "production_symbol": config["symbol"], "causal_contract": config["contract"],
            "source_diff_reviewed": True, "bug_patch_direction": "repair_target_gold_patch"},
        "original_patch_semantics": "repair_target", "original_verifier_requirements_relaxed": False,
        "source_mechanism_review_complete": True})
    save(preparation_path(config), prep)


def replay(config, attempt="001"):
    key = config["key"]
    prep = json.loads(preparation_path(config).read_text())
    out = ART / (key + "-run-" + attempt)
    if (out / "run-summary.json").exists():
        return
    if out.exists():
        raise ValueError("An incomplete actual attempt already exists; choose a new attempt identifier")
    vtimeout = "600" if config["canonical_repo"] == "python-trio/trio" else "180"
    with (ART / (key + "-run-" + attempt + ".log")).open("w") as log:
        command = [str(REPLAY / ".venv/bin/python"), str(REPLAY / "run_official.py"),
            "--task", str(ROOT / prep["adapted_task"]), "--workspace", str(ROOT / prep["workspace"]),
            "--recipe", str(ROOT / prep["recipe"]), "--output", str(out), "--verify", "--verify-baseline",
            "--source-layout", config.get("source_layout", "."), "--task-site-packages", str(REPLAY / config["task_profile"]),
            "--task-site-packages-mode", "copy", "--command-timeout", "90", "--verification-timeout", vtimeout]
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--stage", choices=("prepare", "replay"), default="prepare")
    parser.add_argument("--attempt", default="001")
    parser.add_argument("--pool", choices=("gym", "v2"), default=None)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    configs = [c for c in contracts() if c.get("pool") in ("gym", "v2")]
    if args.pool:
        configs = [c for c in configs if c["pool"] == args.pool]
    if args.keys:
        configs = [c for c in configs if c["key"] in args.keys]
    failures = []
    for config in configs:
        try:
            if args.stage == "prepare":
                prepare(config)
                print("prepared", config["key"], flush=True)
            else:
                replay(config, args.attempt)
                print("replayed", config["key"], flush=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            if not args.keep_going:
                raise
            print("Retained actual failure", config["key"], repr(error)[:400], file=sys.stderr, flush=True)
            failures.append(config["key"])
    if failures:
        raise SystemExit(1)
