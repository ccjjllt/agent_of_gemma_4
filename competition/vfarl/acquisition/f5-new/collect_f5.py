"""Prepare explicitly reviewed F5 source records and real tool runs.

Mirrors s600-common-root/collect.py for the F600 F5 batch (40 Smith-py train
timing tasks). Same six tools, same JUnit file-scope evidence, same protected
paths, same dual-phase (baseline + repaired) official replay. Runner is
replay/.venv/bin/python; per-repo task-site-packages profiles match S.
Reports are written under acquisition/f5-new/reports/ so no S600 file changes.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-new"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
sys.path.insert(0, str(REPLAY))
from prepare_sfr import acquire
from normalize_results import junit_report

REPORTS = HERE / "reports"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path):
    return str(path.relative_to(ROOT))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def contracts():
    return json.loads((HERE / "selected-contracts.json").read_text())["tasks"]


def acquired():
    return {t["instance_id"]: t for t in json.loads((HERE / "acquired-40.json").read_text())["tasks"]}


def preparation_path(config):
    return HERE / (config["key"] + "-preparation.json")


def function_text(path, qualified):
    text = path.read_text()
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
    """Dotted Class[.method] name enclosing a 1-based line number."""
    tree = ast.parse(path.read_text())
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


def smith_inverse_edits(patch, source_root, upstream_root, symbol, key):
    """Repair edits (buggy old -> fixed new) parsed from a bug-introducing patch.

    Mirrors the s600-common-c inverse-edit treatment: '+' lines belong to the
    buggy side, '-' lines to the fixed side. Every hunk must sit inside the
    single declared production symbol; the fully patched files must equal
    their upstream bytes.
    """
    hunks = []
    cur = {"file": None, "old": [], "new": [], "buggy_start": None,
           "base_start": None, "first_plus": None, "first_minus": None}
    in_hunk = False

    def flush():
        if in_hunk and cur["file"] is not None and (cur["old"] or cur["new"]):
            hunks.append((cur["file"], list(cur["old"]), list(cur["new"]),
                          cur["buggy_start"], cur["base_start"],
                          cur["first_plus"], cur["first_minus"]))
        cur.update(old=[], new=[], first_plus=None, first_minus=None)
    import re as _re
    for raw in patch.splitlines():
        if raw.startswith("diff --git "):
            flush()
            in_hunk = False
            m = _re.match(r"diff --git a/\S+ b/(\S+)", raw)
            cur["file"] = m.group(1) if m else None
        elif raw.startswith("@@ "):
            flush()
            in_hunk = True
            m = _re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", raw)
            if not m:
                raise ValueError("Unparseable hunk header: " + raw)
            # Surgery runs on the buggy workspace: the '+' side carries
            # buggy coordinates (the patch direction is base -> buggy).
            cur["base_start"] = int(m.group(1))
            cur["buggy_start"] = int(m.group(3))
        elif not in_hunk or cur["file"] is None:
            continue
        elif raw.startswith("--- ") or raw.startswith("+++ "):
            continue
        elif raw.startswith("\\"):
            continue
        elif raw.startswith(" ") or raw.startswith("-") or raw.startswith("+"):
            # Bug-introducing direction is base -> buggy, so '+' lines (plus
            # context) describe the buggy workspace and '-' lines the fix.
            if raw[:1] in (" ", "+"):
                if raw[:1] == "+" and cur["first_plus"] is None:
                    cur["first_plus"] = len(cur["old"])
                cur["old"].append(raw[1:])
            if raw[:1] in (" ", "-"):
                if raw[:1] == "-" and cur["first_minus"] is None:
                    cur["first_minus"] = len(cur["new"])
                cur["new"].append(raw[1:])
        else:
            raise ValueError("Unexpected patch line: " + raw[:60])
    flush()
    if not hunks:
        raise ValueError("No usable hunks in source patch: " + key)
    files = {h[0] for h in hunks}
    if len(files) != 1:
        raise ValueError("Bug patch touches multiple files %s: %s" % (sorted(files), key))
    # Confinement: the changed lines on BOTH sides of every hunk must sit
    # inside the declared production symbol ('+' lines in buggy coordinates,
    # '-' lines in base/upstream coordinates). This keeps straddling hunks
    # honest while rejecting genuinely multi-function mutations.
    target = next(iter(files))
    resolved = []
    for _, old, new, buggy_start, base_start, first_plus, first_minus in hunks:
        # Changed lines on each side (blank-only fallout, e.g. a deleted
        # separator line between methods, is exempt from confinement but
        # still applied and verified by full-file equality).
        changed_buggy = [l for l in old if l not in new]
        changed_base = [l for l in new if l not in old]
        if changed_buggy + changed_base and all(not l.strip() for l in changed_buggy + changed_base):
            # Blank-only fallout: exempt from confinement, still applied
            # below and verified by full-file equality.
            pass
        else:
            probes = []
            if first_plus is not None:
                probes.append((source_root / target, buggy_start + first_plus))
            if first_minus is not None and upstream_root is not None:
                probes.append((upstream_root / target, base_start + first_minus))
            if not probes:
                raise ValueError("Hunk without changed lines: " + key)
            for probe_path, probe_line in probes:
                found = enclosing_symbol(probe_path, probe_line)
                if found is None or found[0] != symbol:
                    raise ValueError("Hunk at line %s sits outside %s (in %s): %s"
                                     % (probe_line, symbol, found[0] if found else None, key))
        if not old and not new:
            raise ValueError("Empty repair hunk: " + key)
        resolved.append((target, old, new, buggy_start))
    # Positional application on line arrays (reverse order keeps offsets valid).
    by_file = {}
    for target, old, new, start in resolved:
        by_file.setdefault(target, []).append((old, new, start))
    for target, items in by_file.items():
        text = (source_root / target).read_text()
        body = text.split("\n")
        if text.endswith("\n"):
            body = body[:-1]
        for old, new, start in sorted(items, key=lambda t: t[2], reverse=True):
            if old:
                if body[start - 1:start - 1 + len(old)] != old:
                    raise ValueError("Hunk context mismatch at line %s: %s" % (start, key))
                body[start - 1:start - 1 + len(old)] = new
            else:
                body[start:start] = new
        # NOTE: trailing-newline state is decided by the caller against the
        # upstream bytes (a hunk may itself add or drop the final newline).
        yield target, body, items
    # NOTE: per-hunk (old, new) strings for edit_file actions are derived
    # from `items` by the caller with uniqueness expansion.


def prepare(config):
    key = config["key"]
    compile(config["body"], "<reviewed observation " + key + ">", "exec")
    task_doc = json.loads((HERE / (config["instance_id"] + ".task.json")).read_text())
    row = task_doc["row"]
    if row["instance_id"] != config["instance_id"]:
        raise ValueError("Task identity mismatch: " + key)
    if not row.get("problem_statement", "").strip():
        raise ValueError("This batch requires nonempty publisher descriptions")
    if task_doc.get("revision") != REV:
        raise ValueError("Task outside the fixed Smith-py train revision: " + key)
    acq = acquired()[config["instance_id"]]
    commit = acq["execution_commit"]
    if len(commit) != 40:
        raise ValueError("Complete immutable execution commit required: " + key)
    source = ART / (key + "-source")
    buggy = acquire(row["repo"], commit, source)
    upstream = ART / (config["canonical_repo"].replace("/", "__") + "-upstream")
    up = acquire(config["canonical_repo"], config["snapshot"], upstream)
    if up["resolved_revision"] != acq["upstream"]["resolved_revision"]:
        raise ValueError("Upstream snapshot moved under F5 batch: " + key)
    files, missing = [], []
    for path in sorted(upstream.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        rel = path.relative_to(upstream)
        # Trio keeps its suite under src/trio/**/_tests; keep the same
        # complete-verification scope for that layout.
        if not any(part in {"test", "tests", "testing", "_tests"} for part in rel.parts) and str(rel) not in {"conftest.py", "test.py"}:
            continue
        actual, raw = source / rel, path.read_bytes()
        if actual.exists() and actual.read_bytes() != raw:
            raise ValueError("Original verification source differs: " + str(rel))
        files.append({"path": str(rel), "sha256": sha(path), "source_revision": up["resolved_revision"],
                      "present_in_buggy_source": actual.exists()})
        if not actual.exists():
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("Missing binary original asset requires explicit source hydration: " + str(rel)) from exc
            for patch_line in difflib.unified_diff([], text.splitlines(keepends=True),
                    fromfile="/dev/null", tofile="b/" + str(rel)):
                missing.append(patch_line if patch_line.endswith("\n") else
                    patch_line + "\n\\ No newline at end of file\n")
    targets = sorted({node.split("::", 1)[0] for node in row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]})
    if not targets or not all((source / p).exists() or any(f["path"] == p for f in files) for p in targets):
        raise ValueError("Full original required file scope unavailable: " + key)
    adapted = ART / (key + "-task.json")
    save(adapted, {"instance_id": row["instance_id"], "repo": row["repo"], "base_commit": commit,
        "problem_statement": row["problem_statement"], "patch": "", "test_patch": "".join(missing),
        "FAIL_TO_PASS": row["FAIL_TO_PASS"], "PASS_TO_PASS": row["PASS_TO_PASS"],
        "verification_target_files": targets})
    rel = config["relative_path"]
    # The declared production symbol must exist on both sides (read range and
    # repair scoping); the repair edits themselves come from the reviewed
    # bug-introducing patch, mirroring the s600-common-c inverse treatment.
    _, line, end = function_text(source / rel, config["symbol"])
    try:
        function_text(upstream / rel, config["symbol"])
    except ValueError as exc:
        raise ValueError("Production symbol missing upstream: " + key) from exc
    repairs = list(smith_inverse_edits(row["patch"], source, upstream, config["symbol"], key))
    if len(repairs) != 1 or repairs[0][0] != rel:
        raise ValueError("Bug patch outside the reviewed production file: " + key)
    _, new_body, items = repairs[0]
    up_text = (upstream / rel).read_text()
    new_text = "\n".join(new_body) + ("\n" if up_text.endswith("\n") else "")
    if new_text != up_text:
        raise ValueError("Repaired bytes differ from upstream outside reviewed hunks: " + key)
    current = (source / rel).read_text()
    current_lines = current.split("\n")
    edit_pairs = []
    # Descending file order keeps earlier hunk coordinates valid while later
    # (already applied) edits sit below the current block.
    for old_lines, new_lines, start in sorted(items, key=lambda t: t[2], reverse=True):
        # Rebuild against the ORIGINAL hunk body: prepended prefix lines are
        # identical context on both sides, so they join old and new alike.
        old, new = "\n".join(old_lines), "\n".join(new_lines)
        top = start
        while current.count(old) != 1 and top > 1 and (start - top) < 15:
            top -= 1
            old = "\n".join(current_lines[top - 1:start - 1] + old_lines)
            new = "\n".join(current_lines[top - 1:start - 1] + new_lines)
        if current.count(old) != 1:
            raise ValueError("Reviewed edit is not unique in %s: %r" % (key, old[:80]))
        edit_pairs.append((old, new))
        current = current.replace(old, new, 1)
        current_lines = current.split("\n")
    if current != new_text:
        if current.rstrip("\n") != new_text.rstrip("\n"):
            raise ValueError("Actual edit sequence differs from reviewed upstream: " + key)
        # Hunk line arrays omit the terminal-newline marker. Preserve that
        # byte-level difference as a real, uniquely anchored edit action.
        tail_start = max(0, len(current) - 256)
        while current.count(current[tail_start:]) != 1 and tail_start > 0:
            tail_start = max(0, tail_start - 256)
        edit_pairs.append((current[tail_start:], new_text[tail_start:]))
        current = current[:tail_start] + new_text[tail_start:]
    if current != new_text:
        raise ValueError("Serialized edit actions do not restore the reviewed file: " + key)
    module = rel.removeprefix("src/").removesuffix(".py").replace("/", ".")
    command = ("python3 - <<'PYREPLAY'\nfrom pathlib import Path\nimport importlib,hashlib\n"
        + "m=importlib.import_module(" + repr(module) + ")\nactual=Path(m.__file__).resolve();expected=(Path.cwd()/"
        + repr(rel) + ").resolve()\nassert actual==expected,(actual,expected)\n"
        + "print('source_binding',actual,'sha256',hashlib.sha256(actual.read_bytes()).hexdigest())\n"
        + config["body"] + "\nPYREPLAY")
    def run(status):
        return {"tool": "run_command", "arguments": {"command": command}, "expected_status": status,
                "reviewed_command_sha256": hashlib.sha256(command.encode()).hexdigest()}
    def edit_action(pair):
        old, new = pair
        return {"tool": "edit_file", "arguments": {"filepath": rel, "old_string": old, "new_string": new}, "expected_status": "ok"}
    recipe = {"evidence_kind": "human_reviewed_off_policy_demonstration",
        "reviewed_by": "Codex F600 F5 source contract and explicit command review 2026-10-25",
        "review_notes": config["contract"] + " The complete production-file delta is exactly the reviewed hunks inside this function. Original full-file verification and all requirements retained. Actual diagnosis is not a rejected policy action.",
        "actions": [{"tool": "get_status", "arguments": {}, "expected_status": "ok"},
            {"tool": "read_file", "arguments": {"filepath": rel, "start_line": max(1, line-4),
                "end_line": min(end+4, line+120)}, "expected_status": "ok"}, run("error"),
            *[edit_action(pair) for pair in edit_pairs],
            run("ok"), {"tool": "get_status", "arguments": {}, "expected_status": "ok"},
            {"tool": "submit_patch", "arguments": {}, "expected_status": "ok"}],
        "final_response": config["contract"] + " The source-bound reproduction passes after the repair. Patch submitted."}
    recipe_path = REPLAY / ("f5-new-" + key + "-reviewed-recipe.json")
    save(recipe_path, recipe)
    license_path = next((upstream / name for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst")
                         if (upstream / name).exists()), None)
    if license_path is None:
        raise ValueError("Original source license unavailable: " + key)
    copied = HERE / (config["canonical_repo"].replace("/", "__") + ".LICENSE")
    copied.write_bytes(license_path.read_bytes())
    dataset = ROOT / "competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt"
    task_path = HERE / (row["instance_id"] + ".task.json")
    prep = {k: v for k, v in config.items() if k != "body"}
    prep.update({"source_dataset": "SWE-bench/SWE-smith-py", "source_revision": REV, "physical_split": "train",
        "use_partition": "train_only", "course_membership": "F600_F5",
        "original_task": label(task_path), "original_task_sha256": sha(task_path), "execution_commit": commit,
        "buggy_source": buggy, "upstream_source": up, "workspace": label(source),
        "adapted_task": label(adapted), "adapted_task_sha256": sha(adapted), "recipe": label(recipe_path),
        "recipe_sha256": sha(recipe_path), "snapshot_group_id": config["canonical_repo"] + "::" + up["resolved_revision"],
        "repair_family_id": config["canonical_repo"] + "::" + config["symbol"] + "::" + config["family_contract"],
        "verification_target_files": targets, "verification_assets": files,
        "original_FAIL_TO_PASS_count": len(row["FAIL_TO_PASS"]), "original_PASS_TO_PASS_count": len(row["PASS_TO_PASS"]),
        "source_license": {"path": label(copied), "sha256": sha(copied), "revision": up["resolved_revision"]},
        "dataset_license": {"path": label(dataset), "sha256": sha(dataset), "revision": REV, "declared_card_license": "MIT"},
        "mechanism_evidence": {"primary_mechanism": config["submechanism"], "production_symbol": config["symbol"],
            "causal_contract": config["contract"], "source_diff_reviewed": True,
            "bug_patch_direction": "introduces_defect_not_repair_target"},
        "original_patch_semantics": "bug_introducing", "original_verifier_requirements_relaxed": False,
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


def publish(configs, attempt="001", review_name="reviewed-candidates"):
    reviews, reports = [], []
    for config in configs:
        key = config["key"]
        prep_path = preparation_path(config)
        out = ART / (key + "-run-" + attempt)
        if not (out / "run-summary.json").exists():
            continue
        prep = json.loads(prep_path.read_text())
        summary_path = out / "run-summary.json"
        result = json.loads(summary_path.read_text())
        original = json.loads((ROOT / prep["original_task"]).read_text())["row"]
        adapted = json.loads((ROOT / prep["adapted_task"]).read_text())
        phases = {}
        for phase in ("baseline", "repaired"):
            matches = list(out.glob(phase + "-_swegemma_junit_*.xml"))
            if len(matches) != 1:
                phases[phase] = None
                continue
            phases[phase] = junit_report(matches[0], original, adapted)
        events_path = out / "tool-events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        observations = [{"index": e["index"], "actual_response": e["response"]} for e in events if e["tool"] == "run_command"]
        result.update({"original_task": prep["original_task"], "original_task_sha256": prep["original_task_sha256"],
            "adapted_task": prep["adapted_task"], "recipe": prep["recipe"], "replay_output": label(out),
            "reports": phases, "canonical_repo": prep["canonical_repo"], "source_dataset": prep["source_dataset"],
            "source_revision": REV, "physical_split": "train", "primary_adapter": prep["primary_adapter"],
            "submechanism": prep["submechanism"], "course_membership": prep["course_membership"], "execution_commit": prep["execution_commit"],
            "repair_family_id": prep["repair_family_id"], "snapshot_group_id": prep["snapshot_group_id"],
            "mechanism": {"adapter": prep["primary_adapter"], "submechanism": prep["submechanism"], "evidence": prep["mechanism_evidence"]},
            "original_verifier_requirements_relaxed": False, "required_node_identifiers_verbatim_preserved": True,
            "actual_mechanism_observations": observations, "source_preparation": label(prep_path)})
        reviews.append({**prep, "source_review_complete": True, "actual_current_run": label(out),
            "actual_run_summary_sha256": sha(summary_path), "actual_recipe_sha256": sha(ROOT / prep["recipe"]),
            "actual_events_sha256": sha(events_path), "actual_reports": phases, "actual_observations": observations,
            "resolved_by_local_official_verify_task": result["resolved_by_local_official_verify_task"]})
        reports.append((REPORTS / ("f5-new-" + key + "-attempt" + attempt + "-replay-result.json"), result))
    path = HERE / (review_name + ".json")
    if path.exists() and json.loads(path.read_text()).get("tasks") != reviews:
        raise ValueError("Use a new review-name to preserve a frozen earlier source review")
    save(path, {"date": "2026-10-25", "source_revision": REV, "physical_split": "train",
        "scope": "F600 F5 event/stream/lifecycle timing contracts; Smith-py train only; four competition repositories excluded.",
        "independence_basis": "One reviewed production-function contract per selected mutation, with full source delta confirmation; same upstream snapshot grouped.",
        "tasks": reviews})
    for target, result in reports:
        result.update({"source_review_path": label(path), "source_review_sha256": sha(path)})
        save(target, result)
        print(json.dumps({"instance_id": result["instance_id"], "resolved": result["resolved_by_local_official_verify_task"],
            "baseline_F2P_failed": (result["reports"]["baseline"] or {}).get("all_original_F2P_observed_failed")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--stage", choices=("prepare", "replay", "publish", "all"), default="all")
    parser.add_argument("--attempt", default="001")
    parser.add_argument("--review-name", default="reviewed-candidates")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-going", action="store_true",
                        help="Retain failed attempts and continue independent source tasks; never admit missing evidence")
    args = parser.parse_args()
    if not args.attempt.isdecimal() or not args.review_name.replace("-", "").isalnum():
        raise ValueError("Explicit safe attempt and review identifiers required")
    configs = contracts()
    if args.keys:
        configs = [c for c in configs if c["key"] in args.keys]
    failures = []
    from concurrent.futures import ThreadPoolExecutor

    def work(config):
        try:
            if args.stage in {"prepare", "all"}:
                prepare(config)
                print("prepared", config["key"], flush=True)
            if args.stage in {"replay", "all"}:
                replay(config, args.attempt)
                print("replayed", config["key"], flush=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            if not args.keep_going:
                raise
            print("Retained actual failure", config["key"], repr(error), file=sys.stderr, flush=True)
            return config["key"]
        return None

    if args.workers and args.workers > 1 and args.stage == "replay":
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            failures = [r for r in pool.map(work, configs) if r]
    else:
        for config in configs:
            result = work(config)
            if result:
                failures.append(result)
    if args.stage in {"publish", "all"}:
        publish(configs, args.attempt, args.review_name)
    if failures:
        raise SystemExit(1)
