"""Build P012 eval tasks-selected.jsonl + per-task test_patch diffs.

Reads the 12 frozen dev tasks (dev12-freeze.json minus invalid pr_1981 plus
substitution pr_14852), extracts the original issue text (<pr_description>)
from the frozen trajectory rows (metadata + issue text only; never training
input), and emits competition-Task-format JSONL with:
  - problem_statement: the raw issue text
  - test_patch: unified diff adding tests/test_p012_*.py (and tests/p012_lib/sgmllib.py
    for feedparser tasks, BSD vendored support file)
  - FAIL_TO_PASS: [the single criterion node]; PASS_TO_PASS: []
  - patch: '' (gold patch never shipped), hints_text: ''

Writes: evaluation/tasks-selected.jsonl, evaluation/test-patches/<iid>.patch,
evaluation/tasks-manifest.json
"""
import difflib
import hashlib
import json
import re
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CRITERIA = HERE / "criteria"

# instance_id -> (test file name, test node, needs_sgmllib)
TASKS = {
    "keleshev__schema.24a30457.pr_331": ("test_p012_schema_331.py", "test_none_uses_const", False),
    "cookiecutter__cookiecutter.b4451231.pr_1669": ("test_p012_cookiecutter_1669.py", "test_keep_project_on_failure", False),
    "cookiecutter__cookiecutter.b4451231.pr_1779": ("test_p012_cookiecutter_1779.py", "test_no_jinja2_time_needed", False),
    "cookiecutter__cookiecutter.b4451231.pr_2010": ("test_p012_cookiecutter_2010.py", "test_nested_dict_merge_keeps_new_keys", False),
    "kurtmckee__feedparser.cad965a3.pr_334": ("test_p012_feedparser_334.py", "test_parse_basic_feed", True),
    "kurtmckee__feedparser.cad965a3.pr_469": ("test_p012_feedparser_469.py", "test_prefix_read_negative_size", True),
    "kurtmckee__feedparser.cad965a3.pr_417": ("test_p012_feedparser_417.py", "test_parse_no_cgi_name_error", True),
    "kurtmckee__feedparser.cad965a3.pr_344": ("test_p012_feedparser_344.py", "test_idn_import_and_parse", True),
    "conan-io__conan.86f29e13.pr_16559": ("test_p012_conan_16559.py", "test_ampersand_colon_matches_consumer", False),
    "conan-io__conan.86f29e13.pr_13365": ("test_p012_conan_13365.py", "test_subfolder_import_error_surfaced", False),
    "conan-io__conan.86f29e13.pr_13112": ("test_p012_conan_13112.py", "test_remote_add_force", False),
    "conan-io__conan.86f29e13.pr_14852": ("test_p012_conan_14852.py", "test_source_with_remote_python_requires", False),
}

REPOS = {
    "keleshev__schema.24a30457.pr_331": "keleshev/schema",
    "cookiecutter__cookiecutter.b4451231.pr_1669": "cookiecutter/cookiecutter",
    "cookiecutter__cookiecutter.b4451231.pr_1779": "cookiecutter/cookiecutter",
    "cookiecutter__cookiecutter.b4451231.pr_2010": "cookiecutter/cookiecutter",
    "kurtmckee__feedparser.cad965a3.pr_334": "kurtmckee/feedparser",
    "kurtmckee__feedparser.cad965a3.pr_469": "kurtmckee/feedparser",
    "kurtmckee__feedparser.cad965a3.pr_417": "kurtmckee/feedparser",
    "kurtmckee__feedparser.cad965a3.pr_344": "kurtmckee/feedparser",
    "conan-io__conan.86f29e13.pr_16559": "conan-io/conan",
    "conan-io__conan.86f29e13.pr_13365": "conan-io/conan",
    "conan-io__conan.86f29e13.pr_13112": "conan-io/conan",
    "conan-io__conan.86f29e13.pr_14852": "conan-io/conan",
}


def load_freeze_items():
    freeze = json.loads((HERE / "dev12-freeze.json").read_text())["items"]
    sub = json.loads((HERE / "dev12-substitution.json").read_text())
    items = [it for it in freeze
             if it["instance_id"] != sub["removed"]["instance_id"]]
    items.append(sub["added"])
    assert len(items) == 12, len(items)
    return items


def issue_text(source_file, row_index, iid, traj_id):
    c = duckdb.connect()
    got = c.execute(
        "SELECT messages FROM read_parquet(?) WHERE file_row_number=? AND instance_id=? AND traj_id=?",
        [str(ROOT / source_file), row_index, iid, traj_id],
    ).fetchone()
    if not got:
        raise ValueError("atomic source identity mismatch: " + iid)
    for m in json.loads(got[0]):
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                x.get("text", "") for x in content if isinstance(x, dict))
        if not isinstance(content, str):
            continue
        i = content.find("<pr_description>")
        if i >= 0:
            j = content.find("</pr_description>")
            body = content[i + len("<pr_description>"):j if j >= 0 else len(content)]
            return body.strip()
    raise ValueError("no <pr_description> in trajectory user message: " + iid)


def unified_new_file(path, text):
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)
    out = ["--- /dev/null\n", "+++ b/%s\n" % path]
    out.append("@@ -0,0 +1,%d @@\n" % len(lines))
    out.extend("+" + ln if ln.endswith("\n") else "+" + ln + "\n" for ln in lines)
    return "".join(out)


def main():
    items = load_freeze_items()
    by_iid = {it["instance_id"]: it for it in items}
    assert set(by_iid) == set(TASKS), set(TASKS) ^ set(by_iid)
    patch_dir = HERE / "test-patches"
    patch_dir.mkdir(exist_ok=True)
    rows = []
    for iid, (fname, node, needs_sgml) in TASKS.items():
        it = by_iid[iid]
        problem = issue_text(it["source_file"], it["row_index"], iid,
                             it["selected_traj_id"])
        body = (CRITERIA / fname).read_text()
        diff = unified_new_file("tests/" + fname, body)
        if needs_sgml:
            sgml = (CRITERIA / "sgmllib.py").read_text()
            diff += unified_new_file("tests/p012_lib/sgmllib.py", sgml)
            diff += unified_new_file("tests/p012_lib/__init__.py", "")
        (patch_dir / (iid + ".patch")).write_text(diff)
        rows.append({
            "instance_id": iid,
            "repo": REPOS[iid],
            "base_commit": json.loads(
                (HERE / "task-env" / iid / "provenance.json").read_text()
            )["buggy_commit"],
            "problem_statement": problem,
            "patch": "",
            "test_patch": diff,
            "hints_text": "",
            "FAIL_TO_PASS": ["tests/" + fname + "::" + node],
            "PASS_TO_PASS": [],
        })
    out = HERE / "tasks-selected.jsonl"
    out.write_text("".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "count": len(rows),
        "source": "dev12-freeze.json minus invalid pr_1981 plus substitution pr_14852",
        "test_selection": "one criterion FAIL_TO_PASS node per task, verified fail-on-buggy/pass-on-fixed",
        "tasks_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "tasks_bytes": out.stat().st_size,
    }
    (HERE / "tasks-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
