#!/usr/bin/env python3
"""Small CPU CLI integration check for the actual P005 packaged skills.

Uses synthetic repositories only. Each invocation gets a freshly materialized
skill directory, while PYTHONPATH/VIRTUAL_ENV/TMPDIR match the official
SubprocessManager layout. No diagnosis flag, host callbacks, task data or model.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
APP = 'def value():\n    return 1\n\nif __name__ == "__main__":\n    print("VALUE=%d" % value())\n'
HELPER = 'def untouched():\n    return "base"\n'
ISSUE = 'Return 2 from value(). Preserve untouched().\nExact text: \\n "quoted" 中文.\n'


class Fixture:
    def __init__(self, base, variant):
        self.base = base
        self.workspace = base / "workspace"
        self.tmp = base / "tmp"
        self.venv = base / "venv"
        for path in (self.workspace, self.tmp, self.venv):
            path.mkdir(parents=True)
        self.skills = ROOT / "research/005/candidates" / variant / "agent/skills"
        self.env = dict(os.environ, PYTHONPATH=str(self.workspace),
                        VIRTUAL_ENV=str(self.venv), TMPDIR=str(self.tmp),
                        TEMP=str(self.tmp), TMP=str(self.tmp),
                        GIT_AUTHOR_NAME="P005 local check", GIT_AUTHOR_EMAIL="local@example.invalid",
                        GIT_COMMITTER_NAME="P005 local check", GIT_COMMITTER_EMAIL="local@example.invalid")
        (self.workspace / "app.py").write_text(APP)
        (self.workspace / "helper.py").write_text(HELPER)
        self.git("init", "-q")
        self.git("add", "app.py", "helper.py")
        self.git("-c", "commit.gpgsign=false", "commit", "-qm", "synthetic baseline")
        (self.tmp / "issue.txt").write_text(ISSUE)

    def git(self, *args):
        result = subprocess.run(["git", *args], cwd=self.workspace, env=self.env,
                                capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        return result.stdout

    def call(self, skill, script, *args, expected_status=None):
        # Mirrors ADK's fresh materialization + chdir, including sibling imports.
        with tempfile.TemporaryDirectory(prefix="skill-", dir=self.tmp) as td:
            materialized = Path(td)
            shutil.copytree(self.skills / skill / "scripts", materialized / "scripts")
            result = subprocess.run(
                [sys.executable, str(materialized / "scripts" / script), *args],
                cwd=materialized, env=self.env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, (skill, args, result.stderr)
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise AssertionError((skill, args, result.stdout, result.stderr)) from exc
        if expected_status:
            assert data.get("status") == expected_status, data
        else:
            assert data.get("status") not in ("error", "invalid_span", "invalid_replacement", "stale_source"), data
        return data


def check_edit_state_probe(f):
    span = f.call("source_edit", "source_edit.py", "inspect", "--path", "app.py", "--start", "1", "--end", "2")
    assert span["span_id"], span
    replacement = "def value():\n    return 2\n"
    applied = f.call("source_edit", "source_edit.py", "apply", "--span-id", span["span_id"],
                     "--replacement", replacement, expected_status="applied")
    assert (f.workspace / "app.py").read_text().startswith(replacement)
    # Reusing a pre-edit snapshot must not silently edit the changed source.
    f.call("source_edit", "source_edit.py", "apply", "--span-id", span["span_id"],
           "--replacement", replacement, expected_status="stale_source")

    located = f.call("source_probe", "source_probe.py", "locate", "--query", "value", "--question", "Where is value produced?")
    assert "app.py" in json.dumps(located), located
    inspected = f.call("source_probe", "source_probe.py", "inspect", "--target", "app.py:value", "--question", "What does value return?")
    assert "return 2" in json.dumps(inspected), inspected
    probe = f.call("source_probe", "source_probe.py", "probe", "--command", "python3 app.py",
                   "--question", "Does the edited producer return 2?", "--assert", "value:VALUE=2", "--trace")
    assert probe["exit_kind"] == "ok" and probe["assertions"][0]["observed"], probe
    assert any(c["path"] == "app.py" and c["function"] == "value" for c in probe["calls"]), probe

    f.call("working_state", "working_state.py", "record-issue", "--issue-file", "/tmp/issue.txt")
    f.call("working_state", "working_state.py", "record",
           "--trigger", "calling value()", "--expected", "return 2", "--preserve", "untouched()",
           "--fact", "f1|" + probe["observation_id"] + "|producer returned 2",
           "--hypothesis", "h1|producer|correct producer value|f1|", "--chosen", "h1",
           "--next-kind", "inspect", "--next-target", "helper.py", "--next-purpose", "confirm adjacent behavior")
    shown = f.call("working_state", "working_state.py", "show")
    assert "h1" in json.dumps(shown) and "helper.py" in json.dumps(shown), shown
    handoff = f.call("working_state", "working_state.py", "write-handoff", "--code", "app.py:1-2")
    read = f.call("working_state", "working_state.py", "read-handoff", "--handle", handoff["handle"])
    content = read["handoff"]
    assert content["issue"]["text"] == ISSUE, content
    assert content["code"][0]["text"] == replacement
    assert "+    return 2" in content["diff"]["text"]
    assert content["state"]["next_action"]["target"] == "helper.py"
    # The only source diff is the intended edit; ephemeral skill/state files stay out.
    assert f.git("diff", "--name-only").splitlines() == ["app.py"]
    untracked = f.git("ls-files", "--others", "--exclude-standard").splitlines()
    assert not [p for p in untracked if not p.startswith("__pycache__/")], untracked
    return applied


def check_snapshot_restore(f):
    f.git("restore", "--source=HEAD", "--worktree", "--staged", "--", "app.py", "helper.py")
    clean = f.call("source_edit", "patch_store.py", "snapshot", "--note", "clean baseline")
    (f.workspace / "app.py").write_text(APP.replace("return 1", "return 2"))
    (f.workspace / "kept.py").write_text('VALUE = "snapshot"\n')
    saved = f.call("source_edit", "patch_store.py", "snapshot", "--note", "desired producer edit", "--parent", clean["patch_id"])
    (f.workspace / "app.py").write_text(APP.replace("return 1", "return 3"))
    (f.workspace / "helper.py").write_text(HELPER.replace('"base"', '"regression"'))
    (f.workspace / "kept.py").write_text('VALUE = "later"\n')
    (f.workspace / "later.py").write_text('VALUE = "remove me"\n')
    restored = f.call("source_edit", "patch_store.py", "restore", "--patch-id", saved["patch_id"])
    assert restored["restored_patch_id"] == saved["patch_id"]
    assert (f.workspace / "app.py").read_text() == APP.replace("return 1", "return 2")
    assert (f.workspace / "helper.py").read_text() == HELPER, "restore left post-snapshot tracked changes outside recorded paths"
    assert (f.workspace / "kept.py").read_text() == 'VALUE = "snapshot"\n'
    assert not (f.workspace / "later.py").exists(), "restore left a file created after the snapshot"
    f.call("source_edit", "patch_store.py", "restore", "--patch-id", clean["patch_id"])
    assert f.git("diff", "HEAD", "--") == "", "restoring an empty baseline snapshot did not restore HEAD"
    assert not (f.workspace / "kept.py").exists()


def main():
    errors = []
    for variant in ("single_state", "one_handoff"):
        with tempfile.TemporaryDirectory(prefix="p005-skill-integration-") as td:
            f = Fixture(Path(td), variant)
            for name, check in (("edit/state/handoff/probe", check_edit_state_probe), ("snapshot/restore", check_snapshot_restore)):
                try:
                    check(f)
                except Exception as exc:
                    errors.append({"variant": variant, "check": name, "error": str(exc)})
                    print(json.dumps(errors[-1], ensure_ascii=False))
                else:
                    print(variant + ": " + name + " passed")
    if errors:
        raise SystemExit("P005 CLI integration failed (%d checks); no model or real tasks were used." % len(errors))
    print("P005 CLI integration passed for both packaged agents; no root override/diagnosis flags, model calls, or competition tasks.")


if __name__ == "__main__":
    main()
