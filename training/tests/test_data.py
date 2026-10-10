"""Small behavioral tests for observed-history preservation and split boundaries."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA))
import canonicalize as c
import transactions as t
import split_registry as s
import prepare_local as p
import length_census as lc
import sample_behaviors as sb


def trace(name="edit_file", args=None):
    return {"trace_id": "example", "group": {"issue": "a", "repo": "o/r", "commit": "abc"},
            "synthetic": True, "raw_turns": [
                {"role": "system", "content": "Use the supplied tools"},
                {"role": "user", "content": "Fix the observed behavior"},
                {"role": "assistant", "tool_name": name, "tool_call_id": "first",
                 "arguments": args or {"filepath": "x.py", "old_string": "a", "new_string": "b"}},
                {"role": "tool", "tool_call_id": "first",
                 "tool_response": {"status": "error", "error_type": "FileEditError"}},
                {"role": "assistant", "tool_name": "read_file", "tool_call_id": "second",
                 "arguments": {"filepath": "x.py"}},
                {"role": "tool", "tool_call_id": "second",
                 "tool_response": {"status": "ok", "content": "a"}},
                {"role": "assistant", "tool_name": "edit_file", "tool_call_id": "third",
                 "arguments": {"filepath": "x.py", "old_string": "a", "new_string": "b"}},
                {"role": "tool", "tool_call_id": "third", "tool_response": {"status": "ok"}}]}


class DataTests(unittest.TestCase):
    def test_old_string_edit_never_becomes_span_apply(self):
        sample = c.canonicalize_trace(trace())
        call = sample["messages"][2]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "edit_file")
        self.assertIn("old_string", call["arguments"])
        self.assertNotIn("span_id", call["arguments"])
        self.assertFalse(sample["needs_replay"])

    def test_failed_exact_edit_is_context_not_positive_target(self):
        sample = c.canonicalize_trace(trace())
        self.assertNotIn(0, sample["target_assistant_steps"])
        self.assertIn(2, sample["target_assistant_steps"])
        self.assertIn("FileEditError", sample["messages"][3]["content"])

    def test_unknown_tool_requires_replay(self):
        sample = c.canonicalize_trace(trace("str_replace", {"old_str": "a", "new_str": "b"}))
        self.assertTrue(sample["needs_replay"])
        self.assertNotIn(0, sample["target_assistant_steps"])

    def test_bad_arguments_remain_context_not_targets(self):
        sample = c.canonicalize_trace(trace("edit_file", {"old_text": "a"}))
        self.assertFalse(sample["needs_replay"])
        self.assertEqual(sample["masked_invalid_assistant_steps"], [0])
        self.assertIn("old_text", sample["messages"][2]["tool_calls"][0]["function"]["arguments"])

    def test_orphan_reply_fails(self):
        raw = trace()
        raw["raw_turns"][3]["tool_call_id"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "orphan"):
            c.canonicalize_trace(raw)

    def test_late_targets_keep_issue_and_failed_action(self):
        sample = c.canonicalize_trace(trace())
        tx, _ = t.build_transaction(sample, 1, [4, 6])
        self.assertEqual(tx["messages"][1]["content"], "Fix the observed behavior")
        self.assertEqual(len(tx["messages"]), 8)
        self.assertEqual(tx["target_assistant_steps"], [1, 2])
        self.assertEqual(tx["behavior"], "tool_recovery")
        self.assertEqual(tx["recovery_evidence"][0]["failed_call"], "first")
        self.assertEqual(tx["recovery_evidence"][0]["filepath"], "x.py")
        self.assertIn("FileEditError", tx["recovery_evidence"][0]["failed_observation"])
        self.assertTrue(tx["observed_outcome"]["action_succeeded"])
        self.assertIsNone(tx["observed_outcome"].get("task_resolved"))

    def test_failure_without_recovery_not_positive_recovery(self):
        sample = c.canonicalize_trace(trace())
        tx, _ = t.build_transaction(sample, 1, [2, 4])
        self.assertNotEqual(tx["behavior"], "tool_recovery")

    def test_same_snapshot_and_issue_never_cross_splits(self):
        groups = {}
        for issue, repo, commit in [("a", "r", "x"), ("b", "r", "x"),
                                    ("a", "r", "y"), ("c", "r", "z"), ("d", "r2", "w")]:
            key = json.dumps([issue, repo, commit])
            groups[key] = {"issue": issue, "repo": repo, "commit": commit,
                           "members": [key], "content_hashes": [key], "split": "unassigned"}
        reg = {"groups": groups}
        s.assign(reg, .4)
        first = [g["split"] for g in groups.values() if g["issue"] in ("a", "b")]
        self.assertEqual(len(set(first)), 1)
        s.assign(reg, .4, heldout={"a"})
        self.assertTrue(all(g["split"] == "heldout-excluded" for g in groups.values()
                            if g["issue"] in ("a", "b")))

    def test_tool_sequence_alone_is_not_duplicate(self):
        a = c.canonicalize_trace(trace())
        b = c.canonicalize_trace(trace())
        b["messages"][1]["content"] = "A different bug"
        self.assertNotEqual(s.near_dup_key(a), s.near_dup_key(b))

    def test_include_failures_keeps_edit_file_not_span_apply(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_prepare_fixture(root, "ok_issue", "org/repo", "abc", "dev_panel", resolved=False)
            out = root / "out"
            default_report = p.prepare(root / "results", root / "splits.json",
                                       root / "eligibility.json", out / "default")
            self.assertEqual(default_report["traces"], [])
            self.assertTrue(default_report["resolved_only"])
            fail_report = p.prepare(root / "results", root / "splits.json",
                                    root / "eligibility.json", out / "failures",
                                    include_failures=True)
            self.assertEqual(len(fail_report["traces"]), 1)
            self.assertFalse(fail_report["resolved_only"])
            blob = (out / "failures" / "canonical.jsonl").read_text()
            sample = json.loads(blob.splitlines()[0])
            names = [call["function"]["name"]
                     for msg in sample["messages"] if msg.get("tool_calls")
                     for call in msg["tool_calls"]]
            self.assertEqual(names, ["edit_file", "read_file", "edit_file"])
            self.assertNotIn("span_apply", blob)
            self.assertNotIn("span_id", blob)
            self.assertIn("old_string", blob)
            self.assertIn("FileEditError", blob)
            self.assertNotIn(0, sample["target_assistant_steps"])
            self.assertEqual(sample.get("behavior"), "unlabeled")
            tx = json.loads((out / "failures" / "transactions.jsonl").read_text().splitlines()[0])
            self.assertEqual(tx["behavior"], "tool_recovery")
            self.assertNotEqual(tx["behavior"], "local_edit")

    def test_prepare_excludes_heldout_even_with_include_failures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_prepare_fixture(root, "held_issue", "org/repo", "abc", "heldout", resolved=True)
            _write_prepare_fixture(root, "ok_issue", "org/repo", "def", "dev_panel", resolved=True)
            report = p.prepare(root / "results", root / "splits.json",
                               root / "eligibility.json", root / "out",
                               include_failures=True)
            self.assertEqual(report["tasks"], ["ok_issue"])
            blob = (root / "out" / "canonical.jsonl").read_text()
            self.assertNotIn("held_issue", blob)
            self.assertNotIn("HELD_SECRET_SHOULD_NOT_LEAK", blob)

    def test_recovery_does_not_pair_same_tool_different_files(self):
        raw = trace()
        raw["raw_turns"][2]["arguments"]["filepath"] = "a.py"
        raw["raw_turns"][6]["arguments"]["filepath"] = "a.py"
        raw["raw_turns"].extend([
            {"role": "assistant", "tool_name": "edit_file", "tool_call_id": "fourth",
             "arguments": {"filepath": "b.py", "old_string": "a", "new_string": "b"}},
            {"role": "tool", "tool_call_id": "fourth",
             "tool_response": {"status": "error", "error_type": "FileEditError"}},
            {"role": "assistant", "tool_name": "edit_file", "tool_call_id": "fifth",
             "arguments": {"filepath": "b.py", "old_string": "a", "new_string": "b"}},
            {"role": "tool", "tool_call_id": "fifth", "tool_response": {"status": "ok"}},
        ])
        sample = c.canonicalize_trace(raw)
        tx, _ = t.build_transaction(sample, 1, [4, 6, 10])
        files = {item["filepath"] for item in tx["recovery_evidence"]}
        failed = {item["failed_call"] for item in tx["recovery_evidence"]}
        self.assertEqual(tx["behavior"], "tool_recovery")
        self.assertEqual(files, {"a.py", "b.py"})
        self.assertEqual(failed, {"first", "fourth"})
        paired = {(item["failed_call"], item["successful_call"]) for item in tx["recovery_evidence"]}
        self.assertEqual(paired, {("first", "third"), ("fourth", "fifth")})

    def test_success_on_other_file_is_not_recovery(self):
        raw = trace()
        raw["raw_turns"][2]["arguments"]["filepath"] = "a.py"
        raw["raw_turns"][6]["arguments"]["filepath"] = "b.py"
        sample = c.canonicalize_trace(raw)
        tx, _ = t.build_transaction(sample, 1, [4, 6])
        self.assertNotEqual(tx["behavior"], "tool_recovery")
        self.assertEqual(tx["recovery_evidence"], [])
        self.assertEqual(tx["behavior"], "local_edit")

    def test_later_success_closes_recovery_so_it_is_not_repeated(self):
        raw = trace()
        raw["raw_turns"].extend([
            {"role": "assistant", "tool_name": "edit_file", "tool_call_id": "fourth",
             "arguments": {"filepath": "x.py", "old_string": "b", "new_string": "c"}},
            {"role": "tool", "tool_call_id": "fourth", "tool_response": {"status": "ok"}},
        ])
        sample = c.canonicalize_trace(raw)
        first, _ = t.build_transaction(sample, 1, [4, 6])
        later, _ = t.build_transaction(sample, 2, [8])
        self.assertEqual(first["behavior"], "tool_recovery")
        self.assertEqual(first["recovery_evidence"][0]["successful_call"], "third")
        self.assertNotEqual(later["behavior"], "tool_recovery")
        self.assertEqual(later["recovery_evidence"], [])
        self.assertEqual(later["behavior"], "local_edit")

    def test_jammed_filepath_still_matches_same_file_recovery(self):
        raw = trace("read_file", {"filepath": "x.py`,start_line:1"})
        raw["raw_turns"][6]["tool_name"] = "read_file"
        raw["raw_turns"][6]["arguments"] = {"filepath": "x.py"}
        sample = c.canonicalize_trace(raw)
        tx, _ = t.build_transaction(sample, 1, [4, 6])
        self.assertEqual(tx["behavior"], "tool_recovery")
        self.assertEqual(tx["recovery_evidence"][0]["filepath"], "x.py")
        self.assertEqual(tx["recovery_evidence"][0]["failed_call"], "first")

    def test_submit_patch_is_verify_close(self):
        raw = {"trace_id": "example", "group": {"issue": "a", "repo": "o/r", "commit": "abc"},
               "synthetic": True, "raw_turns": [
                   {"role": "system", "content": "Use the supplied tools"},
                   {"role": "user", "content": "Fix the observed behavior"},
                   {"role": "assistant", "tool_name": "edit_file", "tool_call_id": "e1",
                    "arguments": {"filepath": "x.py", "old_string": "a", "new_string": "b"}},
                   {"role": "tool", "tool_call_id": "e1", "tool_response": {"status": "ok"}},
                   {"role": "assistant", "tool_name": "submit_patch", "tool_call_id": "s1",
                    "arguments": {}},
                   {"role": "tool", "tool_call_id": "s1", "tool_response": {"status": "ok"}}]}
        sample = c.canonicalize_trace(raw)
        tx, _ = t.build_transaction(sample, 1, [4])
        self.assertEqual(tx["behavior"], "verify_close")
        self.assertIn("verify_close", tx["behaviors_present"])

    def test_register_keeps_recovery_not_file_order_first_three(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tx.jsonl"
            rows = []
            for i in range(4):
                rows.append(_sample_tx("iss-tx%02d" % (i + 1), "iss", "local_edit",
                                       "edit body %d" % i, sample_id="iss-aaaa"))
            rows.append(_sample_tx("iss-tx99", "iss", "tool_recovery", "recovery body",
                                   sample_id="iss-bbbb",
                                   evidence=[{"failed_call": "f1", "successful_call": "s1"}]))
            rows.append(_sample_tx("iss-tx98", "iss", "verify_close", "verify body",
                                   sample_id="iss-cccc"))
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            reg = s.register(path, max_per_issue=3, seed=20261010, reserved_issues=set())
            members = next(iter(reg["groups"].values()))["members"]
            self.assertIn("iss-tx99", members)
            self.assertIn("iss-tx98", members)
            self.assertEqual(len(members), 3)

    def test_register_drops_reserved_pairing_issue(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tx.jsonl"
            rows = [
                _sample_tx("fastapi_13713-tx01", "fastapi_13713", "local_edit", "a",
                           repo="fastapi/fastapi", commit="deadbeef"),
                _sample_tx("ok-tx01", "ok_issue", "local_edit", "b", repo="o/r", commit="abc"),
                _sample_tx("ok-tx02", "ok_issue", "tool_recovery", "c", repo="o/r", commit="abc",
                           evidence=[{"failed_call": "f", "successful_call": "s"}]),
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            reg = s.register(path, max_per_issue=3, seed=20261010)
            issues = {g["issue"] for g in reg["groups"].values()}
            self.assertNotIn("fastapi_13713", issues)
            self.assertIn("ok_issue", issues)

    def test_missing_behavior_is_reported_not_invented(self):
        selected, audit = sb.select_issue_transactions([
            _sample_tx("a-tx01", "a", "local_edit", "only edits", sample_id="a-one"),
            _sample_tx("a-tx02", "a", "local_edit", "more edits", sample_id="a-two"),
        ], max_per_issue=3, seed=20261010)
        self.assertTrue(all(tx["behavior"] == "local_edit" for tx in selected))
        self.assertIn("tool_recovery", audit["missing_behaviors"])
        self.assertIn("verify_close", audit["missing_behaviors"])
        self.assertNotIn("tool_recovery", [tx["behavior"] for tx in selected])

    def test_census_refuses_v1_output_dir(self):
        with self.assertRaisesRegex(ValueError, "refusing to overwrite v1"):
            lc.refuse_v1_overwrite(lc.V1_OUTPUT_DIR, Path("/tmp/other-report"))


class CensusDecisionTests(unittest.TestCase):
    def test_majority_4096_and_8192_and_gap(self):
        a = lc.decide_max_seq_len([1000, 2000, 3000, 4000])
        self.assertEqual(a["chosen_max_seq_len"], 4096)
        self.assertTrue(a["trainable_ok"])
        self.assertTrue(a["four_k_capacity_unproven"])
        b = lc.decide_max_seq_len([5000, 5000, 5000, 3000])
        self.assertEqual(b["chosen_max_seq_len"], 8192)
        self.assertTrue(b["trainable_ok"])
        self.assertTrue(b["four_k_capacity_unproven"])
        c = lc.decide_max_seq_len([12000, 15000, 20000])
        self.assertEqual(c["chosen_max_seq_len"], 8192)
        self.assertFalse(c["trainable_ok"])
        self.assertIsNotNone(c["coverage_gap"])
        self.assertTrue(c["majority_fit_16384"])
        self.assertNotEqual(c["chosen_max_seq_len"], 16384)
        self.assertNotEqual(c["chosen_max_seq_len"], 2048)
        empty = lc.decide_max_seq_len([])
        self.assertFalse(empty["trainable_ok"])
        self.assertNotEqual(empty["chosen_max_seq_len"], 2048)


def _sample_tx(tid, issue, behavior, text, repo="o/r", commit="abc", evidence=None, sample_id=None):
    return {
        "transaction_id": tid,
        "sample_id": sample_id or tid,
        "group": {"issue": issue, "repo": repo, "commit": commit},
        "messages": [{"role": "user", "content": text},
                     {"role": "assistant", "content": text}],
        "target_assistant_steps": [0],
        "behavior": behavior,
        "recovery_evidence": evidence or [],
        "observed_outcome": {"task_resolved": False, "action_succeeded": True},
    }


def _atif_trace(user_text="Fix the observed behavior"):
    return {
        "schema_version": "ATIF-v1.0",
        "steps": [
            {"step_id": 1, "source": "system", "message": "Use the supplied tools"},
            {"step_id": 2, "source": "user", "message": user_text},
            {"step_id": 3, "source": "agent", "message": "",
             "tool_calls": [{"tool_call_id": "first", "function_name": "edit_file",
                             "arguments": {"filepath": "x.py", "old_string": "a", "new_string": "b"}}],
             "observation": {"content": json.dumps({"status": "error", "error_type": "FileEditError"})}},
            {"step_id": 4, "source": "agent", "message": "",
             "tool_calls": [{"tool_call_id": "second", "function_name": "read_file",
                             "arguments": {"filepath": "x.py"}}],
             "observation": {"content": json.dumps({"status": "ok", "content": "a"})}},
            {"step_id": 5, "source": "agent", "message": "",
             "tool_calls": [{"tool_call_id": "third", "function_name": "edit_file",
                             "arguments": {"filepath": "x.py", "old_string": "a", "new_string": "b"}}],
             "observation": {"content": json.dumps({"status": "ok"})}},
        ],
    }


def _write_prepare_fixture(root, issue, repo, commit, split, resolved):
    results = root / "results" / "run" / "v" / "r1" / "attempts" / ("01-%s" % issue)
    traces = results / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    user = "HELD_SECRET_SHOULD_NOT_LEAK" if split == "heldout" else "Fix the observed behavior"
    (traces / ("trace_%s.json" % issue)).write_text(json.dumps(_atif_trace(user)))
    (results / "result.json").write_text(json.dumps({"resolved": resolved}))
    splits_path = root / "splits.json"
    records = []
    if splits_path.exists():
        records = json.loads(splits_path.read_text())["records"]
    records = [r for r in records if r["instance_id"] != issue]
    records.append({"instance_id": issue, "repo": repo, "base_commit": commit, "split": split})
    splits_path.write_text(json.dumps({"records": records, "dev_panel_ids": []}))
    elig_path = root / "eligibility.json"
    tasks = []
    if elig_path.exists():
        tasks = json.loads(elig_path.read_text())["tasks"]
    tasks = [t for t in tasks if t.get("task_id") != issue]
    tasks.append({"task_id": issue, "eligibility": "eligible"})
    elig_path.write_text(json.dumps({"tasks": tasks}))


if __name__ == "__main__":
    unittest.main()
