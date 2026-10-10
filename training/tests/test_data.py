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


if __name__ == "__main__":
    unittest.main()
