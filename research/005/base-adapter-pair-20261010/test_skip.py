"""Offline schema check for the Task 5 pairing skip record. No GPU."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SKIPPED = HERE / "skipped.json"
RESULTS = HERE / "results.md"
LENGTH_REPORT = ROOT / "research/005/length-census-20261010/length_report.json"
TASK4_BLOCKED = ROOT / "research/005/sft-candidate-20261010/blocked.json"
TASK3_RESULT = ROOT / "research/005/ct-adapter-probe-20261010/result.json"
FAKE_CARRIER = ROOT / "notebooks/005-base-adapter-pair-single-cell.py"
REQUIRED = (
    "skipped",
    "reason",
    "task3",
    "task4",
    "train_issues",
    "candidate_eval_issues",
    "gpu_launched",
    "competition_submit",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class SkipRecordTests(unittest.TestCase):
    def test_skipped_json_schema_and_gate(self):
        data = load_json(SKIPPED)
        for key in REQUIRED:
            self.assertIn(key, data, key)
        self.assertIs(data["skipped"], True)
        self.assertIs(data["pairing_ran"], False)
        self.assertIs(data["gpu_launched"], False)
        self.assertIs(data["competition_submit"], False)
        self.assertIs(data["diagnostic_adapter_used_as_repair_gain_arm"], False)
        self.assertIs(data["notebook_carrier_created"], False)
        self.assertIs(data["resolved_delta_table"], None)
        self.assertEqual(data["task3"]["classification"], "tool_call_ok_no_repair_claim")
        self.assertIs(data["task4"]["trainable_ok"], False)
        self.assertIs(data["task4"]["gpu_launched"], False)
        self.assertIn("trainable_ok=false", data["reason"])
        self.assertIn("tool_call_ok_no_repair_claim", data["reason"])
        self.assertIn("repair-gain", data["reason"])
        self.assertIn("No GPU", data["reason"])

    def test_candidate_eval_disjoint_from_train_and_from_census_pool(self):
        data = load_json(SKIPPED)
        census = load_json(LENGTH_REPORT)
        blocked = load_json(TASK4_BLOCKED)
        self.assertEqual(data["train_issues"], census["train_issues"])
        self.assertEqual(
            data["dev_panel_not_in_train"],
            census["dev_panel_not_in_train"],
        )
        self.assertEqual(len(census["dev_panel_not_in_train"]), 19)
        self.assertIs(blocked["trainable_ok"], False)
        self.assertIs(blocked["gpu_launched"], False)
        train = set(data["train_issues"])
        pool = set(data["dev_panel_not_in_train"])
        cand = data["candidate_eval_issues"]
        self.assertGreaterEqual(len(cand), 8)
        self.assertLessEqual(len(cand), 12)
        self.assertEqual(len(cand), len(set(cand)))
        self.assertTrue(set(cand).isdisjoint(train))
        self.assertTrue(set(cand) <= pool)
        held = set(data["candidate_eval_selection"]["held_back_from_candidate"])
        self.assertEqual(set(cand) | held, pool)
        self.assertTrue(held.isdisjoint(set(cand)))

    def test_task3_classification_is_compatibility_only(self):
        probe = load_json(TASK3_RESULT)
        self.assertEqual(probe["classification"], "tool_call_ok_no_repair_claim")
        self.assertIs(probe["repair_success_claimed"], False)
        self.assertIs(probe["scored_panel_run"], False)

    def test_results_md_and_no_fake_carrier(self):
        text = RESULTS.read_text(encoding="utf-8")
        self.assertIn("SKIPPED", text)
        self.assertIn("No resolved-delta table", text)
        self.assertIn("Did **not** create `notebooks/005-base-adapter-pair-single-cell.py`", text)
        self.assertIn("002-joint-explore-v30-single-cell.py", text)
        self.assertIn("gpu_launched=false", text)
        self.assertFalse(FAKE_CARRIER.exists(), "do not ship a fake completed carrier")


if __name__ == "__main__":
    unittest.main()
