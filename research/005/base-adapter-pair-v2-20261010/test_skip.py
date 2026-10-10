"""Tiny JSON check for the Task 3 pairing skip. No GPU."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SKIPPED = HERE / "skipped.json"
RESULTS = HERE / "results.md"
TASK2 = ROOT / "research/005/capacity-sft-v2-20261010/result.json"
CENSUS = ROOT / "research/005/length-census-v2-20261010/length_report.json"
FAKE_CARRIER = ROOT / "notebooks/005-base-adapter-pair-single-cell.py"
RESERVED = [
    "fastapi_13713", "requests_6592", "requests_6644", "requests_7309",
    "requests_7505", "rich_3130", "rich_3296", "rich_3469",
]


class SkipRecordTests(unittest.TestCase):
    def test_skip_gate_and_pointers(self):
        data = json.loads(SKIPPED.read_text(encoding="utf-8"))
        task2 = json.loads(TASK2.read_text(encoding="utf-8"))
        census = json.loads(CENSUS.read_text(encoding="utf-8"))
        self.assertIs(data["skipped"], True)
        self.assertIs(data["pairing_ran"], False)
        self.assertIs(data["gpu_launched"], False)
        self.assertIs(data["gpu_sft"], False)
        self.assertIs(data["resolved_delta_table"], None)
        self.assertIs(data["diagnostic_adapter_used_as_repair_gain_arm"], False)
        self.assertEqual(data["diagnostic_adapter_kernel"], "ccjjllt/p005-training-ddp-probe")
        self.assertEqual(data["reserved_pairing_issues"], RESERVED)
        self.assertEqual(data["reserved_pairing_issues"], census["reserved_pairing_issues"])
        train_select = set(census["train_issues"]) | set(census["select_issues"])
        self.assertTrue(set(RESERVED).isdisjoint(train_select))
        self.assertIs(data["reserved_disjoint_from_train_select"], True)
        self.assertEqual(data["task2"]["kernel"], "ccjjllt/p005-capacity-sft-v2")
        self.assertIs(data["task2"]["gpu_sft"], False)
        self.assertIs(task2["gpu_sft"], False)
        self.assertIsNone(task2["adapter_fingerprint"])
        self.assertIn("gpu_sft=false", data["reason"])
        self.assertIn("p005-training-ddp-probe", data["reason"])
        self.assertIn("No GPU", data["reason"])
        text = RESULTS.read_text(encoding="utf-8")
        self.assertIn("No resolved-delta table", text)
        self.assertIn("memory-feasible complete-tx SFT", text)
        self.assertIn("Did **not** create `notebooks/005-base-adapter-pair-single-cell.py`", text)
        self.assertFalse(FAKE_CARRIER.exists())


if __name__ == "__main__":
    unittest.main()
