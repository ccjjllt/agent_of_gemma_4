"""Tiny JSON check for the Task 3 SFT/pairing skip. No GPU."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SKIPPED = HERE / "skipped.json"
RESULTS = HERE / "results.md"
TASK1 = ROOT / "research/005/save-on-cpu-6032-20261010/result.json"
BLOCKED = ROOT / "research/005/save-on-cpu-6032-20261010/blocked-longseq.json"
CENSUS = ROOT / "research/005/length-census-v2-20261010/length_report.json"
FAKE_CARRIER = ROOT / "notebooks/005-base-adapter-pair-single-cell.py"
RESERVED = [
    "fastapi_13713", "requests_6592", "requests_6644", "requests_7309",
    "requests_7505", "rich_3130", "rich_3296", "rich_3469",
]


class SkipRecordTests(unittest.TestCase):
    def test_skip_gate_and_pointers(self):
        data = json.loads(SKIPPED.read_text(encoding="utf-8"))
        task1 = json.loads(TASK1.read_text(encoding="utf-8"))
        blocked = json.loads(BLOCKED.read_text(encoding="utf-8"))
        census = json.loads(CENSUS.read_text(encoding="utf-8"))
        self.assertIs(data["skipped"], True)
        self.assertIs(data["sft_ran"], False)
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
        self.assertEqual(data["task1"]["kernel"], "ccjjllt/p005-save-on-cpu-6032")
        self.assertIs(data["task1"]["gpu_sft"], False)
        self.assertIs(task1["first_adam_step_completed"], False)
        self.assertIsNone(data["task1"]["adapter_fingerprint"])
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIs(data["task2"]["backprop_9841"], False)
        self.assertIs(data["task2"]["backprop_12832"], False)
        self.assertIs(data["task2"]["new_independent_train_issues"], False)
        self.assertIs(data["task2"]["gpu_sft"], False)
        self.assertIn("gpu_sft=false", data["reason"])
        self.assertIn("p005-save-on-cpu-6032", data["reason"])
        self.assertIn("No GPU", data["reason"])
        self.assertIn("p005-training-ddp-probe", data["reason"])
        self.assertIn("Live MLP/dequant", data["next_cut"])
        self.assertIn("Not a 16K retry", data["next_cut"])
        text = RESULTS.read_text(encoding="utf-8")
        self.assertIn("No resolved-delta table", text)
        self.assertIn("ccjjllt/p005-save-on-cpu-6032", text)
        self.assertIn("Live MLP/dequant", text)
        self.assertIn("**Not** a 16K retry", text)
        self.assertIn("Did **not** create `notebooks/005-base-adapter-pair-single-cell.py`", text)
        self.assertFalse(FAKE_CARRIER.exists())


if __name__ == "__main__":
    unittest.main()
