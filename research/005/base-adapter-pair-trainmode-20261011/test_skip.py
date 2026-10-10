"""Tiny JSON check for the Task 4 SFT/pairing skip. No GPU."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SKIPPED = HERE / "skipped.json"
RESULTS = HERE / "results.md"
TASK1 = ROOT / "research/005/save-on-cpu-6032-trainmode-20261011/result.json"
BLOCKED = ROOT / "research/005/save-on-cpu-6032-trainmode-20261011/blocked-longseq.json"
SFT_BLOCK = ROOT / "research/005/sft-candidate-20261011/blocked.json"
CENSUS = ROOT / "research/005/length-census-v2-20261010/length_report.json"
HISTORICAL = ROOT / "research/005/save-on-cpu-6032-20261010/result.json"
WINDOW = ROOT / "research/005/data-expansion-cpu-20261011/window_report.json"
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
        sft = json.loads(SFT_BLOCK.read_text(encoding="utf-8"))
        census = json.loads(CENSUS.read_text(encoding="utf-8"))
        historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
        window = json.loads(WINDOW.read_text(encoding="utf-8"))
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
        self.assertEqual(data["task1"]["kernel_live_slug"], "ccjjllt/p005-save-on-cpu-6032-train-mode")
        self.assertEqual(data["task1"]["kernel_metadata_id"], "ccjjllt/p005-save-on-cpu-6032-tm")
        self.assertIs(data["task1"]["gpu_sft"], False)
        self.assertIs(task1["first_adam_step_completed"], False)
        self.assertIs(task1["comparison_valid_for_checkpointed_training"], True)
        self.assertIsNone(data["task1"]["adapter_fingerprint"])
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIs(blocked["gpu_launched"], False)
        self.assertIs(data["task2"]["backprop_9841"], False)
        self.assertIs(data["task2"]["backprop_12832"], False)
        self.assertIs(data["task2"]["new_independent_train_issues"], False)
        self.assertEqual(data["task2"]["new_independent_issues"], 0)
        self.assertEqual(data["task2"]["new_recovery_events"], 0)
        self.assertIs(window["independent_issue_growth"]["corpus_cannot_grow_without_new_rollouts"], True)
        self.assertEqual(window["new_recovery_event_count"], 0)
        self.assertIs(data["task2"]["gpu_sft"], False)
        self.assertIn("gpu_sft=false", data["reason"])
        self.assertIn("p005-save-on-cpu-6032-train-mode", data["reason"])
        self.assertIn("p005-save-on-cpu-6032-tm", data["reason"])
        self.assertIn("No GPU", data["reason"])
        self.assertIn("p005-training-ddp-probe", data["reason"])
        self.assertIn("SDPA", data["next_cut"])
        self.assertIn("Not a 16K retry", data["next_cut"])
        self.assertEqual(task1["status"], "valid_comparison_no_first_adam")
        self.assertEqual(historical["status"], "invalid_checkpointed_training_comparison")
        self.assertEqual(sft["status"], "blocked")
        self.assertIs(sft["gpu_launched"], False)
        self.assertIs(sft["complete_train_length_backprop"], False)
        self.assertIs(sft["new_independent_train_issues"], False)
        text = RESULTS.read_text(encoding="utf-8")
        self.assertIn("No resolved-delta table", text)
        self.assertIn("ccjjllt/p005-save-on-cpu-6032-train-mode", text)
        self.assertIn("ccjjllt/p005-save-on-cpu-6032-tm", text)
        self.assertIn("**Not** a 16K retry", text)
        self.assertIn("Did **not** create `notebooks/005-base-adapter-pair-single-cell.py`", text)
        self.assertFalse(FAKE_CARRIER.exists())


if __name__ == "__main__":
    unittest.main()
