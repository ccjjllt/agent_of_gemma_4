"""Offline schema check for the train-mode SFT block record. No 31B, no GPU."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLOCKED = HERE / "blocked.json"
LENGTH_REPORT = ROOT / "research/005/length-census-v2-20261010/length_report.json"
DDP_YAML = ROOT / "training/configs/sft_qat_nf4_ddp.yaml"
RESULT = ROOT / "research/005/save-on-cpu-6032-trainmode-20261011/result.json"
LONGSEQ = ROOT / "research/005/save-on-cpu-6032-trainmode-20261011/blocked-longseq.json"
WINDOW = ROOT / "research/005/data-expansion-cpu-20261011/window_report.json"
REQUIRED = (
    "trainable_ok",
    "chosen_max_seq_len",
    "window_counts_train",
    "length_report",
    "reason",
    "gpu_launched",
    "complete_train_length_backprop",
    "new_independent_train_issues",
)


class BlockRecordTests(unittest.TestCase):
    def test_blocked_json_schema_and_gates(self):
        data = json.loads(BLOCKED.read_text(encoding="utf-8"))
        for key in REQUIRED:
            self.assertIn(key, data, key)
        self.assertEqual(data["status"], "blocked")
        self.assertIs(data["trainable_ok"], False)
        self.assertIs(data["sft_ran"], False)
        self.assertEqual(data["chosen_max_seq_len"], 8192)
        self.assertIs(data["gpu_launched"], False)
        self.assertIs(data["gpu_sft"], False)
        self.assertIs(data["complete_train_length_backprop"], False)
        self.assertIs(data["first_adam_step_completed"], False)
        self.assertIs(data["new_independent_train_issues"], False)
        self.assertEqual(data["new_independent_issues"], 0)
        self.assertEqual(data["new_recovery_events"], 0)
        self.assertEqual(
            data["length_report"],
            "research/005/length-census-v2-20261010/length_report.json",
        )
        census = json.loads(LENGTH_REPORT.read_text(encoding="utf-8"))
        self.assertIs(census["trainable_ok"], False)
        self.assertEqual(census["chosen_max_seq_len"], 8192)
        self.assertEqual(
            data["window_counts_train"],
            census["decision"]["window_counts_train"],
        )
        self.assertEqual(data["window_counts_train"]["8192"], 0)
        self.assertEqual(data["n_train_transactions"], 11)
        self.assertIn("complete train length", data["reason"])
        self.assertIn("Do not truncate", data["reason"])
        self.assertIn("No GPU", data["reason"])
        self.assertIn("p005-training-ddp-probe", data["reason"])
        self.assertIs(data["did_not_modify_sft_qat_nf4_ddp_yaml"], True)
        self.assertEqual(data["training_config_max_sequence_length_untouched"], 1024)
        yaml_text = DDP_YAML.read_text(encoding="utf-8")
        self.assertIn("max_sequence_length: 1024", yaml_text)
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        longseq = json.loads(LONGSEQ.read_text(encoding="utf-8"))
        window = json.loads(WINDOW.read_text(encoding="utf-8"))
        self.assertIs(result["first_adam_step_completed"], False)
        self.assertEqual(longseq["status"], "BLOCKED")
        self.assertEqual(window["new_recovery_event_count"], 0)
        self.assertIs(
            window["independent_issue_growth"]["corpus_cannot_grow_without_new_rollouts"],
            True,
        )


if __name__ == "__main__":
    unittest.main()
