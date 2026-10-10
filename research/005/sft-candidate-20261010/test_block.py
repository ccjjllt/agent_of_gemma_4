"""Offline schema check for the Task 4 SFT block record. No 31B, no GPU."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLOCKED = HERE / "blocked.json"
README = HERE / "README.md"
LENGTH_REPORT = ROOT / "research/005/length-census-20261010/length_report.json"
DDP_YAML = ROOT / "training/configs/sft_qat_nf4_ddp.yaml"
REQUIRED = (
    "trainable_ok",
    "chosen_max_seq_len",
    "window_counts_train",
    "length_report",
    "reason",
    "gpu_launched",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class BlockRecordTests(unittest.TestCase):
    def test_blocked_json_schema_and_census_pointer(self):
        data = load_json(BLOCKED)
        for key in REQUIRED:
            self.assertIn(key, data, key)
        self.assertIs(data["trainable_ok"], False)
        self.assertEqual(data["chosen_max_seq_len"], 8192)
        self.assertIs(data["gpu_launched"], False)
        self.assertEqual(
            data["length_report"],
            "research/005/length-census-20261010/length_report.json",
        )
        self.assertTrue(LENGTH_REPORT.is_file(), "missing Task 2 census")
        census = load_json(LENGTH_REPORT)
        self.assertIs(census["trainable_ok"], False)
        self.assertEqual(census["chosen_max_seq_len"], 8192)
        self.assertEqual(
            data["window_counts_train"],
            census["decision"]["window_counts_train"],
        )
        self.assertEqual(data["window_counts_train"]["4096"], 0)
        self.assertEqual(data["window_counts_train"]["8192"], 3)
        self.assertEqual(data["n_train_transactions"], 15)
        self.assertIsInstance(data["reason"], str)
        self.assertIn("trainable_ok=false", data["reason"])
        self.assertIn("Do not truncate", data["reason"])
        self.assertIn("3 in-window prefixes", data["reason"])
        self.assertIn("2K", data["reason"])
        self.assertIn("No GPU", data["reason"])

    def test_readme_states_block_and_unblock_paths(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn("BLOCKED", text)
        self.assertIn("complete-context", text)
        self.assertIn("4K/16K", text)
        self.assertIn("synthetic 2K", text)
        self.assertIn("gpu_launched=false", text)
        self.assertIn("sft_qat_nf4_ddp.yaml", text)

    def test_ddp_yaml_not_rewritten_as_if_8192_sft_ran(self):
        text = DDP_YAML.read_text(encoding="utf-8")
        self.assertIn("max_sequence_length: 1024", text)
        self.assertNotIn("max_sequence_length: 8192", text)
        data = load_json(BLOCKED)
        self.assertEqual(data["training_config_max_sequence_length_untouched"], 1024)
        self.assertIs(data["did_not_modify_sft_qat_nf4_ddp_yaml"], True)


if __name__ == "__main__":
    unittest.main()
