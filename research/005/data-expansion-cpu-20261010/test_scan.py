"""Offline checks for the Task 2 CPU expansion scan. No 31B, no GPU."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BLOCKED = ROOT / "research/005/save-on-cpu-6032-20261010/blocked-longseq.json"
V2 = ROOT / "research/005/length-census-v2-20261010"
RESERVED = (
    "fastapi_13713",
    "requests_6592",
    "requests_6644",
    "requests_7309",
    "requests_7505",
    "rich_3130",
    "rich_3296",
    "rich_3469",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExpansionScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("p005_scan_review", HERE / "scan.py")
        cls.scan = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scan)

    def test_reserved_snapshots_are_excluded_from_actual_measured_pool(self):
        records = [{"instance_id": "fastapi_13713", "repo": "r/a", "base_commit": "same"}]
        rows = [
            {"transaction_id": "reserved", "group": {"issue": "fastapi_13713", "repo": "r/a", "commit": "same"}},
            {"transaction_id": "alias", "group": {"issue": "new_alias", "repo": "r/a", "commit": "same"}},
            {"transaction_id": "keep", "group": {"issue": "new_issue", "repo": "r/b", "commit": "other"}},
        ]
        measured, dropped = self.scan.measured_pool(rows, records)
        self.assertEqual([r["transaction_id"] for r in measured], ["keep"])
        self.assertEqual(set(dropped), {"reserved", "alias"})

    def test_new_issue_growth_is_relative_to_historical_v2(self):
        frozen = {"prepare": {"tasks": ["old_issue"]}}
        measured = [{"issue": "old_issue"}, {"issue": "new_issue"}, {"issue": "new_issue"}]
        self.assertEqual(self.scan.new_issue_ids(measured, frozen), ["new_issue"])
        self.assertEqual(self.scan.new_issue_ids([{"issue": "old_issue"}], frozen), [])

    def test_blocked_longseq_no_gpu(self):
        data = load(BLOCKED)
        self.assertEqual(data["status"], "BLOCKED")
        self.assertIs(data["gpu_launched"], False)
        self.assertIs(data["first_adam_step_completed"], False)
        self.assertIs(data["did_not_push_9841"], True)
        self.assertIs(data["did_not_push_12832"], True)
        self.assertIs(data["did_not_retry_16k"], True)
        self.assertIs(data["did_not_rewrite_fsdp"], True)
        self.assertEqual(data["quota_this_task_s"], 0)
        self.assertIn("forward", data["traceback_class"]["baseline"]["failure_phase"])
        self.assertEqual(data["traceback_class"]["baseline"]["last_decoder_layer"], 1)
        self.assertEqual(data["traceback_class"]["baseline"]["tried_to_allocate_mib"], 248)
        self.assertIn("SIGKILL", data["traceback_class"]["save_on_cpu"]["host_signal"])
        ids = {row["raw_token_length"] for row in data["untested_complete_train_lengths"]}
        self.assertEqual(ids, {9841, 12832})

    def test_v2_census_not_overwritten(self):
        report = load(HERE / "window_report.json")
        self.assertIs(report["did_not_overwrite_v2_census"], True)
        by_path = {row["path"]: row["sha256"] for row in report["v2_census_sha256_unchanged"]}
        for path in (
            V2 / "length_report.json",
            V2 / "sample_list.json",
            V2 / "train_issues.json",
        ):
            rel = "research/005/length-census-v2-20261010/" + path.name
            self.assertEqual(by_path[rel], sha256_file(path), rel)

    def test_reserved_eight_out_and_no_new_issues(self):
        report = load(HERE / "window_report.json")
        lengths = load(HERE / "complete_tx_lengths.json")
        issues = {row["issue"] for row in lengths["transactions"]}
        self.assertTrue(issues.isdisjoint(RESERVED))
        self.assertEqual(report["reserved_in_measured"], [])
        self.assertEqual(len(lengths["transactions"]), 214)
        self.assertEqual(report["prepare"]["independent_task_count"], 9)
        growth = report["independent_issue_growth"]
        self.assertEqual(growth["new_eligible_issues_with_traces"], [])
        self.assertIs(growth["corpus_cannot_grow_without_new_rollouts"], True)

    def test_window_counts_and_short_labeled(self):
        report = load(HERE / "window_report.json")
        windows = report["all_complete_prefixes"]["windows"]
        self.assertEqual(windows["2048"]["transactions"], 0)
        self.assertEqual(windows["4096"]["transactions"], 0)
        self.assertEqual(windows["8192"]["transactions"], 24)
        self.assertEqual(windows["16384"]["transactions"], 98)
        self.assertEqual(report["naturally_short_labeled_le_8192"]["n"], 6)
        self.assertEqual(report["naturally_short_labeled_le_8192"]["independent_recovery_events"], 0)
        self.assertEqual(report["known_length_check"]["measured"], 6032)
        labeled = load(HERE / "short_labeled.json")["transactions"]
        self.assertEqual(len(labeled), 6)
        self.assertTrue(all(r["behavior"] == "local_edit" for r in labeled))
        self.assertEqual(sum(1 for r in labeled if not r["in_v2_capped_sample"]), 5)
        self.assertIs(report["gpu_launched"], False)

    def test_note_states_no_new_rollouts_and_short_prefixes(self):
        text = (HERE / "note.md").read_text(encoding="utf-8")
        self.assertIn("new rollouts", text)
        self.assertIn("214", text)
        self.assertIn("8192", text)
        self.assertIn("tx01", text)
        self.assertIn("Reserved eight", text)


if __name__ == "__main__":
    unittest.main()
