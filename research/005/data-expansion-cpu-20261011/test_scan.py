"""Offline checks for the 20261011 CPU independent-issue scan. No 31B, no GPU."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V2 = ROOT / "research/005/length-census-v2-20261010"
PREV = ROOT / "research/005/data-expansion-cpu-20261010"
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
V2_SHA256 = {
    "length_report.json":
        "2a219102b6779f5ea9aa37e696c1089e67cc0283b77c5fe23f73720c246eef76",
    "sample_list.json":
        "05cbf49bcd7b67cf8fc8d98937a35031d6192b183a3ac2dcdc8adcd138ee79c1",
    "train_issues.json":
        "b25fe2c97599d4d15d5e0c32c69b1ef6af4a7f6a95a662109b446e865a9159b1",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScanLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("p005_scan_20261011", HERE / "scan.py")
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

    def test_prefixes_of_the_same_v2_issues_are_not_independent(self):
        frozen = {"prepare": {"tasks": [
            "fastapi_11194", "fastapi_14873", "fastapi_5077", "fastapi_9425",
            "rich_2725", "rich_3518", "rich_3777", "rich_3942", "rich_4075",
        ]}}
        prefixes = [{"issue": tid} for tid in frozen["prepare"]["tasks"] for _ in range(3)]
        self.assertEqual(self.scan.new_issue_ids(prefixes, frozen), [])

    def test_reserved_ids_cannot_become_independent_issues(self):
        frozen = {"prepare": {"tasks": []}}
        rows = [{"issue": tid} for tid in RESERVED] + [{"issue": "fresh_ok"}]
        self.assertEqual(self.scan.new_issue_ids(rows, frozen), ["fresh_ok"])

    def test_file_is_new_only_if_sha256_absent_from_previous_inventory(self):
        prev = {"files": [{"sha256": "a" * 64, "path": "results/old.json"}]}
        files = [
            {"task_id": "old", "sha256": "a" * 64, "path": "results/old.json", "mtime_ns": 1},
            {"task_id": "old", "sha256": "b" * 64, "path": "results/copy.json", "mtime_ns": 2},
        ]
        hashes, source = self.scan.apply_previous_inventory(files, prev)
        self.assertEqual(source, "stored_files_sha256")
        self.assertEqual(hashes, {"a" * 64})
        self.assertTrue(files[0]["in_previous_inventory"])
        self.assertFalse(files[1]["in_previous_inventory"])

    def test_counts_match_reconstructs_previous_hashes_without_inventing_files(self):
        digest = "c" * 64
        prev = {
            "n_trace_files": 1,
            "unique_issues_with_traces": ["fastapi_11194"],
            "issues": [{"task_id": "fastapi_11194", "n_traces": 1}],
            "files": [],
        }
        files = [{"task_id": "fastapi_11194", "sha256": digest,
                  "path": "results/a/trace_fastapi_11194.json", "mtime_ns": 1}]
        hashes, source = self.scan.apply_previous_inventory(files, prev)
        self.assertEqual(source, "reconstructed_counts_match")
        self.assertEqual(hashes, {digest})
        self.assertTrue(files[0]["in_previous_inventory"])

    def test_extra_sha256_is_new_when_issue_trace_count_grows(self):
        prev = {
            "n_trace_files": 1,
            "unique_issues_with_traces": ["fastapi_11194"],
            "issues": [{"task_id": "fastapi_11194", "n_traces": 1}],
            "files": [],
        }
        files = [
            {"task_id": "fastapi_11194", "sha256": "a" * 64,
             "path": "results/old/trace_fastapi_11194.json", "mtime_ns": 1},
            {"task_id": "fastapi_11194", "sha256": "b" * 64,
             "path": "results/new/trace_fastapi_11194.json", "mtime_ns": 9},
        ]
        hashes, source = self.scan.apply_previous_inventory(files, prev)
        self.assertEqual(source, "reconstructed_oldest_n_per_issue")
        self.assertEqual(hashes, {"a" * 64})
        self.assertTrue(files[0]["in_previous_inventory"])
        self.assertFalse(files[1]["in_previous_inventory"])

    def test_extra_file_for_known_issue_is_prefix_not_independent(self):
        counted = {"fastapi_11194"}
        historical = {"fastapi_11194"}
        row = {
            "task_id": "fastapi_11194", "in_previous_inventory": False,
            "would_prepare": True, "pairing_reserved": False, "reserved_repo_commit": False,
        }
        extra = self.scan.classify_new_file(row, counted, historical)
        self.assertTrue(extra["new_file"])
        self.assertTrue(extra["prefix_of_already_counted_issue"])
        self.assertFalse(extra["independent_issue_candidate"])
        self.assertIn("already-counted", extra["note"])


class ScanReportTests(unittest.TestCase):
    def test_v2_census_bytes_match_frozen_sha256(self):
        for name, expected in V2_SHA256.items():
            path = V2 / name
            self.assertEqual(sha256_file(path), expected, name)

    def test_v2_census_not_overwritten_in_report(self):
        report = load(HERE / "window_report.json")
        self.assertIs(report["did_not_overwrite_v2_census"], True)
        by_path = {row["path"]: row["sha256"] for row in report["v2_census_sha256_unchanged"]}
        for name, expected in V2_SHA256.items():
            rel = "research/005/length-census-v2-20261010/" + name
            self.assertEqual(by_path[rel], expected, rel)
            self.assertEqual(by_path[rel], sha256_file(V2 / name), rel)

    def test_20261010_reports_not_overwritten(self):
        prev = load(PREV / "trace_inventory.json")
        self.assertEqual(prev["n_trace_files"], 92)
        self.assertEqual(len(prev["eligible_issues"]), 9)
        self.assertNotIn("files", prev)
        self.assertTrue((PREV / "window_report.json").is_file())
        self.assertEqual(
            load(PREV / "window_report.json")["schema"],
            "p005-data-expansion-cpu-20261010")

    def test_no_new_independent_issues_or_gpu(self):
        report = load(HERE / "window_report.json")
        growth = report["independent_issue_growth"]
        nine = [
            "fastapi_11194", "fastapi_14873", "fastapi_5077", "fastapi_9425",
            "rich_2725", "rich_3518", "rich_3777", "rich_3942", "rich_4075",
        ]
        self.assertEqual(growth["new_eligible_issues_with_traces"], [])
        self.assertEqual(growth["v2_prepare_issues"], nine)
        self.assertEqual(growth["inventory_would_prepare_issues"], nine)
        self.assertEqual(growth["current_measured_issues"], [])
        self.assertIs(growth["corpus_cannot_grow_without_new_rollouts"], True)
        self.assertIs(report["gpu_launched"], False)
        self.assertIs(report["did_not_launch_new_rollouts"], True)
        self.assertIs(report["did_not_invent_issues"], True)
        self.assertIs(report["did_not_truncate"], True)
        self.assertEqual(report["protocol_id"], "p005-harness-tools-v1")
        self.assertEqual(report["reserved_in_measured"], [])
        self.assertTrue(set(report["reserved_pairing_issues"]) == set(RESERVED))

    def test_did_not_reprepare_old_prefixes_when_no_new_traces(self):
        report = load(HERE / "window_report.json")
        self.assertEqual(report["n_new_files"], 0)
        self.assertEqual(report["new_files"], [])
        self.assertEqual(report["new_eligible_traces"], [])
        self.assertIs(report["did_not_reprepare_old_prefixes"], True)
        self.assertNotIn("all_complete_prefixes", report)
        self.assertEqual(report["new_usable_summary"]["n"], 0)
        self.assertEqual(report["new_recovery_event_count"], 0)

    def test_inventory_records_sha256_and_reserved_flags(self):
        inv = load(HERE / "trace_inventory.json")
        self.assertEqual(inv["n_trace_files"], 92)
        self.assertEqual(len(inv["files"]), 92)
        self.assertEqual(len(inv["eligible_issues"]), 9)
        self.assertEqual(inv["n_new_files"], 0)
        sha256s = {row["sha256"] for row in inv["files"]}
        self.assertEqual(len(sha256s), 92)
        for row in inv["files"]:
            self.assertEqual(len(row["sha256"]), 64)
            self.assertTrue(row["path"].startswith("results/"))
            self.assertIn("pairing_reserved", row)
            self.assertIn("reserved_repo_commit", row)
            self.assertIn("in_previous_inventory", row)
            self.assertTrue(row["in_previous_inventory"])
            if row["task_id"] in RESERVED:
                self.assertTrue(row["pairing_reserved"])
                self.assertFalse(row["would_prepare"])

    def test_eligible_issues_without_unused_protocol_compatible_trace(self):
        report = load(HERE / "window_report.json")
        expected = [
            "fastapi_11194", "fastapi_14873", "fastapi_5077", "fastapi_9425",
            "rich_2725", "rich_3518", "rich_3777", "rich_3942", "rich_4075",
        ]
        self.assertEqual(
            report["eligible_non_reserved_issues_without_unused_protocol_compatible_trace"],
            expected)
        self.assertTrue(set(expected).isdisjoint(RESERVED))
        gap = report["non_reserved_split_issues_with_no_protocol_compatible_trace"]
        self.assertGreater(len(gap), 0)
        self.assertTrue(all(row["issue"] not in RESERVED for row in gap))
        self.assertTrue(all(row["issue"] not in expected for row in gap))

    def test_note_states_no_new_rollouts_without_reprinting_214_table(self):
        text = (HERE / "note.md").read_text(encoding="utf-8")
        self.assertIn("new rollouts", text)
        self.assertIn("Reserved eight", text)
        self.assertIn("fastapi_11194", text)
        self.assertIn("p005-harness-tools-v1", text)
        self.assertNotIn("| window |", text)
        self.assertNotIn("p0=5235", text)


if __name__ == "__main__":
    unittest.main()
