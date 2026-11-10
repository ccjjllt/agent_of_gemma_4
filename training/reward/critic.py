"""Process evidence from observed runtime records, never model narration.

The critic deliberately has no free-text/model-response input.  Tool events,
edits and test records must be structured records captured by the tool runtime.
Private target paths are used only for aggregate matching and are not returned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Iterable, Mapping


def _records(values: Iterable[Mapping[str, object]] | None) -> list[Mapping[str, object]]:
    if values is None:
        return []
    return [row for row in values if isinstance(row, Mapping)]


def _path(row: Mapping[str, object]) -> str | None:
    value = row.get("path", row.get("file_path"))
    if not isinstance(value, str) or not value:
        return None
    # Runtime paths are POSIX sandbox paths.  Resolve only lexical '.'/'..'
    # enough for suffix comparison; no filesystem access or target disclosure.
    path = PurePosixPath(value)
    parts = [part for part in path.parts if part not in ("/", ".")]
    if ".." in parts:
        return None
    return "/".join(parts)


def _kind(row: Mapping[str, object]) -> str:
    value = row.get("event_type", row.get("type", row.get("kind", row.get("tool", ""))))
    return str(value).lower()


def _targets(private_targets: object) -> tuple[set[str], set[str]]:
    if private_targets is None:
        return set(), set()
    if isinstance(private_targets, Mapping):
        files = private_targets.get("files", private_targets.get("paths", ()))
        functions = private_targets.get("functions", private_targets.get("symbols", ()))
    else:
        files, functions = private_targets, ()
    if isinstance(files, (str, bytes)):
        files = (files,)
    if isinstance(functions, (str, bytes)):
        functions = (functions,)
    normalized_files = {_path({"path": item}) for item in files if isinstance(item, str)}
    normalized_files.discard(None)
    normalized_functions = {item for item in functions if isinstance(item, str) and item}
    return normalized_files, normalized_functions


def _matches(path: str, targets: set[str]) -> bool:
    return any(path == target or path.endswith("/" + target) or target.endswith("/" + path) for target in targets)


def _exit_code(row: Mapping[str, object]) -> object:
    return row.get("exit_code", row.get("returncode"))


class ProcessCritic:
    """Produce auditable process labels without adding reward weights."""

    def evaluate(
        self,
        *,
        tool_events: Iterable[Mapping[str, object]] | None = None,
        edits: Iterable[Mapping[str, object]] | None = None,
        tests: Iterable[Mapping[str, object]] | None = None,
        differential: Mapping[str, object] | None = None,
        private_targets: Mapping[str, object] | Iterable[str] | None = None,
    ) -> dict:
        events = _records(tool_events)
        edit_rows = _records(edits)
        test_rows = _records(tests)
        target_files, target_functions = _targets(private_targets)

        read_kinds = {"read", "read_file", "open", "view", "search", "grep", "glob", "lsp"}
        located_rows = [row for row in events + edit_rows if _kind(row) in read_kinds | {"edit", "write", "patch"}]
        observed_paths = [path for row in located_rows if (path := _path(row)) is not None]
        matched_count = len({target for target in target_files
                             if any(_matches(path, {target}) for path in observed_paths)})
        edited_paths = [path for row in edit_rows if (path := _path(row)) is not None]
        edited_match_count = len({target for target in target_files
                                  if any(_matches(path, {target}) for path in edited_paths)})
        if not target_files:
            localization_status = "unknown"
        elif edited_match_count:
            localization_status = "edited"
        elif matched_count:
            localization_status = "observed"
        else:
            localization_status = "not_observed"

        navigation = [row for row in events if _kind(row) in read_kinds]
        symbol_events = [row for row in navigation if row.get("symbol") or row.get("symbols") or row.get("calls")]
        observed_symbols = {
            symbol for row in symbol_events
            for symbol in ([row.get("symbol")] if isinstance(row.get("symbol"), str) else row.get("symbols", []))
            if isinstance(symbol, str)
        }
        if not target_functions:
            function_status = "observed" if symbol_events else "unknown"
        else:
            function_status = "observed" if observed_symbols & target_functions else "not_observed"

        executed_tests = [row for row in test_rows if isinstance(row.get("command"), (str, list)) and isinstance(_exit_code(row), int)]
        failed_tests = [row for row in executed_tests if _exit_code(row) != 0]
        structured_failures = [row for row in failed_tests if (
            row.get("junit_available") is True or isinstance(row.get("structured_output"), Mapping)
            or isinstance(row.get("failure_counts"), Mapping)
        )]
        test_audit = [{
            "command_sha256": hashlib.sha256(json.dumps(
                row["command"], sort_keys=True, ensure_ascii=False,
            ).encode("utf-8")).hexdigest(),
            "command_kind": "pytest" if "pytest" in str(row["command"]) else "other",
            "exit_code": _exit_code(row),
            "structured_output_observed": bool(
                row.get("junit_available") is True or isinstance(row.get("structured_output"), Mapping)
                or isinstance(row.get("failure_counts"), Mapping)
            ),
        } for row in executed_tests]

        differential = differential if isinstance(differential, Mapping) else {}
        p = differential.get("P")
        g = differential.get("G")
        pair_eligible = differential.get("eligible") is True
        improvement_status = ("observed" if pair_eligible and isinstance(p, (int, float)) and p > 0 else
                              "not_observed" if pair_eligible and p == 0 else "unknown")
        regression_coverage = (differential.get("coverage") or {}).get("regression", {})
        preservation_status = ("observed" if pair_eligible and g == 0 and regression_coverage.get("complete") is True else
                               "regressed" if pair_eligible and isinstance(g, (int, float)) and g > 0 else "unknown")

        return {
            "schema_version": "process_evidence_v1",
            "evidence_source": "structured_runtime_events_and_official_differential_only",
            "target_file_localization": {
                "status": localization_status,
                "private_target_count": len(target_files),
                "observed_target_count": matched_count,
                "edited_target_count": edited_match_count,
                "paths_disclosed": False,
            },
            "related_functions_or_call_chain": {
                "status": function_status,
                "navigation_event_count": len(navigation),
                "structured_symbol_event_count": len(symbol_events),
                "matched_private_target_count": len(observed_symbols & target_functions),
            },
            "failing_test_identification": {
                "status": "observed" if structured_failures else "not_observed" if executed_tests else "unknown",
                "executed_test_command_count": len(executed_tests),
                "failed_command_count": len(failed_tests),
                "structured_failure_count": len(structured_failures),
                "commands": test_audit,
            },
            "post_change_improvement": {
                "status": improvement_status,
                "P": p if pair_eligible else None,
                "fixed_count": (differential.get("counts") or {}).get("fixed") if pair_eligible else None,
            },
            "regression_preservation": {
                "status": preservation_status,
                "G": g if pair_eligible else None,
                "regressed_count": (differential.get("counts") or {}).get("regressed") if pair_eligible else None,
                "coverage_complete": regression_coverage.get("complete") is True,
            },
            "model_self_report_used": False,
        }


build_process_evidence = ProcessCritic().evaluate
