#!/usr/bin/env python3
"""CPU-only checks of the generated P005 carrier, including its real run loop.

No packages are installed. No model or real Evaluator is imported. The exact
generated loop executes against two lightweight fake evaluator outcomes so that
artifact handling, exception continuation and cleanup are exercised together.
"""
import ast
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
import uuid

from build_005_single_state import OUTPUT_PY, OUTPUT_NB, collect_bundle, render


def select_nodes(tree, names):
    return [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]


def compile_nodes(nodes):
    return compile(ast.Module(body=nodes, type_ignores=[]), str(OUTPUT_PY), "exec")


def embedded_json(tree, name):
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return json.loads(ast.literal_eval(node.value.args[0]))


def check_runtime_guard(tree):
    guard = compile_nodes(select_nodes(tree, {"require_supported_runtime"}))
    for version, system, machine, should_pass in (
            ((3, 12), "Linux", "x86_64", True),
            ((3, 13), "Linux", "x86_64", False),
            ((3, 12), "Darwin", "arm64", False)):
        env = {"sys": SimpleNamespace(version_info=version, version=".".join(map(str, version)), executable="mock-python"),
               "platform": SimpleNamespace(system=lambda: system, machine=lambda: machine)}
        exec(guard, env)
        try:
            env["require_supported_runtime"]()
        except RuntimeError:
            assert not should_pass, (version, system)
        else:
            assert should_pass, (version, system)


def check_run_loop(tree, startup_failure=False):
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "server" for t in n.targets))
    loop = compile_nodes(tree.body[start:])
    helpers = compile_nodes(select_nodes(tree, {
        "atomic_text", "atomic_json", "sha256_file", "run_sync", "ArchivingVllmServer"}))
    events = []
    with tempfile.TemporaryDirectory(prefix="p005-carrier-check-") as tmp:
        root = Path(tmp)
        run_root = root / "run"
        run_root.mkdir()
        tasks = {x: SimpleNamespace(instance_id=x, repo="mock/repo") for x in ("success", "failure")}
        schedule = [{"task_id": x, "variant": "single_state", "attempt_id": x,
                     "results_dir": str(run_root / "single_state/r1/attempts" / x)} for x in tasks]
        states = {x["attempt_id"]: {**x, "status": "planned"} for x in schedule}

        class FakeServer:
            def __init__(self, config, adapter_manifest):
                self.log_path = root / "live-server.log"
                self.base_url = "http://mock.invalid/v1"

            def start(self):
                events.append("server-start")
                self.log_path.write_text("mock server startup\n")
                if startup_failure:
                    raise RuntimeError("simulated startup failure")

            def stop(self, timeout=10):
                # ArchivingVllmServer must copy this before the underlying stop.
                assert (run_root / "vllm_server.archived.log").is_file()
                events.append("server-stop")
                self.log_path.unlink(missing_ok=True)

            def create_model_registry(self, **kwargs):
                return {"mock": "not a model"}

        class FakeEvaluator:
            def __init__(self, config):
                self.config = config
                self.docker = SimpleNamespace(sandboxes={}, cleanup_all=lambda: events.append("sandbox-cleanup"))

            async def evaluate_task(self, task, task_index, total_tasks, context):
                events.append("evaluate-" + task.instance_id)
                context.llm_calls_used = 2
                context.submitted_patch = "recovered patch\n"
                if task.instance_id == "failure":
                    raise ValueError("simulated task failure")
                return SimpleNamespace(
                    instance_id=task.instance_id, repo=task.repo, resolved=True,
                    agent_patch="diff --git a/x.py b/x.py\n", test_exit_code=0,
                    duration_seconds=0.1, tool_calls=1, total_llm_calls=2, error=None,
                    trace_json_path=None, trace={"events": ["mock-tool-call"]},
                    test_output="mock target passed\n")

        def eval_config(**kwargs):
            return SimpleNamespace(**kwargs, budget={}, harness={})

        def context(**kwargs):
            return SimpleNamespace(**kwargs, llm_calls_used=0, submitted_patch=None)

        def save_trace(trace, dest, task_id):
            path = dest / "traces" / ("trace_" + task_id.replace("/", "__") + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(trace))

        def append_result(result, dest, results):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "mock_incremental.json").write_text(json.dumps([r.instance_id for r in results]))

        env = dict(globals())
        env.update({
            "RUN_ROOT": run_root, "MODEL_PATH": root / "model", "TARGET_MODEL_NAME": "mock",
            "VllmServer": FakeServer, "VllmConfig": lambda **kw: SimpleNamespace(**kw),
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: False)),
            "tp_size": 4, "adapters": SimpleNamespace(adapters={}),
            "SELECTED_VARIANTS": ("single_state",), "REPEAT_ID": "r1",
            "schedule": schedule, "state": states, "by_id": tasks, "run_manifest": {},
            "build_submission_limits": lambda: ({}, {}), "EvalConfig": eval_config,
            "Evaluator": FakeEvaluator, "SwegemmaContext": context,
            "ContextCacheConfig": lambda **kw: kw, "EventsCompactionConfig": lambda **kw: kw,
            "TASKS_PATH": root / "tasks.jsonl", "DATA_DIR": root / "data",
            "AGENT_ROOT": root / "agent", "budgets": {"timeout_seconds": 60, "max_tool_calls": 50,
                                                          "max_time_minutes": 5, "max_turns": 80},
            "save_trace_artifact": save_trace, "append_task_result": append_result,
        })
        exec(helpers, env)
        try:
            exec(loop, env)
        except RuntimeError as exc:
            assert startup_failure and "simulated startup failure" in str(exc)
        else:
            assert not startup_failure
        assert events.count("server-start") == events.count("server-stop") == 1
        final = json.loads((run_root / "joint_summary.json").read_text())["single_state"]
        if startup_failure:
            assert final["returned_results"] == 0
            assert set(final["states"].values()) == {"not_run"}
            assert (run_root / "batch_exception.txt").is_file()
        else:
            assert final["resolved"] == final["returned_results"] == 1
            assert final["states"] == {"success": "completed", "failure": "error"}
            assert events.count("sandbox-cleanup") == 2
            good = Path(schedule[0]["results_dir"])
            bad = Path(schedule[1]["results_dir"])
            assert json.loads((good / "result.json").read_text())["resolved"] is True
            assert (good / "traces/trace_success.json").is_file()
            assert (good / "patches/success.patch").is_file()
            assert (good / "test_outputs/success.log").is_file()
            assert (good / "artifact_manifest.json").is_file()
            assert json.loads((bad / "result.json").read_text())["resolved"] is None
            assert (bad / "submitted_patch_before_exception.patch").read_text() == "recovered patch\n"
        assert not (root / "live-server.log").exists()


def main():
    py, nb, hashes = render()
    assert OUTPUT_PY.read_text() == py, "rebuild carrier after candidate changes"
    assert OUTPUT_NB.read_text() == nb, "rebuild notebook after candidate changes"
    tree = ast.parse(py)
    bundle = embedded_json(tree, "BUNDLE")
    assert bundle == collect_bundle()
    assert len(bundle["single_state"]) == len(hashes["single_state"])
    for path, text in bundle["single_state"].items():
        assert hashlib.sha256(text.encode()).hexdigest() == hashes["single_state"][path]
    notebook = json.loads(nb)
    assert "".join(notebook["cells"][1]["source"]) == py
    assert notebook["cells"][1]["outputs"] == []
    check_runtime_guard(tree)
    check_run_loop(tree)
    check_run_loop(tree, startup_failure=True)
    print("P005 carrier checks passed: bundle/SHA parity, Python guard, success+error artifacts, startup-failure cleanup. No ML imports or GPU calls.")


if __name__ == "__main__":
    main()
