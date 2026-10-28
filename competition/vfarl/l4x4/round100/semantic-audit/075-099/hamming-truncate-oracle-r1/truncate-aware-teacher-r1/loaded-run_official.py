"""Record reviewed, off-policy demonstrations using the unmodified official tools.

This runner executes a human-reviewed action recipe. It performs no model inference
and creates neither on-policy rollouts nor counterfactual preference pairs.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from adk_submission import ModelRegistry
from google.adk.tools import FunctionTool
from swegemma.budget import EvaluationBudget, HarnessLimits
from swegemma.config import EvalConfig
from swegemma.context import SwegemmaContext
from swegemma.harness.agent_runner import build_agent_prompt
from swegemma.harness.container_setup import (
    setup_baseline_commit,
    setup_git_exclude,
    setup_workspace_test_config,
)
from swegemma.harness.verification import (
    _is_protected_test_or_config_path,
    verify_task,
)
from swegemma.models import Task
from swegemma.sandbox import SubprocessManager
from swegemma.tools import create_tools
if __name__ == '__main__' and not __package__:
    # Preserve the existing direct-file CLI when PYTHONPATH contains only the
    # official host packages rather than this repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from competition.vfarl.l4x4.bytecode import LEGACY, POLICIES, configure_manager

SIX_TOOLS = {"run_command", "read_file", "write_file", "edit_file", "get_status", "submit_patch"}
EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".DS_Store"}


@dataclasses.dataclass(frozen=True)
class ExternalTrainTask(Task):
    """Explicit task-data adaptation for external required-node specifications.

    The competition derives targets from test_patch. SWE-smith also requires
    unchanged upstream files through PASS_TO_PASS, which must not be omitted.
    """

    verification_target_files: tuple[str, ...] = ()

    @property
    def test_files(self) -> list[str]:
        return list(self.verification_target_files)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ExplicitPromptSetting(argparse.Action):
    """Record CLI settings that cannot override a selection-bound prompt."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        supplied = list(getattr(namespace, "_explicit_prompt_settings", ()))
        supplied.append(option_string)
        namespace._explicit_prompt_settings = supplied


def selection_prompt_profile(selection_path: Path, task_path: Path):
    """Load only hash-bound historical budget/harness for this current task.

    Historical outcomes are not evidence for the selected problem version. The
    caller must check the unmodified official builder's resulting prompt before
    creating a sandbox; numeric JSON types are deliberately not coerced.
    """
    selection_path, task_path = Path(selection_path).resolve(), Path(task_path).resolve()
    selection_bytes, task_bytes = selection_path.read_bytes(), task_path.read_bytes()
    selection, payload = json.loads(selection_bytes), json.loads(task_bytes)
    entries = selection.get("tasks", ())
    if (not isinstance(entries, list) or len(entries) != 100
            or len({entry["instance_id"] for entry in entries}) != 100
            or any(entry.get("physical_split") != "train" for entry in entries)):
        raise ValueError("Prompt selection requires the explicit fixed100 training inventory")
    matches = [entry for entry in entries if entry["instance_id"] == payload["instance_id"]]
    if len(matches) != 1 or matches[0]["task_input_sha256"] != sha256_bytes(task_bytes):
        raise ValueError("Prompt selection does not match the actual task instance and input SHA")
    entry = matches[0]
    report_path = Path(entry["actual_replay_result"])
    if not report_path.is_absolute():
        report_path = Path(__file__).resolve().parents[3] / report_path
    report_path = report_path.resolve()
    report_bytes = report_path.read_bytes()
    if sha256_bytes(report_bytes) != entry["actual_replay_result_sha256"]:
        raise ValueError("Prompt selection source replay report SHA mismatch")
    report = json.loads(report_bytes)
    if report.get("instance_id") != entry["instance_id"]:
        raise ValueError("Prompt selection source replay instance mismatch")
    budget = EvaluationBudget(**report["evaluation_budget"])
    harness = HarnessLimits(**report["harness_limits"])
    profile = {
        "format": "selection-bound-official-prompt-profile-v1",
        "selection_path": str(selection_path),
        "selection_sha256": sha256_bytes(selection_bytes),
        "instance_id": entry["instance_id"],
        "task_input_sha256": sha256_bytes(task_bytes),
        "source_replay_result_path": str(report_path),
        "source_replay_result_sha256": sha256_bytes(report_bytes),
        "expected_prompt_sha256": entry["official_initial_user_prompt_sha256"],
        "evaluation_budget": dataclasses.asdict(budget),
        "harness_limits": dataclasses.asdict(harness),
        "scope": "Historical report supplies prompt configuration only; its outcomes are not reassigned to the current task version",
    }
    return budget, harness, profile


def bind_selection_prompt(profile: dict, actual_prompt: str) -> dict:
    """Require actual official prompt bytes, not a reconstructed substitute."""
    actual_sha256 = sha256_bytes(actual_prompt.encode())
    if actual_sha256 != profile["expected_prompt_sha256"]:
        raise ValueError("Official prompt SHA differs from prompt selection: expected "
                         + profile["expected_prompt_sha256"] + ", actual " + actual_sha256)
    return {**profile, "actual_prompt_sha256": actual_sha256}


def parent_runtime_evidence() -> dict:
    selected, distributions = {}, {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        # metadata.distributions follows the import search path. A later global
        # distribution must not overwrite the package selected from the venv.
        selected.setdefault(name, dist.version)
        distributions.setdefault(name, []).append({"version": dist.version,
                                                   "root": str(dist.locate_file(""))})
    bindings = {}
    for name in ("numpy", "pydantic"):
        try:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            bindings[name] = {"version": module.__version__, "file": str(path),
                              "file_sha256": sha256_bytes(path.read_bytes())}
        except Exception as error:
            bindings[name] = {"import_error": str(error)}
    return {"runtime_packages": selected, "parent_runtime_import_bindings": bindings,
            "parent_python": {"executable": sys.executable, "prefix": sys.prefix,
                              "version": sys.version},
            "runtime_distribution_collisions": {name: values for name, values in distributions.items()
                                                  if len(values) > 1},
            "runtime_package_selection": "first distribution by import search path; numpy/pydantic additionally imported and bound"}


def source_inventory(root: Path) -> dict:
    entries = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if path.is_symlink():
            target = path.readlink()
            if target.is_absolute() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Snapshot symlink escapes the source workspace: {rel}")
            entries.append({"path": str(rel), "symlink": str(target)})
        elif path.is_file():
            entries.append({"path": str(rel), "sha256": sha256_bytes(path.read_bytes()),
                            "mode": stat.S_IMODE(path.stat().st_mode), "bytes": path.stat().st_size})
    digest = sha256_bytes(json.dumps(entries, sort_keys=True).encode())
    return {"tree_sha256": digest, "files": entries, "file_count": len(entries)}


def prepare_workspace(source: Path, target: Path) -> None:
    shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns(*EXCLUDED_PARTS))


def validate_recipe(recipe: dict) -> None:
    if recipe.get("evidence_kind") != "human_reviewed_off_policy_demonstration":
        raise ValueError("Recipe must explicitly identify its off-policy demonstration provenance")
    if not recipe.get("reviewed_by") or not recipe.get("review_notes"):
        raise ValueError("A concrete recipe review is required before executing source commands")
    if not recipe.get("actions") or recipe["actions"][-1]["tool"] != "submit_patch":
        raise ValueError("This demonstration format requires submit_patch as its final action")
    if sum(action["tool"] == "submit_patch" for action in recipe["actions"]) != 1:
        raise ValueError("A demonstration must have one final submission")
    if "final_response" in recipe and (not isinstance(recipe["final_response"], str) or not recipe["final_response"].strip()):
        raise ValueError("A reviewed final_response must be nonempty human-authored text")
    for action in recipe["actions"]:
        name, arguments = action["tool"], action["arguments"]
        if name not in SIX_TOOLS:
            raise ValueError(f"Outside the six-tool course: {name}")
        if name == "run_command":
            command = arguments["command"]
            if action.get("reviewed_command_sha256") != sha256_bytes(command.encode()):
                raise ValueError("run_command text differs from the manually reviewed command hash")
        if name in {"edit_file", "write_file"}:
            path = arguments["filepath"]
            rel = path.removeprefix("/workspace/")
            if _is_protected_test_or_config_path(rel):
                raise ValueError(f"Demonstrations must not alter protected upstream files: {path}")


async def replay(args: argparse.Namespace) -> None:
    runtime_evidence = parent_runtime_evidence()
    for path_name in ("task", "workspace", "recipe", "output"):
        setattr(args, path_name, getattr(args, path_name).resolve())
    if args.source_layout and (Path(args.source_layout).is_absolute() or ".." in Path(args.source_layout).parts):
        raise ValueError("Source layout must be a relative directory inside each workspace")
    if args.source_preload_module and (not args.source_import_alias_root or not all(part.isidentifier() for part in args.source_preload_module.split("."))):
        raise ValueError("A preload module must be a declared package name used with the import-alias profile")
    if args.task_site_packages:
        args.task_site_packages = args.task_site_packages.resolve()
        if not args.task_site_packages.is_dir():
            raise ValueError("Task dependency profile must be an existing site-packages directory")
    task_python_profile = None
    if args.task_python:
        args.task_python = args.task_python.resolve()
        if not args.task_python.is_file() or not args.task_site_packages:
            raise ValueError("A task interpreter requires an existing binary and explicit task-site-packages")
        task_python_profile = {"path": str(args.task_python), "sha256": sha256_bytes(args.task_python.read_bytes()),
            "version": subprocess.check_output([str(args.task_python), "--version"], text=True).strip()}
    sandbox_executables = {}
    for declaration in args.sandbox_executable:
        name, separator, raw_path = declaration.partition("=")
        binary = Path(raw_path).resolve()
        if not separator or not name or Path(name).name != name or not binary.is_file():
            raise ValueError("Sandbox executable must be NAME=existing_absolute_file")
        if name in {"python", "python3", "pytest"}:
            raise ValueError("Dependency executable mappings cannot replace the interpreter or verifier")
        sandbox_executables[name] = {"path": str(binary), "sha256": sha256_bytes(binary.read_bytes())}
    task_payload = json.loads(args.task.read_text())
    recipe = json.loads(args.recipe.read_text())
    validate_recipe(recipe)
    task_class = ExternalTrainTask if "verification_target_files" in task_payload else Task
    allowed_task_fields = {field.name for field in dataclasses.fields(task_class)}
    task = task_class(**{key: value for key, value in task_payload.items() if key in allowed_task_fields})
    # Gold patches are retained only in the source task definition, never in the visible prompt.
    if task.repo.lower() in {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}:
        raise ValueError("The four competition repositories are excluded from this training course")
    prompt_profile = None
    if getattr(args, "prompt_selection", None):
        if getattr(args, "_explicit_prompt_settings", ()):
            raise ValueError("--prompt-selection cannot be combined with explicit budget/harness settings: "
                             + ", ".join(args._explicit_prompt_settings))
        budget, harness, prompt_profile = selection_prompt_profile(args.prompt_selection, args.task)
        args.time_minutes, args.tool_calls = budget.time_minutes, budget.tool_calls
        args.command_timeout = harness.command_timeout_seconds
        args.verification_timeout = harness.verification_timeout_seconds
        config = EvalConfig(tasks_path=args.task, snapshots_dir=args.output,
                            results_dir=args.output, submission_dir=args.output,
                            models=ModelRegistry(), sandbox="subprocess", budget=budget,
                            harness=harness, graph_dir="", embeddings_dir="",
                            enable_sandbox_testing=True)
        actual_prompt = build_agent_prompt(task, config, declared_tools=sorted(SIX_TOOLS),
                                           enable_sandbox_testing=True)
        prompt_profile = bind_selection_prompt(prompt_profile, actual_prompt)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing replay: {args.output}")
    args.output.mkdir(parents=True)
    inventory = source_inventory(args.workspace)
    write_json(args.output / "source-inventory.json", inventory)
    write_json(args.output / "recipe.json", recipe)
    if prompt_profile is None:
        budget = EvaluationBudget(time_minutes=args.time_minutes, tool_calls=args.tool_calls, turns=500)
        harness = HarnessLimits(command_timeout_seconds=args.command_timeout,
                                verification_timeout_seconds=args.verification_timeout)
        config = EvalConfig(tasks_path=args.task, snapshots_dir=args.output,
                            results_dir=args.output, submission_dir=args.output,
                            models=ModelRegistry(), sandbox="subprocess", budget=budget,
                            harness=harness, graph_dir="", embeddings_dir="",
                            enable_sandbox_testing=True)
        actual_prompt = build_agent_prompt(task, config, declared_tools=sorted(SIX_TOOLS),
                                           enable_sandbox_testing=True)
    (args.output / "agent-prompt.txt").write_text(actual_prompt, encoding="utf-8")
    manager = SubprocessManager(base_dir=args.output / "sandboxes", system_site_packages=True)
    bytecode_policy = configure_manager(manager, getattr(args, 'source_bytecode_policy', LEGACY))
    original_start = manager.start
    original_stop = manager.stop
    verification_phase = {"label": "phase-a"}
    import_aliases = {}
    import_alias_metadata = {}
    source_layout_bindings = {}
    task_dependency_bindings = {}
    task_runtime_packages = ({dist.metadata["Name"]: dist.version
                              for dist in importlib.metadata.distributions(path=[str(args.task_site_packages)])}
                             if args.task_site_packages else None)
    if args.source_import_alias_root:
        args.source_import_alias_root = args.source_import_alias_root.resolve()
        args.source_import_alias_root.mkdir(parents=True, exist_ok=False)

    def start_with_source_layout() -> str:
        new_id = original_start()
        if args.task_python:
            # The official host stays on its supported interpreter. Repository
            # commands get a clean task-specific interpreter and dependency path;
            # do not carry compiled host-version packages into the new venv.
            subprocess.run([str(args.task_python), "-m", "venv", "--clear", "--without-pip",
                            str(manager.sandboxes[new_id]["venv"])], check=True, capture_output=True)
        for name, binary in sandbox_executables.items():
            destination = manager.sandboxes[new_id]["venv"] / "bin" / name
            if destination.exists():
                raise FileExistsError(f"Refusing to replace a sandbox executable: {destination}")
            destination.symlink_to(binary["path"])
        if args.task_site_packages:
            dependency_paths = manager.sandboxes[new_id]
            entries = []
            for site_packages in dependency_paths["venv"].glob("lib/python*/site-packages"):
                # The official ADK host and repository commands can legitimately
                # require different dependency versions, as in the Docker backend.
                # Only the sandbox's Python search path is affected by this profile.
                if args.task_site_packages_mode == "copy":
                    # Some repositories classify installed library code by the
                    # interpreter's real sysconfig purelib. Use a normal private
                    # package location instead of altering sysconfig or tests.
                    shutil.copytree(args.task_site_packages, site_packages, dirs_exist_ok=True,
                                    symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    entries.append({"mode": "copy", "source": str(args.task_site_packages),
                                    "target": str(site_packages), "bytecode_cache_copied": False})
                else:
                    entry = f"import sys; sys.path.insert(0, {str(args.task_site_packages)!r})\n"
                    profile = site_packages / "_vfarl_task_dependencies.pth"
                    profile.write_text(entry)
                    entries.append({"mode": "pth", "path": str(profile), "content": entry})
            task_dependency_bindings[new_id] = entries
        if args.source_layout or args.source_import_alias_root:
            paths = manager.sandboxes[new_id]
            source_root = paths["workspace"]
            if args.source_import_alias_root:
                source_root = args.source_import_alias_root / f"a{len(import_aliases)}"
                source_root.symlink_to(paths["workspace"], target_is_directory=True)
                import_aliases[new_id] = source_root
                import_alias_metadata[new_id] = {"alias": str(source_root),
                                                  "target": str(paths["workspace"]),
                                                  "input_tree_sha256": inventory["tree_sha256"],
                                                  "pth_entries": []}
            source_path = source_root / args.source_layout
            source_layout_bindings[new_id] = {
                "workspace": str(paths["workspace"]),
                "source_path": str(source_path),
                "precedence": "sys_path_insert_0",
                "pth_entries": [],
            }
            for site_packages in paths["venv"].glob("lib/python*/site-packages"):
                if args.source_import_alias_root:
                    # An explicit local environment profile for upstream suites
                    # whose path handling assumes a short Docker workspace path.
                    entry = f"import sys; sys.path.insert(0, {str(source_path)!r})\n"
                    if args.source_preload_module:
                        module_name = args.source_preload_module
                        binding_path = paths["root"] / "_vfarl_import_binding.json"
                        entry = entry.rstrip("\n") + f"; import {module_name}; import json; " + \
                            f"json.dump({{'module': {module_name!r}, 'file': {module_name}.__file__, 'package_path': list(getattr({module_name}, '__path__', ())) }}, open({str(binding_path)!r}, 'w'))\n"
                else:
                    # A plain .pth path is appended after system-site packages.
                    # The task's own src tree must take precedence over an
                    # installed distribution sharing the same package name.
                    entry = f"import sys; sys.path.insert(0, {str(source_path)!r})\n"
                (site_packages / "_vfarl_workspace_layout.pth").write_text(entry)
                source_layout_bindings[new_id]["pth_entries"].append({
                    "path": str(site_packages / "_vfarl_workspace_layout.pth"),
                    "content": entry,
                })
                if args.source_import_alias_root:
                    import_alias_metadata[new_id]["pth_entries"].append({"path": str(site_packages / "_vfarl_workspace_layout.pth"), "content": entry})
        return new_id

    def preserve_reports_then_stop(stopping_id: str) -> None:
        paths = manager.sandboxes.get(stopping_id, {})
        report_label = "phase-a" if stopping_id == sandbox_id else verification_phase["label"]
        for report in paths.get("tmp", Path("/nonexistent")).glob("_swegemma_junit_*.xml"):
            destination = args.output / f"{report_label}-{report.name}"
            shutil.copy2(report, destination)
        if stopping_id in source_layout_bindings:
            write_json(args.output / f"{report_label}-source-layout.json",
                       source_layout_bindings[stopping_id])
        if stopping_id in import_aliases:
            metadata = dict(import_alias_metadata[stopping_id])
            metadata["alias_resolves_to_workspace"] = import_aliases[stopping_id].resolve() == paths["workspace"].resolve()
            binding_path = paths["root"] / "_vfarl_import_binding.json"
            if binding_path.exists():
                binding = json.loads(binding_path.read_text())
                imported_file = Path(binding["file"])
                binding["actual_init_sha256"] = sha256_bytes(imported_file.read_bytes())
                binding["actual_init_resolved_path"] = str(imported_file.resolve())
                metadata["actual_preloaded_package"] = binding
            try:
                metadata["current_workspace_tree_sha256"] = source_inventory(paths["workspace"])["tree_sha256"]
            except Exception as error:
                metadata["inventory_capture_error"] = str(error)
            write_json(args.output / f"{report_label}-import-alias.json", metadata)
        original_stop(stopping_id)
        if stopping_id in import_aliases:
            import_aliases[stopping_id].unlink(missing_ok=True)

    # Preserve reports before the original stop deletes temporary files. The
    # optional start hook also adds the declared src layout to the local venv.
    # Official translation, tools, context and verification functions stay intact;
    # the separately declared bytecode profile can prepend task-shell exports.
    manager.start = start_with_source_layout
    manager.stop = preserve_reports_then_stop
    started = time.perf_counter()
    sandbox_id = manager.start()
    workspace = manager.sandboxes[sandbox_id]["workspace"]
    ctx = None
    observations = []
    messages = [{"role": "user", "content": actual_prompt}]
    try:
        prepare_workspace(args.workspace, workspace)
        setup = manager.exec(sandbox_id, "git init -q && git config user.name 'VFARL CPU replay' && git config user.email 'vfarl-local@example.invalid' && git config gc.auto 0 && git config gc.autoDetach false")
        if setup.exit_code:
            raise RuntimeError(setup.stderr or setup.stdout)
        setup_git_exclude(manager, sandbox_id)
        setup_workspace_test_config(manager, sandbox_id, repo=task.repo)
        setup_baseline_commit(manager, sandbox_id, "official_tool_replay_baseline")
        baseline = manager.exec(sandbox_id, "git rev-parse _swegemma_baseline")
        if baseline.exit_code:
            raise RuntimeError("Official _swegemma_baseline was not created")
        # A fresh, history-free repository is used by both phases; future upstream
        # Git history from a download/cache is never copied to the agent workspace.
        snapshot = args.output / "source-snapshot.tgz"
        with tarfile.open(snapshot, "w:gz") as archive:
            for path in sorted(workspace.iterdir()):
                archive.add(path, arcname=path.name, recursive=True)
        ctx = SwegemmaContext(docker_manager=manager, container_id=sandbox_id,
                              task=task, budget=budget, harness=harness)
        tools = create_tools(ctx)
        write_json(args.output / "adk-six-tool-declarations.json", [
            FunctionTool(tools[name])._get_declaration().model_dump(exclude_none=True, mode="json")
            for name in sorted(SIX_TOOLS)
        ])
        ctx.start_agent_session()
        phase_a_setup_seconds = time.perf_counter() - started
        with (args.output / "tool-events.jsonl").open("w", encoding="utf-8") as events:
            for index, action in enumerate(recipe["actions"]):
                call_id = f"replay_{index:04d}"
                arguments = action["arguments"]
                before = {"tool_calls_used": ctx.tool_calls_used,
                          "remaining_time_seconds": ctx.remaining_time_seconds}
                call_started = time.perf_counter()
                response = tools[action["tool"]](**arguments)
                elapsed = time.perf_counter() - call_started
                event = {"index": index, "tool_call_id": call_id,
                         "tool": action["tool"], "arguments": arguments,
                         "response": response, "wall_seconds": elapsed,
                         "budget_before": before,
                         "budget_after": {"tool_calls_used": ctx.tool_calls_used,
                                          "remaining_time_seconds": ctx.remaining_time_seconds},
                         "model_output_tokens": None, "old_logprobs": None}
                observations.append(event)
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.flush()
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [{"id": call_id, "type": "function",
                                                 "function": {"name": action["tool"],
                                                              "arguments": json.dumps(arguments)}}]})
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "name": action["tool"], "content": json.dumps(response, ensure_ascii=False)})
                expected_status = action.get("expected_status")
                if expected_status and response.get("status") != expected_status:
                    raise RuntimeError(f"Recipe stopped at action {index}: expected {expected_status}, got {response.get('status')}")
        final_response = recipe.get("final_response")
        if final_response:
            messages.append({"role": "assistant", "content": final_response})
            write_json(args.output / "terminal-response.json", {
                "text": final_response,
                "provenance": "human_authored_reviewed_off_policy_recipe",
                "recorded_after_actual_submit_patch_response": True,
                "model_generated": False,
                "tool_call_added": False,
            })
        ctx.stop_agent_session()
        phase_a_total_seconds = time.perf_counter() - started
        (args.output / "submitted.patch").write_text(ctx.submitted_patch or "", encoding="utf-8")
        write_json(args.output / "messages.json", messages)
        verification = None
        baseline_verification = None
        baseline_verification_wall_seconds = None
        verification_wall_seconds = None
        if args.verify:
            if args.verify_baseline:
                verification_phase["label"] = "baseline"
                baseline_started = time.perf_counter()
                baseline_verification = await verify_task(manager, config, task, snapshot,
                                                          agent_patch="", start_time=baseline_started)
                baseline_verification_wall_seconds = time.perf_counter() - baseline_started
                write_json(args.output / "baseline-official-verification.json", baseline_verification.model_dump(mode="json"))
            verification_phase["label"] = "repaired"
            verification_started = time.perf_counter()
            verification = await verify_task(manager, config, task, snapshot,
                                             agent_patch=ctx.submitted_patch or "", start_time=verification_started)
            verification_wall_seconds = time.perf_counter() - verification_started
            write_json(args.output / "official-verification.json", verification.model_dump(mode="json"))
        imports = {}
        for name, function in {name: tools[name] for name in SIX_TOOLS}.items():
            file_path = Path(inspect.getsourcefile(function))
            imports[name] = {"module": function.__module__, "path": str(file_path),
                             "file_sha256": sha256_bytes(file_path.read_bytes())}
        write_json(args.output / "run-summary.json", {
            "instance_id": task.instance_id, "repo": task.repo, "base_commit_metadata": task.base_commit,
            "source_task_sha256": sha256_bytes(args.task.read_bytes()),
            "source_tree_sha256": inventory["tree_sha256"],
            "baseline_commit": baseline.stdout.strip(),
            "snapshot_sha256": sha256_bytes(snapshot.read_bytes()),
            "evidence_kind": recipe["evidence_kind"], "behavior_policy": None,
            "demonstration_authorship": "Codex agents specified and reviewed the action recipe; not a sampled Gemma rollout or end-user manual annotation. The human_reviewed schema label is legacy terminology.",
            "initial_prompt_source": "swegemma.harness.agent_runner.build_agent_prompt",
            "initial_prompt_sha256": sha256_bytes(actual_prompt.encode()),
            "initial_prompt_declared_tools": sorted(SIX_TOOLS),
            "text_only_terminal_response_recorded": bool(final_response),
            "terminal_response_provenance": "human_authored_reviewed_off_policy_recipe" if final_response else None,
            "same_state_counterfactual": False, "on_policy": False,
            "backend": "official_SubprocessManager_noncontainer",
            "source_bytecode_policy": bytecode_policy,
            "task_data_adapter": task_class.__name__,
            "verification_target_files": task.test_files,
            "sandbox_source_layout_path": args.source_layout,
            "sandbox_source_import_alias_root": str(args.source_import_alias_root) if args.source_import_alias_root else None,
            "sandbox_source_import_aliases": {key: str(value) for key, value in import_aliases.items()},
            "sandbox_source_preload_module": args.source_preload_module,
            "task_site_packages_profile": str(args.task_site_packages) if args.task_site_packages else None,
            "task_site_packages_mode": args.task_site_packages_mode if args.task_site_packages else None,
            "task_runtime_packages": task_runtime_packages,
            "task_python_profile": task_python_profile,
            "task_dependency_bindings": task_dependency_bindings,
            "sandbox_dependency_executables": sandbox_executables,
            "competition_Docker_environment_equivalent": False,
            "six_tools_only": True, "tool_events": len(observations),
            "evaluation_budget": dataclasses.asdict(budget),
            "harness_limits": dataclasses.asdict(harness),
            "tool_calls_charged": ctx.tool_calls_used,
            "tools_called": sorted({event["tool"] for event in observations}),
            "submitted_patch_bytes": len((ctx.submitted_patch or "").encode()),
            "official_verify_task_invoked": args.verify,
            "baseline_official_verify_task_invoked": args.verify and args.verify_baseline,
            "baseline_resolved_by_local_official_verify_task": baseline_verification.resolved if baseline_verification else None,
            "resolved_by_local_official_verify_task": verification.resolved if verification else None,
            "agent_elapsed_seconds": ctx.agent_elapsed_seconds,
            "phase_a_setup_seconds": phase_a_setup_seconds,
            "phase_a_total_seconds_including_setup": phase_a_total_seconds,
            "baseline_verification_wall_seconds": baseline_verification_wall_seconds,
            "phase_b_verification_wall_seconds": verification_wall_seconds,
            "total_wall_seconds": time.perf_counter() - started,
            "tool_imports": imports,
            **({"prompt_selection_profile": prompt_profile} if prompt_profile is not None else {}),
            **runtime_evidence,
            "limitations": ["Human-authored actions, no model inference or policy logprobs",
                            "Local subprocess shares the host OS; not a security boundary",
                            "Source task test-node semantics must be audited separately",
                            "No state restoration or causal preference pair is claimed"],
        })
    except Exception as error:
        write_json(args.output / "messages.json", messages)
        write_json(args.output / "run-error.json", {"error": str(error), "completed_tool_events": len(observations)})
        raise
    finally:
        if ctx and ctx.agent_end_time is None:
            ctx.stop_agent_session()
        manager.cleanup_all()
        if args.source_import_alias_root:
            args.source_import_alias_root.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-baseline", action="store_true")
    parser.add_argument("--prompt-selection", type=Path,
                        help="Opt in to exact fixed100 selection-bound prompt budget/harness and SHA; conflicts with explicit budget/harness flags")
    parser.add_argument("--source-layout", default="", help="Explicit relative package directory, e.g. src; added to each sandbox venv")
    parser.add_argument("--source-import-alias-root", type=Path, help="Explicit new short directory of workspace symlinks for a documented local path compatibility profile")
    parser.add_argument("--source-preload-module", default="", help="Execute the real package initializer through the explicit alias before pytest changes sys.path")
    parser.add_argument("--task-site-packages", type=Path, help="Explicit preinstalled task-only dependency profile for sandbox Python; official ADK host dependencies remain separate")
    parser.add_argument("--task-site-packages-mode", choices=("pth", "copy"), default="pth", help="Use external dependency search path or copy packages into each private sandbox's normal purelib, excluding old bytecode")
    parser.add_argument("--task-python", type=Path, help="Task-only Python binary; rebuilds sandbox venv without host packages, requires --task-site-packages")
    parser.add_argument("--source-bytecode-policy", choices=POLICIES, default=LEGACY,
                        help="Explicit prospective task-Python cache semantics; legacy replay behavior stays default")
    parser.add_argument("--sandbox-executable", action="append", default=[], help="Expose a reviewed installed dependency binary as NAME=ABS_PATH in the sandbox venv/bin; never replaces Python or pytest")
    parser.add_argument("--time-minutes", action=ExplicitPromptSetting, type=float, default=60)
    parser.add_argument("--tool-calls", action=ExplicitPromptSetting, type=int, default=100)
    parser.add_argument("--command-timeout", action=ExplicitPromptSetting, type=int, default=300)
    parser.add_argument("--verification-timeout", action=ExplicitPromptSetting, type=int, default=300)
    asyncio.run(replay(parser.parse_args()))


if __name__ == "__main__":
    main()
