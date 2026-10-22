#!/usr/bin/env python3
"""Build the standalone P005 single-state Kaggle carrier without importing ML code.

The complete service/evaluation/archive loop is inherited from the existing P002
v30 carrier. P002 files are read-only inputs. Candidate files are embedded afresh
on every build; no Kaggle runtime checkout or external skill path is required.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "notebooks/002/002-joint-explore-v30-single-cell.py"
CANDIDATE = ROOT / "research/005/candidates/single_state/agent"
OUTPUT_PY = ROOT / "notebooks/005/005-single-state-single-cell.py"
OUTPUT_NB = ROOT / "notebooks/005/005-single-state.ipynb"
MODEL_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
MODEL_TOKENIZER_BYTES = 32169626
EXPOSED_TASKS = (
    "rich_3942", "rich_3130", "fastapi_14873", "fastapi_5077",
    "requests_6644", "rich_3518",
)
EVAL_TEXT = """# P005 budgets; timeout applies to individual commands and verification.
evaluation:
  timeout_seconds: 60
  max_tool_calls: 50
  max_time_minutes: 5
  max_turns: 80
"""


def literal_assignment(source, name):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing source assignment: " + name)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("P002 runtime anchor changed: " + old[:90])
    return source.replace(old, new, 1)


def collect_bundle():
    files = {}
    for path in sorted(CANDIDATE.rglob("*")):
        if path.is_symlink():
            raise ValueError("Candidate must contain real files: " + str(path))
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(CANDIDATE).as_posix()
        if rel == "README.md" or path.suffix not in {".py", ".md", ".yaml", ".yml", ".json", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
        files[rel] = text
    if "eval_config.yaml" not in files:
        files["eval_config.yaml"] = EVAL_TEXT
    required = {"agent.yaml", "prompts/system.md", "configs/sampling.yaml",
                "skills/source-edit/SKILL.md", "skills/working-state/SKILL.md",
                "skills/source-probe/SKILL.md"}
    if not required <= files.keys():
        raise ValueError("Incomplete candidate: " + repr(required - files.keys()))
    return {"single_state": files}


RUNTIME_CHECK = '''
def require_supported_runtime():
    # v30 includes cp312-only native wheels. Fail before installing ANY package.
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            'P005 requires a fresh Linux CPython 3.12 kernel for wheelhouse v30; '
            f'current={sys.version.split()[0]} executable={sys.executable}. '
            'Select a Python 3.12 Kaggle image/kernel, then rerun. '
            'No packages have been installed. Do not force cp312 wheels into 3.13.')
    if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('P005 wheelhouse v30 requires Linux x86_64; no packages installed.')


require_supported_runtime()
'''

SCHEDULE = '''
# Start with one already-exposed task. Set RUN_MODE='panel' for all six, or edit
# SELECTED_TASK_IDS explicitly (a non-empty tuple drawn from EXPOSED_TASK_IDS).
RUN_MODE = 'smoke'
SELECTED_TASK_IDS = None
REPEAT_ID = 'r1'
SELECTED_VARIANTS = ('single_state',)
RUN_ID = 'run-005-single-state-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
INPUT_ROOT = Path('/kaggle/input')
WORKING_DIR = Path('/kaggle/working')
RUN_ROOT = WORKING_DIR / 'results' / '005' / RUN_ID
AGENT_ROOT = WORKING_DIR / 'agent_exec' / RUN_ID
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
'''

SETUP = '''
if RUN_MODE not in ('smoke', 'panel'):
    raise ValueError("RUN_MODE must be 'smoke' or 'panel'")
selected_tasks = tuple(SELECTED_TASK_IDS) if SELECTED_TASK_IDS is not None else (
    ('rich_3518',) if RUN_MODE == 'smoke' else tuple(EXPOSED_TASK_IDS))
if not selected_tasks or len(selected_tasks) != len(set(selected_tasks)):
    raise ValueError('Select a non-empty task list without duplicate IDs')
if not set(selected_tasks) <= set(EXPOSED_TASK_IDS):
    raise ValueError('This carrier only contains the six already-exposed task IDs; edit the research panel explicitly.')
if not REPEAT_ID or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in REPEAT_ID):
    raise ValueError('REPEAT_ID must be a simple filename component')
DATA_DIR = resolve_input(
    [INPUT_ROOT / 'competitions/gemma-4-developer-agent', INPUT_ROOT / 'gemma-4-developer-agent'],
    'tasks.jsonl')
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
WHEELHOUSE_DIR = resolve_input(
    [INPUT_ROOT / 'datasets/metric/gemma-4-developer-agent-wheelhouse',
     INPUT_ROOT / 'gemma-4-developer-agent-wheelhouse'],
    'swegemma-0.2.11-py3-none-any.whl')
MODEL_PATH = resolve_input(
    [INPUT_ROOT / 'models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
     INPUT_ROOT / 'gemma-4-31b-it-qat-w4a16-ct/2'],
    'tokenizer.json', 'gemma-4-31b-it-qat-w4a16-ct')
tokenizer_path = MODEL_PATH / 'tokenizer.json'
if (not tokenizer_path.is_file() or tokenizer_path.stat().st_size != MODEL_PIN['tokenizer_bytes']
        or sha256_file(tokenizer_path) != MODEL_PIN['tokenizer_sha256']):
    raise RuntimeError('Attached model tokenizer differs from official google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
assert not RUN_ROOT.exists() and not AGENT_ROOT.exists(), 'Never overwrite a prior run'
RUN_ROOT.mkdir(parents=True)
AGENT_ROOT.mkdir(parents=True)
FROZEN_DIR = RUN_ROOT / 'frozen_bundle'
FROZEN_DIR.mkdir()
atomic_text(FROZEN_DIR / 'experiment_manifest.json', MANIFEST_TEXT)
atomic_json(FROZEN_DIR / 'embedded_config_sha256.json', EXPECTED_CONFIG_HASHES)
atomic_json(FROZEN_DIR / 'model_pin.json', MODEL_PIN)
for variant, file_map in BUNDLE.items():
    for relative, content in file_map.items():
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        assert digest == EXPECTED_CONFIG_HASHES[variant][relative], f'Config changed: {variant}/{relative}'
        atomic_text(FROZEN_DIR / 'variants' / variant / relative, content)
        atomic_text(AGENT_ROOT / variant / relative, content)
full_schedule = []
for ordinal, task_id in enumerate(selected_tasks, 1):
    attempt_id = f'{ordinal:02d}-{task_id}-single_state'
    full_schedule.append({
        'ordinal': ordinal, 'task_slot': ordinal, 'within_task': 1,
        'task_id': task_id, 'variant': 'single_state', 'attempt_id': attempt_id,
        'selected': True,
        'results_dir': str(RUN_ROOT / 'single_state' / REPEAT_ID / 'attempts' / attempt_id),
        'config_sha256': EXPECTED_CONFIG_HASHES['single_state'],
    })
schedule = full_schedule
state = {e['attempt_id']: {**e, 'status': 'planned'} for e in schedule}
atomic_json(RUN_ROOT / 'full_schedule.json', full_schedule)
atomic_json(RUN_ROOT / 'selected_schedule.json', schedule)
atomic_json(RUN_ROOT / 'attempt_status.json', state)
print(f'RUN_ID={RUN_ID} MODE={RUN_MODE} ATTEMPTS={len(schedule)}', flush=True)
'''

WHEEL_CHECK = '''
# Validate CPython/platform tags before installation; wheelhouse v30 is NOT a
# universal Python environment. Packaging normally ships with the Kaggle image.
try:
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
except ImportError:
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import parse_wheel_filename
supported_tags = set(sys_tags())
expected_wheels = WHEELHOUSE_V30_FILES
actual_wheels = {p.name: p.stat().st_size for p in source_wheels}
if actual_wheels != expected_wheels:
    raise RuntimeError('Attached wheelhouse is not the recorded v30 inventory: '
                       f'missing={sorted(set(expected_wheels)-set(actual_wheels))} '
                       f'extra={sorted(set(actual_wheels)-set(expected_wheels))} '
                       f'size_changed={[n for n in actual_wheels.keys() & expected_wheels.keys() if actual_wheels[n] != expected_wheels[n]]}')
incompatible = []
for path in source_wheels:
    fixed_name = path.name.replace('cu128', '+cu128') if 'cu128' in path.name and '+' not in path.name else path.name
    if not parse_wheel_filename(fixed_name)[3] & supported_tags:
        incompatible.append(path.name)
if incompatible:
    raise RuntimeError(f'Incompatible wheels for this Python/platform; no install attempted: {incompatible}')
'''

CANDIDATE_CHECK = '''
common = EXPERIMENT['common']
common_sampling = yaml.safe_load(BUNDLE['single_state']['configs/sampling.yaml'])
budgets = yaml.safe_load(BUNDLE['single_state']['eval_config.yaml'])['evaluation']
config = yaml.safe_load(BUNDLE['single_state']['agent.yaml'])
assert config['tools'] == common['tool_list'] and 'adapter' not in config
assert config['skills'] == ['skills/source-edit', 'skills/working-state', 'skills/source-probe']
assert budgets == common['eval_budgets']
assert common_sampling == common['sampling']
assert validate_single_declared_model(AGENT_ROOT / 'single_state') == TARGET_MODEL_NAME
for path in (AGENT_ROOT / 'single_state').rglob('*'):
    assert not path.is_symlink(), f'No symlinks in submission: {path}'
'''


def render():
    baseline = BASE.read_text(encoding="utf-8")
    original = json.loads(literal_assignment(baseline, "MANIFEST_TEXT"))
    bundle = collect_bundle()
    hashes = {v: {p: hashlib.sha256(t.encode("utf-8")).hexdigest()
                  for p, t in files.items()} for v, files in bundle.items()}
    manifest = {
        "study": "P005 single_state inference candidate; smoke or six exposed tasks; no measured result at build time",
        "common": original["common"], "data": original["data"],
        "exposed_task_ids": EXPOSED_TASKS,
        "research_condition": "Whole single_state package: source editing, hypotheses/state, and decision-oriented probes. Not an isolated skill effect.",
        "source_context": "Official v30 sandbox executor; no custom host callbacks. Candidate scripts resolve their sandbox workspace.",
        "runtime": {"python": "CPython 3.12", "system": "Linux x86_64", "wheelhouse_version": 30},
        "panel_note": "rich_3130 and requests_6644 retain historical evaluation ambiguity; smoke/panel counts are not independent efficacy claims.",
    }
    manifest["common"].pop("single_variable", None)
    model_pin = {"ref": "google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2",
                 "version": 2, "tokenizer_sha256": MODEL_TOKENIZER_SHA256,
                 "tokenizer_bytes": MODEL_TOKENIZER_BYTES,
                 "source": "Official Kaggle MCP file download, 2026-10-10; model weight shards remain the attached version 2 input."}
    wheel_data = json.loads((ROOT / "research/005/official-20261010/wheel-files.json").read_text())
    wheel_inventory = {d["name"]: int(d["total_bytes"]) for d in wheel_data["dataset_files"]
                       if d["name"].endswith(".whl") and "cutlass" not in d["name"].lower()}
    imports = baseline[baseline.index("from pathlib import Path"):baseline.index("# Change only this list")]
    helpers1 = baseline[baseline.index("def input_snapshot"):baseline.index("BUNDLE =")]
    helpers2 = baseline[baseline.index("def sha256_file"):baseline.index("FROZEN_DIR =")]
    body = baseline[baseline.index("# Install before importing target packages."):]
    # Drop image-wide file deletion. The carrier installs only the official v30 wheels.
    start = body.index("for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth'")
    end = body.index("assert WHEELHOUSE_DIR.is_dir()", start)
    body = body[:start] + body[end:]
    body = replace_once(body, "assert source_wheels, 'Wheelhouse is empty'\n",
                        "assert source_wheels, 'Wheelhouse is empty'\n" + WHEEL_CHECK)
    begin = body.index("common = EXPERIMENT['common']")
    end = body.index("eval_signature = inspect.signature", begin)
    body = body[:begin] + CANDIDATE_CHECK + body[end:]
    body = replace_once(body,
        "'budgets': budgets, 'sampling': common_sampling, 'single_variable': 'skill only (T1 = T0 + skills/repo_navigation)',     'sandbox': 'subprocess',",
        "'budgets': budgets, 'sampling': common_sampling, 'research_condition': EXPERIMENT['research_condition'], 'sandbox': 'subprocess',\n    'run_mode': RUN_MODE, 'selected_task_ids': list(selected_tasks), 'model_pin': MODEL_PIN,")
    body = body.replace("submission-002-explore-{variant}.zip", "submission-005-{variant}.zip")
    # Preserve structured failed attempts as well as successful evaluator returns.
    body = replace_once(body,
        "            atomic_text(attempt_dir / 'exception.txt', traceback.format_exc())\n            print(f'  ERROR",
        "            atomic_text(attempt_dir / 'exception.txt', traceback.format_exc())\n            atomic_json(attempt_dir / 'result.json', {'instance_id': task.instance_id, 'resolved': None, 'status': 'exception', 'error_type': type(exc).__name__, 'error': str(exc)})\n            print(f'  ERROR")
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    constants = ("BUNDLE = json.loads(" + repr(json.dumps(bundle, ensure_ascii=False, sort_keys=True)) + ")\n"
                 + "EXPECTED_CONFIG_HASHES = " + repr(hashes) + "\n"
                 + "MANIFEST_TEXT = " + repr(manifest_text) + "\n"
                 + "EXPERIMENT = json.loads(MANIFEST_TEXT)\n"
                 + "EXPOSED_TASK_IDS = " + repr(EXPOSED_TASKS) + "\n"
                 + "MODEL_PIN = " + repr(model_pin) + "\n"
                 + "WHEELHOUSE_V30_FILES = " + repr(wheel_inventory) + "\n")
    header = ("# Generated by notebooks/builders/build_005_single_state.py; do not edit embedded files.\n"
              "# P005 single_state: complete v30 service/evaluation carrier, not yet GPU-run.\n"
              "# Default: one exposed smoke task. RUN_MODE='panel' selects the six exposed tasks.\n"
              "# Requires Linux CPython 3.12, official wheelhouse v30 and model version 2.\n"
              "# Running this cell loads the GPU model; generation/build checks do not.\n")
    source = header + imports + RUNTIME_CHECK + SCHEDULE + helpers1 + constants + helpers2 + SETUP + body
    ast.parse(source, filename=str(OUTPUT_PY))
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "cells": [
            {"cell_type": "markdown", "id": "p005-intro", "metadata": {},
             "source": ["# P005 single_state — v30 complete carrier\n",
                        "Implementation is ready for a GPU smoke run; no model result is claimed.\n",
                        "Use a **Linux Python 3.12** kernel and attach competition data, wheelhouse **v30**, and official CT model **version 2**.\n",
                        "The code cell defaults to one already-exposed task. Set `RUN_MODE = 'panel'` for all six.\n",
                        "This is one candidate package, not an isolated efficacy comparison. Run artifacts and the submission ZIP are written under `/kaggle/working/results/005/`.\n"]},
            {"cell_type": "code", "id": "p005-run", "metadata": {}, "execution_count": None,
             "outputs": [], "source": source.splitlines(keepends=True)},
        ],
    }
    return source, json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare generated files with current sources; never run model code")
    args = parser.parse_args()
    py, nb, hashes = render()
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, text in ((OUTPUT_PY, py), (OUTPUT_NB, nb))
                 if not p.is_file() or p.read_text(encoding="utf-8") != text]
        if stale:
            raise SystemExit("Generated files are stale; run build_005_single_state.py: " + ", ".join(stale))
    else:
        OUTPUT_PY.write_text(py, encoding="utf-8")
        OUTPUT_NB.write_text(nb, encoding="utf-8")
    print(json.dumps({"mode": "check" if args.check else "build", "files": len(hashes["single_state"]),
                      "carrier_sha256": hashlib.sha256(py.encode()).hexdigest(),
                      "python": str(OUTPUT_PY.relative_to(ROOT)), "notebook": str(OUTPUT_NB.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
