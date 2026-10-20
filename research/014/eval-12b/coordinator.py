"""Run the frozen P014 paired repair experiment on one continuously batched server.

This is an experiment runner, not a test suite. Inputs and answers stay outside
the Agent sandbox; only issue text and the baseline repository enter it.
The launch file supplies SETTINGS, BUNDLE and WORKER_SOURCE as frozen literals.
This 12B derivative adds repair-only P013 adapter routing and the official 12B override.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(data)
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def locate(name):
    found = list(Path('/kaggle/input').rglob(name))
    if len(found) != 1:
        raise RuntimeError(f'Expected one {name}, found {found[:8]}')
    return found[0]


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=4)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


"""12B routing component for the shared P014 inference carrier.

No model is loaded and no task is run on import. The carrier embeds this source,
the frozen agent text files and both JSON pins in its single uploaded code_file.
"""

from pathlib import Path
import hashlib
import json
import shutil


MODEL_ALIAS = "gemma-4-12b-it-qat-w4a16-ct"
LORA_NAME = "p013_behavior_lora"
ARMS = ("base", "adapter")
MAX_CONCURRENT = 4
VLLM_EXTRA_ARGS = [
    "--max-num-seqs", "4", "--hf-overrides",
    '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_adapter(input_root, agent_root, adapter_source):
    """Stage only the independently exported final adapter, never checkpoints.

    The SFT kernel output contains several adapters. Restrict to its final
    training/adapter directory, then verify both exact frozen file hashes.
    """
    input_root, agent_root = Path(input_root), Path(agent_root)
    source_slug = adapter_source["source_kernel"].split("/")[1]
    candidates = [
        path for path in input_root.rglob("adapter_model.safetensors")
        if source_slug in path.parts and path.parent.name == "adapter"
        and path.parent.parent.name == "training"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one final P013 training/adapter output; found "
            + str([str(path) for path in candidates])
        )
    src = candidates[0].parent
    dest = agent_root / "adapter" / "adapters" / LORA_NAME
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, pin in adapter_source["files"].items():
        source = src / name
        actual = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
        if actual != pin:
            raise RuntimeError("P013 adapter differs from frozen source: " + name)
        shutil.copyfile(source, dest / name)
        files[name] = actual
    config = json.loads((dest / "adapter_config.json").read_text())
    if config["r"] != 8 or config["lora_alpha"] != 16:
        raise RuntimeError("Unexpected trained LoRA rank/alpha")
    if (agent_root / "base" / "adapters").exists():
        raise RuntimeError("Base arm must not contain an adapter")
    return {"source": str(src), "staged": str(dest), "files": files,
            "lora_name": LORA_NAME, "repair_only": True}


def prove_role_routing(agent_root):
    """Return the role mapping after checking the actual staged YAML files."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))
    root = Path(agent_root)
    repairs = {}
    for arm in ARMS:
        agent = yaml.load((root / arm / "agent.yaml").read_text(), Loader=Loader)
        repair = yaml.load((root / arm / "repair.yaml").read_text(), Loader=Loader)
        review = yaml.load((root / arm / "review.yaml").read_text(), Loader=Loader)
        if agent["agent_class"] != "SequentialAgent" or agent.get("adapter"):
            raise RuntimeError("Root must be a base sequential workflow")
        expected = LORA_NAME if arm == "adapter" else None
        if repair.get("adapter") != expected or review.get("adapter") is not None:
            raise RuntimeError("LoRA must route only to the adapter arm repair role")
        if repair["model"] != MODEL_ALIAS or review["model"] != MODEL_ALIAS:
            raise RuntimeError("12B model identity mismatch")
        if repair["tools"] != [
            "run_command", "read_file", "edit_file", "write_file", "get_status"
        ]:
            raise RuntimeError("Repair must retain the five-tool training protocol")
        if review["include_contents"] != "none":
            raise RuntimeError("Review context separation changed")
        prompt = (root / arm / "prompts/review.md").read_text()
        if "{problem_description}" not in prompt or "{repair_handoff}" not in prompt:
            raise RuntimeError("Review is missing explicit issue or handoff")
        repairs[arm] = repair
    normalized_adapter = dict(repairs["adapter"])
    normalized_adapter.pop("adapter")
    if normalized_adapter != repairs["base"]:
        raise RuntimeError("Repair arms differ beyond the intended LoRA reference")
    for rel in ("agent.yaml", "review.yaml", "prompts/repair.md",
                "prompts/review.md", "configs/sampling.yaml", "eval_config.yaml"):
        if (root / "base" / rel).read_bytes() != (root / "adapter" / rel).read_bytes():
            raise RuntimeError("Unexpected arm difference: " + rel)
    return {"base": {"repair": MODEL_ALIAS, "review": MODEL_ALIAS},
            "adapter": {"repair": LORA_NAME, "review": MODEL_ALIAS}}


def register_worker_aliases(models, agent_dir, served_model, base_url):
    """Use the same adapter aliases as official VllmServer/ModelRegistry."""
    from adk_submission import discover_adapters
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    models.register(MODEL_ALIAS, LiteLlm(
        model="openai/" + served_model, api_base=base_url, api_key="EMPTY"))
    adapters = discover_adapters(
        agent_dir, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    for name in adapters.adapters:
        for alias in (name, "adapter:" + name):
            models.register(alias, LiteLlm(
                model="openai/" + name, api_base=base_url, api_key="EMPTY"))
    return adapters

ADAPTER_SOURCE = {'source_kernel': 'ccjjllt/p013-12b-behavior-sft-34/1', 'lora_name': 'p013_behavior_lora', 'files': {'adapter_config.json': {'bytes': 1192, 'sha256': '443b33985cdc21066672bc529b72facdb4a205678944315e7cbee8696b6a4dee'}, 'adapter_model.safetensors': {'bytes': 115020728, 'sha256': '54de0ca38cdcb284509f7678cd59077217b169e3ee84649e78791faa8b5ae2a3'}}, 'training_complete': True, 'runtime_compatibility_verified': False}
MODEL_PIN = {'ref': 'google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2', 'model_alias': 'gemma-4-12b-it-qat-w4a16-ct', 'tokenizer_bytes': 32169626, 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'config_sha256': 'ba14713d084391532b285f7c252b954b7c6c0db97f427a9d8fb788edc2949168', 'tokenizer_config_sha256': 'ee66825f1a587ca13bcb90d7e60f59bc99e84f3b8bc0cefdc3ea595cd5abf773', 'vllm_extra_args': ['--max-num-seqs', '4', '--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}']}


def main():
    started = globals().get('P014_SESSION_START', time.monotonic())
    run_id = 'run-014-' + SETTINGS['study'] + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    out = Path('/kaggle/working/results/014') / run_id
    out.mkdir(parents=True)
    deadline = started + min(6900, SETTINGS.get('wall_seconds', 6900))
    tasks_path = locate('agent-tasks.json')
    panel = tasks_path.parent
    task_document = json.loads(tasks_path.read_text())
    tasks = task_document if isinstance(task_document, list) else task_document['tasks']
    by_id = {t['instance_id']: t for t in tasks}
    selected = SETTINGS['task_ids']
    if len(selected) != 4 or len(set(selected)) != 4:
        raise RuntimeError('12B first batch requires exactly four task IDs fixed before launch')
    if SETTINGS['model_alias'] != MODEL_ALIAS:
        raise RuntimeError('12B model alias mismatch')
    tasks = [by_id[iid] for iid in selected]
    criteria_document = json.loads((panel / 'host-criteria.json').read_text())
    criterion_items = criteria_document if isinstance(criteria_document, list) else criteria_document['items']
    criteria = {item['instance_id']: item for item in criterion_items}
    arms = list(BUNDLE)
    if arms != list(ARMS):
        raise RuntimeError('12B arm order must be base, adapter')
    agents = out / 'frozen-agents'
    for arm, files in BUNDLE.items():
        for rel, text in files.items():
            save(agents / arm / rel, text)
    adapter_identity = stage_adapter('/kaggle/input', agents, ADAPTER_SOURCE)
    role_routing = prove_role_routing(agents)
    manifest = {'run_id': run_id, 'settings': SETTINGS, 'task_ids': selected,
                'task_input_sha256': digest(tasks_path), 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'config_sha256': {arm: {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()} for arm, files in BUNDLE.items()},
                'adapter_source': adapter_identity, 'role_routing': role_routing,
                'model_pin': MODEL_PIN, 'planned_attempts': len(tasks) * len(arms), 'concurrency': SETTINGS.get('concurrency', 4),
                'source': 'new developer panel; host-only criteria; no reference edit enters agent',
                'versions': {name: importlib.metadata.version(name) for name in ('swegemma', 'adk-submission', 'vllm', 'torch')}}
    save(out / 'experiment-manifest.json', manifest)
    save(out / 'frozen-tasks.json', tasks)
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    model = locate('tokenizer.json') if not SETTINGS.get('model_path_pattern') else None
    if model is None:
        model_dirs = [p.parent for p in Path('/kaggle/input').rglob('tokenizer.json') if SETTINGS['model_path_pattern'] in str(p)]
        if len(model_dirs) != 1:
            raise RuntimeError(f'Model mount ambiguous/missing: {model_dirs}')
        model = model_dirs[0]
    else:
        model = model.parent
    for filename, key in [('tokenizer.json', 'tokenizer_sha256'), ('config.json', 'config_sha256'), ('tokenizer_config.json', 'tokenizer_config_sha256')]:
        if digest(model / filename) != MODEL_PIN[key]:
            raise RuntimeError('12B CT v2 source differs: ' + filename)
    if (model / 'tokenizer.json').stat().st_size != MODEL_PIN['tokenizer_bytes']:
        raise RuntimeError('12B tokenizer byte count differs')
    manifest['hardware'] = [{'name': torch.cuda.get_device_name(i), 'memory': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    if len(manifest['hardware']) != 4 or any('L4' not in x['name'] for x in manifest['hardware']):
        raise RuntimeError(f'Requires four L4s, got {manifest["hardware"]}')
    adapter_manifest = discover_adapters(str(agents / arms[-1]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    if sorted(adapter_manifest.adapters) != [LORA_NAME]:
        raise RuntimeError('Expected only the P013 behavior LoRA in adapter arm')
    base_manifest = discover_adapters(str(agents / 'base'), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    if base_manifest.adapters:
        raise RuntimeError('Base arm unexpectedly contains adapters')
    config = VllmConfig(model=str(model), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                        max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
                        enable_auto_tool_choice=True, enable_lora=True, tensor_parallel_size=4,
                        startup_timeout=1200, extra_args=['--max-num-seqs', str(SETTINGS.get('concurrency', 4)), *VLLM_EXTRA_ARGS[2:]])
    server = VllmServer(config, adapter_manifest=adapter_manifest)
    worker = out / 'attempt_worker.py'
    save(worker, WORKER_SOURCE)
    schedule = []
    for slot, task in enumerate(tasks):
        for arm in (arms if slot % 2 == 0 else list(reversed(arms))):
            ordinal = len(schedule) + 1
            entry = {'ordinal': ordinal, 'task_slot': slot + 1, 'task_id': task['instance_id'], 'variant': arm,
                     'attempt_id': f'{ordinal:02d}-{task["instance_id"]}-{arm}'}
            entry['results_dir'] = str(out / 'attempts' / entry['attempt_id'])
            schedule.append((entry, task))
    save(out / 'schedule.json', [entry for entry, task in schedule])
    active, rows = [], []
    telemetry_stop = threading.Event()
    def telemetry():
        with (out / 'gpu-utilization.jsonl').open('a') as log:
            while not telemetry_stop.is_set():
                p = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                log.write(json.dumps({'seconds': time.monotonic()-started, 'samples': p.stdout.strip().splitlines()})+'\n'); log.flush()
                telemetry_stop.wait(5)
    thread = threading.Thread(target=telemetry, daemon=True)
    def collect(record, forced=None):
        entry, process, log, launch_time = record
        log.close()
        path = Path(entry['results_dir']) / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {'resolved': None, 'error': forced or f'worker exit {process.returncode}'}
        row = {**entry, **result, 'worker_exit_code': process.returncode, 'outer_seconds': time.monotonic()-launch_time}
        if forced:
            row['error'] = forced
        rows.append(row)
        save(out / 'analysis-rows.json', rows)
        print('RESULT ' + json.dumps(row, ensure_ascii=False), flush=True)
    try:
        thread.start()
        print('PHASE model_start', flush=True)
        server.start()
        manifest['model_startup_seconds'] = time.monotonic()-started
        manifest['server_command'] = server.build_cmd()
        save(out / 'experiment-manifest.json', manifest)
        print('PHASE rolling_evaluation', flush=True)
        queue = iter(schedule)
        exhausted = False
        while active or not exhausted:
            while not exhausted and len(active) < SETTINGS.get('concurrency', 4) and time.monotonic() < deadline-150:
                try:
                    entry, task = next(queue)
                except StopIteration:
                    exhausted = True
                    break
                attempt = Path(entry['results_dir']); attempt.mkdir(parents=True)
                snapshot = panel / 'snapshots' / (task['instance_id'] + '.tar.gz')
                spec = {'entry': entry, 'agent_dir': str(agents / entry['variant']), 'task': task,
                        'snapshot': str(snapshot), 'base_url': str(server.base_url),
                        'served_model': str(getattr(server, 'effective_model_path', model)), 'model_alias': SETTINGS['model_alias'],
                        'budgets': SETTINGS['budgets'], 'judge': criteria[task['instance_id']],
                        'panel_root': str(panel), 'total_tasks': len(schedule)}
                save(attempt / 'spec.json', spec)
                log = (attempt / 'worker.log').open('w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                process = subprocess.Popen([sys.executable, '-u', str(worker), str(attempt/'spec.json')], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((entry, process, log, time.monotonic()))
                print('LAUNCH ' + entry['attempt_id'], flush=True)
            if time.monotonic() >= deadline-150:
                exhausted = True
            for record in active[:]:
                entry, process, log, launch_time = record
                forced = None
                if process.poll() is None and (time.monotonic()-launch_time > SETTINGS.get('attempt_seconds', 1020) or time.monotonic() >= deadline-60):
                    forced = 'experiment_wall_timeout'; stop(process)
                if process.poll() is not None:
                    collect(record, forced); active.remove(record)
            if active:
                time.sleep(1)
    except BaseException:
        save(out / 'failure.txt', traceback.format_exc())
        raise
    finally:
        for record in active:
            stop(record[1]); collect(record, 'interrupted')
        telemetry_stop.set(); thread.join(timeout=10)
        server_log = getattr(server, 'log_path', None)
        if server_log and Path(server_log).exists():
            shutil.copyfile(server_log, out/'vllm-server.log')
        server.stop(timeout=10)
        summary = {'completed_attempts': len(rows), 'planned_attempts': len(schedule), 'wall_seconds': time.monotonic()-started,
                   'arms': {arm: {'resolved': sum(r.get('resolved') is True and r['variant']==arm for r in rows), 'attempts': sum(r['variant']==arm for r in rows)} for arm in arms}}
        save(out / 'summary.json', summary)
        print('FINAL ' + json.dumps(summary), flush=True)
        with zipfile.ZipFile('/kaggle/working/p014-'+SETTINGS['study']+'-output.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file() and path.suffix not in ('.safetensors', '.pt', '.bin'):
                    archive.write(path, path.relative_to(out.parent))


if __name__ == '__main__':
    main()
