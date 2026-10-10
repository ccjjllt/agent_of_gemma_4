"""Snapshot a private offline DDP probe. Building NEVER launches Kaggle work.

Only after the CPU NF4 cache run is COMPLETE and its success report is checked,
an operator may separately push with --accelerator NvidiaL4 --timeout 1500.
The generated kernel mounts two CPU notebook outputs and no 62 GB source model.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FILES = ("training/train_sft.py", "training/train_ddp.py", "training/common.py",
         "training/export_peft.py", "training/memory.py", "training/encoding/gemma_tool_chat.py",
         "training/configs/harness_tools.json")

WRAPPER = r'''
import csv, hashlib, importlib.util, json, math, os, platform, shutil, signal
import struct, subprocess, sys, threading, time, traceback
from pathlib import Path

OUT = Path('/kaggle/working/ddp-probe')
BUNDLE = Path('/tmp/p005-ddp-bundle')
DEPS = Path('/tmp/p005-ddp-deps')
WORKER_TIMEOUT = 1200

def write_json(path, value):
    if not isinstance(value, dict):
        raise TypeError('Probe report must be a JSON object')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def read_dict(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError('Expected a JSON object: ' + str(path))
    return value

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def unpack_bundle():
    BUNDLE.mkdir(parents=True, exist_ok=False)
    expected = SOURCE_MANIFEST['files']
    if set(expected) != set(BUNDLE_FILES):
        raise RuntimeError('Bundle manifest filenames differ')
    for name, content in BUNDLE_FILES.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Invalid bundle path')
        encoded = content.encode()
        item = expected[name]
        if len(encoded) != item['bytes'] or hashlib.sha256(encoded).hexdigest() != item['sha256']:
            raise RuntimeError('Bundled source hash mismatch: ' + name)
        destination = BUNDLE / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
    write_json(BUNDLE / 'bundle-manifest.json', SOURCE_MANIFEST)
    write_json(OUT / 'source-manifest.json', SOURCE_MANIFEST)
    spec = importlib.util.spec_from_file_location('p005_ddp_probe_worker', BUNDLE / 'worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker

def checked_wheels(input_root):
    matches = sorted(Path(input_root).rglob('cpu-stage-report.json'))
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one staged CPU dependency report')
    stage = read_dict(matches[0])
    if stage.get('success') is not True:
        raise RuntimeError('CPU dependency preparation did not succeed')
    if stage['python'].split()[0].split('.')[:2] != platform.python_version().split('.')[:2]:
        raise RuntimeError('CPU/GPU Python ABI differs; do not install incompatible wheels')
    if len(stage['wheels']) != 7:
        raise RuntimeError('Expected the reviewed seven non-torch wheels')
    wheels = []
    for item in stage['wheels']:
        name = item['name']
        if Path(name).name != name or not name.endswith('.whl'):
            raise RuntimeError('Invalid staged wheel filename')
        if name.lower().startswith(('torch-', 'triton-', 'nvidia_')):
            raise RuntimeError('Replacing stock GPU torch/CUDA is prohibited')
        path = matches[0].parent / 'training-wheels' / name
        if path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
            raise RuntimeError('Staged wheel bytes differ: ' + name)
        wheels.append(str(path))
    return wheels, {'report': str(matches[0]), 'python': stage['python'], 'wheels': stage['wheels']}

def kill_process_group(proc):
    if proc is None:
        return
    # Kill by session/process-group, even if the torchrun leader already exited.
    # Its descendants must not continue consuming GPU after failure/timeout.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        proc.wait(timeout=10)

def adapter_files(adapter):
    config = read_dict(adapter / 'adapter_config.json')
    if config.get('peft_type') != 'LORA' or config.get('r') != 8:
        raise RuntimeError('Retained adapter has unexpected PEFT type/rank')
    weights = adapter / 'adapter_model.safetensors'
    size = weights.stat().st_size
    with weights.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise RuntimeError('Truncated adapter safetensors header')
        header_size = struct.unpack('<Q', prefix)[0]
        if not 0 < header_size <= min(size - 8, 16 * 1024 * 1024):
            raise RuntimeError('Invalid adapter safetensors header size')
        header = json.loads(f.read(header_size))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    if not tensors or not all('lora_' in name for name in tensors):
        raise RuntimeError('Adapter file contains no valid LoRA tensor names')
    for name, tensor in tensors.items():
        offsets = tensor['data_offsets']
        if len(offsets) != 2 or not 0 <= offsets[0] < offsets[1] <= size - 8 - header_size:
            raise RuntimeError('Invalid LoRA tensor offsets: ' + name)
    return {'tensor_count': len(tensors), 'files': [
        {'name': name, 'bytes': (adapter / name).stat().st_size, 'sha256': sha256_file(adapter / name)}
        for name in ('adapter_config.json', 'adapter_model.safetensors')]}

def retain_latest_adapter(output):
    training = output / 'training'
    checkpoints, invalid = [], []
    for directory in sorted(training.glob('checkpoint-*')):
        progress_path = directory / 'progress.json'
        if not progress_path.is_file():
            invalid.append({'path': str(directory), 'reason': 'no atomic completion marker'})
            continue
        try:
            progress = read_dict(progress_path)
            adapter = directory / 'adapter'
            if not (adapter / 'adapter_config.json').is_file() or not (adapter / 'adapter_model.safetensors').is_file():
                raise RuntimeError('missing adapter files')
            checkpoints.append((int(progress['global_step']), directory, progress))
        except Exception as exc:
            invalid.append({'path': str(directory), 'reason': repr(exc)})
    retained = None
    # Verify the latest complete adapter; if its file is damaged, try an earlier
    # completed checkpoint rather than discarding already proven training work.
    for step, directory, progress in sorted(checkpoints, reverse=True):
        try:
            inspection = adapter_files(directory / 'adapter')
            destination = output / 'adapter'
            if destination.exists():
                raise RuntimeError('Refusing to overwrite a previously retained adapter')
            shutil.move(str(directory / 'adapter'), str(destination))
            # PEFT may create README.md. Keep the actual config/weights only.
            for file in destination.iterdir():
                if file.name not in {'adapter_config.json', 'adapter_model.safetensors'}:
                    if file.is_dir(): shutil.rmtree(file)
                    else: file.unlink()
            retained = {'step': step, 'path': str(destination), 'progress': progress, **inspection}
            break
        except Exception as exc:
            invalid.append({'path': str(directory), 'reason': repr(exc)})
    removed_bytes = 0
    for path in training.rglob('*.pt'):
        removed_bytes += path.stat().st_size
        path.unlink()
    # Only probe-owned checkpoint and duplicate final-adapter directories.
    for directory in training.glob('checkpoint-*'):
        shutil.rmtree(directory)
    if (training / 'adapter').exists():
        shutil.rmtree(training / 'adapter')
    return {'retained': retained, 'removed_optimizer_and_rng_bytes': removed_bytes,
            'completed_checkpoint_steps': sorted(step for step, _, _ in checkpoints),
            'invalid_checkpoints': invalid}

def collect_results(output):
    ranks = []
    for path in sorted(output.glob('rank-*-report.json')):
        try:
            ranks.append(read_dict(path))
        except Exception as exc:
            ranks.append({'path': str(path), 'status': 'unreadable', 'error': repr(exc)})
    metrics, metric_errors = [], []
    metric_path = output / 'training' / 'metrics.jsonl'
    if metric_path.is_file():
        for line_number, line in enumerate(metric_path.read_text().splitlines(), 1):
            if line.strip():
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict): raise TypeError('Step metric must be a JSON object')
                    metrics.append(item)
                except Exception as exc:
                    metric_errors.append({'line': line_number, 'error': repr(exc)})
    rank_zero = next((value for value in ranks if value.get('rank') == 0), {})
    zero_steps = {entry['step']: entry for entry in rank_zero.get('steps', [])}
    for metric in metrics:
        diagnostic = zero_steps.get(metric['step'], {}).get('diagnostic_timing', {}).get('total_seconds')
        metric['raw_seconds_include_probe_diagnostics'] = True
        if diagnostic is not None:
            metric['rank0_probe_validation_seconds'] = diagnostic
            adjusted = max(0., metric['seconds'] - diagnostic)
            metric['estimated_seconds_excluding_probe_validation'] = adjusted
            metric['estimated_selected_tokens_per_second_excluding_probe_validation'] = metric['supervised_tokens'] / adjusted if adjusted > 0 else None
    verified = []
    if len(ranks) == 4 and {value.get('rank') for value in ranks} == set(range(4)):
        for step in range(1, 5):
            values = [next((x for x in r.get('steps', []) if x['step'] == step), None) for r in ranks]
            if all(value and value['all_rank_hashes_equal'] and value['all_rank_adapters_changed']
                   and all(g['finite'] and g['nonzero_gradient_tensors'] > 0 for g in value['all_rank_gradients'])
                   for value in values):
                hashes = {u['trainable_sha256'] for value in values for u in value['all_rank_updates']}
                metric = next((x for x in metrics if x['step'] == step), None)
                if len(hashes) == 1 and metric and math.isfinite(metric['loss']) and metric['grad_norm'] > 0:
                    verified.append(step)
    return {'rank_reports': ranks, 'step_results': metrics, 'metric_read_errors': metric_errors,
            'all_rank_verified_steps': verified,
            'first_two_steps_verified': all(step in verified for step in (1, 2))}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'schema': 'p005-ddp-probe-wrapper-v1', 'status': 'started', 'success': False,
              'python': sys.version, 'platform': platform.platform(),
              'worker_timeout_seconds': WORKER_TIMEOUT, 'recommended_server_timeout_seconds': 1500,
              'accelerator_request': 'NvidiaL4 (four L4 devices)', 'internet': False,
              'cache_kernel': 'ccjjllt/p005-cpu-nf4-cache',
              'source_manifest': SOURCE_MANIFEST, 'synthetic': True, 'training_eligible': False,
              'capacity_claim': 'Only completed measured lengths; no 4096-token claim', 'events': []}
    def event(stage, **values):
        report.update(values)
        report['events'].append({'stage': stage, 'elapsed_seconds': time.monotonic() - started})
        write_json(OUT / 'wrapper-report.json', report)
        print(json.dumps({'stage': stage, **values}, allow_nan=False), flush=True)
    stop = threading.Event()
    def monitor():
        with (OUT / 'gpu-telemetry.jsonl').open('w', buffering=1) as log:
            while not stop.is_set():
                sample = {'elapsed_seconds': time.monotonic() - started}
                try:
                    query = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
                    sample.update(csv=query.stdout.strip(), exit_code=query.returncode)
                    sample['gpus'] = [dict(zip(('index', 'name', 'total_mib', 'used_mib', 'utilization_percent', 'power_watts'),
                                               [item.strip() for item in row])) for row in csv.reader(query.stdout.splitlines())]
                except Exception as exc:
                    sample['error'] = repr(exc)
                log.write(json.dumps(sample, allow_nan=False) + '\n')
                stop.wait(10)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    proc = reader = None
    exit_code = 2
    try:
        worker = unpack_bundle()
        cache = worker.find_cache('/kaggle/input')
        cache_info = worker.verify_cache(cache)
        # CPU producer success is mandatory before model loading or torchrun.
        event('cpu_cache_verified', cache=cache_info)
        wheels, wheel_info = checked_wheels('/kaggle/input')
        event('cpu_wheels_verified', dependencies=wheel_info)
        install_started = time.monotonic()
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(DEPS), *wheels],
                       check=True, timeout=120)
        event('dependencies_installed', dependency_install_seconds=time.monotonic() - install_started)
        environment = dict(os.environ, PYTHONPATH=str(DEPS) + os.pathsep + str(BUNDLE),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
            TORCH_NCCL_ASYNC_ERROR_HANDLING='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc_per_node=4', '--max_restarts=0', '--monitor_interval=1',
                   str(BUNDLE / 'worker.py'), '--cache', str(cache), '--output-root', str(OUT)]
        run_started = time.monotonic()
        with (OUT / 'worker.log').open('w') as log:
            proc = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, start_new_session=True)
            def relay():
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith('{') or 'Traceback' in line or 'OutOfMemory' in line:
                        print(line.rstrip(), flush=True)
            reader = threading.Thread(target=relay, daemon=True)
            reader.start()
            event('torchrun_started', command=command, pid=proc.pid)
            try:
                exit_code = proc.wait(timeout=WORKER_TIMEOUT)
                event('torchrun_finished', worker_exit_code=exit_code,
                      worker_wall_seconds=time.monotonic() - run_started)
            except subprocess.TimeoutExpired:
                exit_code = 124
                event('torchrun_timeout', timed_out=True, worker_exit_code=exit_code,
                      worker_wall_seconds=time.monotonic() - run_started)
            finally:
                kill_process_group(proc)
                reader.join(timeout=10)
    except Exception as exc:
        event('wrapper_failed', error=repr(exc), traceback=traceback.format_exc())
    finally:
        kill_process_group(proc)
        stop.set()
        thread.join(timeout=10)
        try:
            retention = retain_latest_adapter(OUT)
            report['adapter_retention'] = retention
            results = collect_results(OUT)
            report.update(results)
            completed = (exit_code == 0 and results['all_rank_verified_steps'] == [1, 2, 3, 4]
                         and retention['retained'] is not None and retention['retained']['step'] == 4
                         and all(rank['status'] == 'completed' for rank in results['rank_reports']))
            report['success'] = completed
            report['status'] = 'completed' if completed else ('partial_success' if results['first_two_steps_verified'] else 'failed')
            if not completed and exit_code == 0: exit_code = 2
        except Exception as exc:
            report.update(status='failed', success=False, collection_error=repr(exc), collection_traceback=traceback.format_exc())
            exit_code = exit_code or 2
        event('session_end', total_seconds=time.monotonic() - started, exit_code=exit_code)
    return exit_code

if __name__ == '__main__':
    raise SystemExit(main())
'''


def build(destination):
    sources = {name: (REPO / name).read_text() for name in FILES}
    sources["worker.py"] = (HERE / "ddp-probe" / "worker.py").read_text()
    for name, content in sources.items():
        if name.endswith(".py"):
            ast.parse(content, filename=name)
    items = {name: {"sha256": hashlib.sha256(content.encode()).hexdigest(), "bytes": len(content.encode())}
             for name, content in sorted(sources.items())}
    manifest = {"schema": "p005-ddp-source-bundle-v1", "files": items,
                "bundle_sha256": hashlib.sha256(json.dumps(items, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    source = "BUNDLE_FILES = " + repr(sources) + "\nSOURCE_MANIFEST = " + repr(manifest) + "\n" + WRAPPER
    ast.parse(source, filename="probe.py")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "probe.py").write_text(source)
    metadata = {"id": "ccjjllt/p005-training-ddp-probe", "title": "P005 Training DDP Probe",
                "code_file": "probe.py", "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": [],
                "competition_sources": ["gemma-4-developer-agent"],
                "kernel_sources": ["ccjjllt/p005-cpu-nf4-cache", "ccjjllt/p005-training-cpu-stage"],
                "model_sources": []}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    launch = {"launch_performed": False, "requires_cpu_cache_complete": True,
              "required_cache_success_fields": ["cache-report.json:success=true", "cache-child-report.json:stage=complete,success=true"],
              "command_for_operator_only": ["kaggle", "kernels", "push", "-p", str(destination),
                                            "--accelerator", "NvidiaL4", "--timeout", "1500"],
              "worker_timeout_seconds": 1200, "expected_devices": 4, "source_manifest": manifest}
    (destination / "launch-plan.json").write_text(json.dumps(launch, indent=2) + "\n")
    return {"built": str(destination), "launch_performed": False, "bundle_sha256": manifest["bundle_sha256"],
            "source_files": len(sources), "code_bytes": len(source.encode())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "ddp-probe")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
