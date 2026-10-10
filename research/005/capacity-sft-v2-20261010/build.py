"""Build the private offline capacity+SFT kernel. Building NEVER launches Kaggle."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FILES = (
    "training/train_sft.py",
    "training/train_ddp.py",
    "training/common.py",
    "training/export_peft.py",
    "training/memory.py",
    "training/encoding/gemma_tool_chat.py",
    "training/configs/harness_tools.json",
    "training/configs/sft_qat_nf4_ddp_16384.yaml",
)
ENCODED = {
    "encoded-train-16384.jsonl": HERE / "encoded" / "encoded-train-16384.jsonl",
    "encoded-capacity-8192.jsonl": HERE / "encoded" / "encoded-capacity-8192.jsonl",
}

WRAPPER = r'''
import base64, csv, gzip, hashlib, importlib.util, json, math, os, platform, shutil, signal
import struct, subprocess, sys, threading, time, traceback
from pathlib import Path

OUT = Path('/kaggle/working/capacity-sft-v2')
BUNDLE = Path('/tmp/p005-capacity-sft-bundle')
DEPS = Path('/tmp/p005-capacity-sft-deps')
WORKER_TIMEOUT = 1500

def write_json(path, value):
    if not isinstance(value, dict):
        raise TypeError('Report must be a JSON object')
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
    data_dir = BUNDLE / 'data'
    data_dir.mkdir()
    for name, blob in ENCODED_GZ.items():
        raw = gzip.decompress(base64.b64decode(blob['gz_b64']))
        if len(raw) != blob['bytes'] or hashlib.sha256(raw).hexdigest() != blob['sha256']:
            raise RuntimeError('Encoded data hash mismatch: ' + name)
        (data_dir / name).write_bytes(raw)
    write_json(BUNDLE / 'bundle-manifest.json', SOURCE_MANIFEST)
    write_json(OUT / 'source-manifest.json', SOURCE_MANIFEST)
    spec = importlib.util.spec_from_file_location('p005_capacity_sft_worker', BUNDLE / 'worker.py')
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
    return {'tensor_count': len(tensors), 'files': [
        {'name': name, 'bytes': (adapter / name).stat().st_size, 'sha256': sha256_file(adapter / name)}
        for name in ('adapter_config.json', 'adapter_model.safetensors')]}

def retain_sft_adapter(output):
    training = output / 'sft'
    source = training / 'adapter'
    invalid = []
    retained = None
    if source.is_dir() and (source / 'adapter_config.json').is_file() and (source / 'adapter_model.safetensors').is_file():
        try:
            inspection = adapter_files(source)
            destination = output / 'adapter'
            if destination.exists():
                raise RuntimeError('Refusing to overwrite a previously retained adapter')
            shutil.move(str(source), str(destination))
            for file in destination.iterdir():
                if file.name not in {'adapter_config.json', 'adapter_model.safetensors'}:
                    if file.is_dir(): shutil.rmtree(file)
                    else: file.unlink()
            retained = {'path': str(destination), **inspection}
        except Exception as exc:
            invalid.append({'path': str(source), 'reason': repr(exc)})
    removed_bytes = 0
    for path in output.rglob('*.pt'):
        removed_bytes += path.stat().st_size
        path.unlink()
    for directory in list(output.glob('capacity-*/checkpoint-*')) + list((training).glob('checkpoint-*') if training.exists() else []):
        shutil.rmtree(directory, ignore_errors=True)
    for leftover in (output / 'capacity-8192' / 'adapter', output / 'capacity-16384' / 'adapter', training / 'adapter'):
        if leftover.exists():
            shutil.rmtree(leftover)
    return {'retained': retained, 'removed_optimizer_and_rng_bytes': removed_bytes, 'invalid': invalid}

def collect_results(output):
    ranks = []
    for path in sorted(output.glob('rank-*-report.json')):
        try:
            ranks.append(read_dict(path))
        except Exception as exc:
            ranks.append({'path': str(path), 'status': 'unreadable', 'error': repr(exc)})
    metrics, metric_errors = [], []
    metric_path = output / 'sft' / 'metrics.jsonl'
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
    capacity = rank_zero.get('capacity') or []
    gpu_sft = bool(rank_zero.get('gpu_sft'))
    return {'rank_reports': ranks, 'sft_metrics': metrics, 'metric_read_errors': metric_errors,
            'capacity': capacity, 'gpu_sft': gpu_sft,
            'sft_skip_reason': rank_zero.get('sft_skip_reason'),
            'rank_zero_status': rank_zero.get('status'),
            'capacity_windows_ok': [c.get('window') for c in capacity if c.get('backprop_ok')]}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'schema': 'p005-capacity-sft-wrapper-v1', 'status': 'started', 'success': False,
              'python': sys.version, 'platform': platform.platform(),
              'worker_timeout_seconds': WORKER_TIMEOUT, 'recommended_server_timeout_seconds': 1800,
              'accelerator_request': 'NvidiaL4 (four L4 devices)', 'internet': False,
              'cache_kernel': 'ccjjllt/p005-cpu-nf4-cache',
              'source_manifest': SOURCE_MANIFEST, 'synthetic': False,
              'loss_is_not_repair_gain': True, 'did_not_truncate': True,
              'did_not_use_diagnostic_synthetic_2k': True, 'events': []}
    def event(stage, **values):
        report.update(values)
        report['events'].append({'stage': stage, 'elapsed_seconds': time.monotonic() - started})
        write_json(OUT / 'wrapper-report.json', report)
        print(json.dumps({'stage': stage, **{k: values[k] for k in values if k != 'traceback'}}, default=str, allow_nan=False), flush=True)
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
                   str(BUNDLE / 'worker.py'), '--cache', str(cache), '--bundle', str(BUNDLE),
                   '--output-root', str(OUT)]
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
            retention = retain_sft_adapter(OUT)
            report['adapter_retention'] = retention
            results = collect_results(OUT)
            report.update(results)
            capacity_recorded = bool(results['capacity'])
            sft_ok = results['gpu_sft'] and retention['retained'] is not None
            skipped_ok = ((not results['gpu_sft']) and capacity_recorded
                          and results['rank_zero_status'] in {'capacity_oom', 'capacity_only'})
            completed = exit_code == 0 and (sft_ok or skipped_ok)
            report['success'] = completed
            report['status'] = 'completed' if completed else 'failed'
            report['gpu_sft'] = bool(sft_ok)
            if sft_ok:
                report['adapter_fingerprint'] = retention['retained']['files']
            if not completed and exit_code == 0:
                exit_code = 2
        except Exception as exc:
            report.update(status='failed', success=False, collection_error=repr(exc),
                          collection_traceback=traceback.format_exc())
            exit_code = exit_code or 2
        event('session_end', total_seconds=time.monotonic() - started, exit_code=exit_code)
    return exit_code

if __name__ == '__main__':
    raise SystemExit(main())
'''


def build(destination):
    sources = {name: (REPO / name).read_text() for name in FILES}
    sources["training/configs/chat_template.jinja"] = (HERE / "chat_template.jinja").read_text()
    sources["worker.py"] = (HERE / "worker.py").read_text()
    for name, content in sources.items():
        if name.endswith(".py"):
            ast.parse(content, filename=name)
    encoded_gz = {}
    for name, path in ENCODED.items():
        raw = path.read_bytes()
        encoded_gz[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "gz_b64": base64.b64encode(gzip.compress(raw, 9)).decode("ascii"),
        }
    items = {name: {"sha256": hashlib.sha256(content.encode()).hexdigest(), "bytes": len(content.encode())}
             for name, content in sorted(sources.items())}
    manifest = {"schema": "p005-capacity-sft-source-bundle-v1", "files": items,
                "encoded": {name: {"sha256": blob["sha256"], "bytes": blob["bytes"]}
                            for name, blob in encoded_gz.items()},
                "bundle_sha256": hashlib.sha256(
                    json.dumps({"files": items, "encoded": {n: encoded_gz[n]["sha256"] for n in encoded_gz}},
                               sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    source = ("ENCODED_GZ = " + repr(encoded_gz) + "\nBUNDLE_FILES = " + repr(sources)
              + "\nSOURCE_MANIFEST = " + repr(manifest) + "\n" + WRAPPER)
    ast.parse(source, filename="probe.py")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "probe.py").write_text(source)
    metadata = {"id": "ccjjllt/p005-capacity-sft-v2", "title": "P005 Capacity SFT v2",
                "code_file": "probe.py", "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": [],
                "competition_sources": ["gemma-4-developer-agent"],
                "kernel_sources": ["ccjjllt/p005-cpu-nf4-cache", "ccjjllt/p005-training-cpu-stage"],
                "model_sources": []}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    launch = {"launch_performed": False, "requires_cpu_cache_complete": True,
              "required_cache_success_fields": [
                  "cache-report.json:success=true",
                  "cache-child-report.json:stage=complete,success=true"],
              "command_for_operator_only": ["kaggle", "kernels", "push", "-p", str(destination),
                                            "--accelerator", "NvidiaL4", "--timeout", "1800"],
              "worker_timeout_seconds": 1500, "expected_devices": 4, "source_manifest": manifest,
              "quota_note": "4xL4 bills 2x wall. Suggested cap capacity 0.6h + SFT 1.0h billed."}
    (destination / "launch-plan.json").write_text(json.dumps(launch, indent=2) + "\n")
    return {"built": str(destination), "launch_performed": False, "bundle_sha256": manifest["bundle_sha256"],
            "source_files": len(sources), "encoded_files": len(encoded_gz),
            "code_bytes": len(source.encode())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "kernel")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
