"""Build the private 6032-token save_on_cpu diagnostic kernel. Building NEVER launches Kaggle."""
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
    "training/configs/sft_qat_nf4_ddp.yaml",
)
ENCODED = {
    "encoded-capacity-8192.jsonl": HERE / "encoded" / "encoded-capacity-8192.jsonl",
}
KERNEL_ID = "ccjjllt/p005-save-on-cpu-6032-tm"
KAGGLE_TIMEOUT = 1080
ARM_TIMEOUT = 450
WRAPPER_DEADLINE = 1000

WRAPPER = r'''
import base64, csv, gzip, hashlib, importlib.util, json, os, platform, shutil, signal
import subprocess, sys, threading, time, traceback
from pathlib import Path

OUT = Path('/kaggle/working/save-on-cpu-6032-tm')
BUNDLE = Path('/tmp/p005-save-on-cpu-bundle')
DEPS = Path('/tmp/p005-save-on-cpu-deps')
KERNEL_ID = 'ccjjllt/p005-save-on-cpu-6032-tm'
ARM_TIMEOUT = 450
WRAPPER_DEADLINE = 1000
ARMS = (('baseline', 0), ('save_on_cpu', 1))

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
    spec = importlib.util.spec_from_file_location('p005_save_on_cpu_worker', BUNDLE / 'worker.py')
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

def discard_adapters(output):
    removed = []
    for path in list(output.rglob('adapter')) + list(output.rglob('adapter_model.safetensors')):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
        elif path.is_file():
            path.unlink()
            removed.append(str(path))
    for path in output.rglob('*.pt'):
        path.unlink()
        removed.append(str(path))
    return removed

def collect_arm(arm_dir):
    ranks = []
    for path in sorted(arm_dir.glob('rank-*-report.json')):
        try:
            ranks.append(read_dict(path))
        except Exception as exc:
            ranks.append({'path': str(path), 'status': 'unreadable', 'error': repr(exc)})
    rank_zero = next((value for value in ranks if value.get('rank') == 0), {})
    return {'arm': arm_dir.name, 'rank_reports': ranks, 'rank_zero_status': rank_zero.get('status'),
            'optimizer_reached': bool(rank_zero.get('optimizer_reached')),
            'oom': bool(rank_zero.get('oom')), 'failure_phase': rank_zero.get('failure_phase'),
            'last_decoder_layer': rank_zero.get('last_decoder_layer'),
            'phases': rank_zero.get('phases') or [], 'load_seconds': rank_zero.get('load_seconds'),
            'traceback': rank_zero.get('traceback'),
            'decoder_checkpoint_calls': rank_zero.get('decoder_checkpoint_calls'),
            'host_memory': rank_zero.get('host_memory')}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + WRAPPER_DEADLINE
    report = {'schema': 'p005-save-on-cpu-wrapper-v2', 'status': 'started', 'success': False,
              'python': sys.version, 'platform': platform.platform(),
              'kernel_id': KERNEL_ID,
              'arm_timeout_seconds': ARM_TIMEOUT, 'wrapper_deadline_seconds': WRAPPER_DEADLINE,
              'recommended_server_timeout_seconds': 1080,
              'accelerator_request': 'NvidiaL4 (four L4 devices)', 'internet': False,
              'cache_kernel': 'ccjjllt/p005-cpu-nf4-cache',
              'source_manifest': SOURCE_MANIFEST, 'synthetic': False,
              'loss_is_not_repair_gain': True, 'did_not_truncate': True,
              'did_not_retry_16k': True, 'adapter_discarded': True, 'events': [], 'arms': []}
    def event(stage, **values):
        report.update(values)
        report['events'].append({'stage': stage, 'elapsed_seconds': time.monotonic() - started})
        write_json(OUT / 'wrapper-report.json', report)
        print(json.dumps({'stage': stage, **{k: values[k] for k in values if k != 'traceback'}},
                         default=str, allow_nan=False), flush=True)
    stop = threading.Event()
    def monitor():
        with (OUT / 'gpu-telemetry.jsonl').open('w', buffering=1) as log:
            while not stop.is_set():
                sample = {'elapsed_seconds': time.monotonic() - started}
                try:
                    query = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8)
                    sample['csv'] = query.stdout.strip()
                    sample['exit_code'] = query.returncode
                    sample['gpus'] = [dict(zip(('index', 'name', 'total_mib', 'used_mib', 'utilization_percent', 'power_watts'),
                                               [item.strip() for item in row])) for row in csv.reader(query.stdout.splitlines())]
                except Exception as exc:
                    sample['error'] = repr(exc)
                sample['host'] = {
                    'meminfo': None,
                    'cgroup_memory_events': None,
                    'cgroup_memory_current': None,
                }
                try:
                    meminfo = Path('/proc/meminfo')
                    if meminfo.is_file():
                        sample['host']['meminfo'] = meminfo.read_text()
                    else:
                        sample['host']['meminfo_missing'] = True
                except Exception:
                    sample['host']['meminfo_missing'] = True
                found_events = False
                for candidate in ('/sys/fs/cgroup/memory.events', '/sys/fs/cgroup/memory/memory.events'):
                    p = Path(candidate)
                    try:
                        if p.is_file():
                            sample['host']['cgroup_memory_events'] = p.read_text()
                            sample['host']['cgroup_memory_events_path'] = candidate
                            found_events = True
                            break
                    except Exception:
                        sample['host']['cgroup_memory_events_missing'] = True
                if not found_events:
                    sample['host']['cgroup_memory_events_missing'] = True
                found_current = False
                for candidate in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
                    p = Path(candidate)
                    try:
                        if p.is_file():
                            sample['host']['cgroup_memory_current'] = p.read_text().strip()
                            found_current = True
                            break
                    except Exception:
                        sample['host']['cgroup_memory_current_missing'] = True
                if not found_current:
                    sample['host']['cgroup_memory_current_missing'] = True
                log.write(json.dumps(sample, allow_nan=False) + '\n')
                stop.wait(15)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
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
        for arm, flag in ARMS:
            remaining = deadline - time.monotonic()
            arm_out = OUT / arm
            arm_out.mkdir(parents=True, exist_ok=False)
            if remaining < 150:
                report['arms'].append({'arm': arm, 'skipped': True, 'reason': 'insufficient remaining time for a fresh load'})
                event('arm_skipped', arm=arm, remaining_seconds=remaining)
                continue
            timeout = min(ARM_TIMEOUT, max(30, remaining - 20))
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                       '--nproc_per_node=4', '--max_restarts=0', '--monitor_interval=1',
                       str(BUNDLE / 'worker.py'), '--arm', arm, '--save-on-cpu', str(flag),
                       '--cache', str(cache), '--bundle', str(BUNDLE), '--output-root', str(arm_out)]
            run_started = time.monotonic()
            proc = reader = None
            arm_exit = 2
            with (arm_out / 'worker.log').open('w') as log:
                proc = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=True)
                def relay(stream=proc.stdout, sink=log):
                    for line in stream:
                        sink.write(line)
                        sink.flush()
                        if line.startswith('{') or 'Traceback' in line or 'OutOfMemory' in line:
                            print(line.rstrip(), flush=True)
                reader = threading.Thread(target=relay, daemon=True)
                reader.start()
                event('torchrun_started', arm=arm, command=command, pid=proc.pid, timeout_seconds=timeout)
                try:
                    arm_exit = proc.wait(timeout=timeout)
                    event('torchrun_finished', arm=arm, worker_exit_code=arm_exit,
                          worker_wall_seconds=time.monotonic() - run_started)
                except subprocess.TimeoutExpired:
                    arm_exit = 124
                    event('torchrun_timeout', arm=arm, timed_out=True, worker_exit_code=arm_exit,
                          worker_wall_seconds=time.monotonic() - run_started)
                finally:
                    kill_process_group(proc)
                    if reader is not None:
                        reader.join(timeout=10)
            summary = collect_arm(arm_out)
            summary['worker_exit_code'] = arm_exit
            summary['fresh_process'] = True
            report['arms'].append(summary)
            write_json(OUT / 'wrapper-report.json', report)
        discarded = discard_adapters(OUT)
        report['discarded_adapter_paths'] = discarded
        statuses = [arm.get('rank_zero_status') for arm in report['arms'] if not arm.get('skipped')]
        recorded = [status in {'completed', 'oom'} for status in statuses]
        completed = len(recorded) == 2 and all(recorded)
        report['success'] = completed
        report['status'] = 'completed' if completed else 'failed'
        exit_code = 0 if completed else 2
    except Exception as exc:
        event('wrapper_failed', error=repr(exc), traceback=traceback.format_exc())
        report['status'] = 'failed'
        report['success'] = False
        exit_code = 2
    finally:
        stop.set()
        thread.join(timeout=10)
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
    encoded_meta = {name: {"sha256": blob["sha256"], "bytes": blob["bytes"]} for name, blob in encoded_gz.items()}
    manifest = {"schema": "p005-save-on-cpu-source-bundle-v1", "files": items, "encoded": encoded_meta,
                "bundle_sha256": hashlib.sha256(
                    json.dumps({"files": items, "encoded": {n: encoded_gz[n]["sha256"] for n in encoded_gz}},
                               sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    source = ("ENCODED_GZ = " + repr(encoded_gz) + "\nBUNDLE_FILES = " + repr(sources)
              + "\nSOURCE_MANIFEST = " + repr(manifest) + "\n" + WRAPPER)
    ast.parse(source, filename="probe.py")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "probe.py").write_text(source)
    metadata = {"id": KERNEL_ID, "title": "P005 save_on_cpu 6032 train-mode",
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
                                            "--accelerator", "NvidiaL4", "--timeout", str(KAGGLE_TIMEOUT)],
              "arm_timeout_seconds": ARM_TIMEOUT, "wrapper_deadline_seconds": WRAPPER_DEADLINE,
              "kaggle_timeout_seconds": KAGGLE_TIMEOUT, "expected_devices": 4,
              "l4_billing_multiplier": 2, "max_billed_hours": round(KAGGLE_TIMEOUT * 2 / 3600, 3),
              "source_manifest": manifest,
              "quota_note": "4xL4 bills 2x wall. Cap ~0.6h billed; timeout 1080s wall."}
    (destination / "launch-plan.json").write_text(json.dumps(launch, indent=2) + "\n")
    return {"built": str(destination), "launch_performed": False, "bundle_sha256": manifest["bundle_sha256"],
            "kernel_id": KERNEL_ID, "source_files": len(sources), "encoded_files": len(encoded_gz),
            "code_bytes": len(source.encode()), "kaggle_timeout_seconds": KAGGLE_TIMEOUT}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "kernel")
    args = parser.parse_args(argv)
    print(json.dumps(build(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
