"""One bounded four-L4 SFT run on newly replayed real transactions.

Called before the shared CT inference server starts. Reuses the immutable P007
NF4 cache, synchronizes only LoRA gradients, and retains the final optimizer
checkpoint. This is an experiment stage, not a capacity sweep.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command, log_path, cwd, timeout, env=None):
    print(json.dumps({'command': command, 'log': str(log_path)}), flush=True)
    with log_path.open('w', buffering=1) as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=8)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=8)
            raise
    print(log_path.read_text(errors='replace')[-10000:], flush=True)
    if code:
        raise RuntimeError(f'Experiment process exited {code}; see {log_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--ct', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        gpu_names = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, timeout=10).splitlines()
        if len(gpu_names) != 4 or any('L4' not in name for name in gpu_names):
            raise RuntimeError('This four-rank experiment requires four L4 GPUs: ' + repr(gpu_names))
        # Small training wheels are already saved beside the CPU NF4 cache.
        manifest = json.loads((args.cache.parent / 'training-wheels-manifest.json').read_text())
        wheels = []
        for item in manifest['wheels']:
            if item['name'].startswith(('peft-', 'accelerate-', 'bitsandbytes-')):
                path = args.cache.parent / 'training-wheels' / item['name']
                if sha256(path) != item['sha256']:
                    raise RuntimeError('Cached training wheel differs: ' + item['name'])
                wheels.append(str(path))
        if len(wheels) != 3:
            raise RuntimeError('Expected the three cached P007 training wheels')
        run_logged([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
                    '--force-reinstall', *wheels], args.out / 'training-dependencies.log',
                   args.bundle, 180)

        sys.path.insert(0, str(args.bundle))
        from training.common import load_config, load_encoded_jsonl
        data_path = args.data or args.bundle / 'research/009/training-data-20261018/encoded-train-12b.jsonl'
        rows, data_info = load_encoded_jsonl(data_path, 8192)
        groups = [row['meta'].get('group', {}) for row in rows]
        max_length = max(len(row['input_ids']) for row in rows)
        cfg = load_config(args.bundle / 'training/configs/sft_12b_qat_nf4.yaml')
        steps = 2 * math.ceil(len(rows) / 4)
        cfg.update({
            'model_name_or_path': str(args.cache),
            'tokenizer_name_or_path': str(args.ct),
            'model_source_manifest': str(args.bundle / 'training/configs/gemma4_12b_sources.json'),
            'chat_template_path': str(args.bundle / 'training/configs/chat_templates/gemma4_12b_ct_v2.jinja'),
            'tool_schema_path': str(args.bundle / 'training/configs/harness_tools.json'),
            'device': 'cuda', 'micro_batch_size': 1, 'gradient_accumulation_steps': 1,
            'epochs': 2, 'learning_rate': 0.00002, 'warmup_ratio': 0.0,
            'max_sequence_length': min(8192, 128 * math.ceil(max_length / 128)),
            'max_steps': steps, 'checkpoint_every_n_steps': steps,
            'seed': 20261018,
        })
        cfg_path = args.out / 'sft-config.json'
        cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
        (args.out / 'data-summary.json').write_text(json.dumps({
            'data': data_info, 'groups': groups,
            'input_lengths': [len(row['input_ids']) for row in rows],
            'planned_optimizer_steps': steps, 'world_size': 4,
            'objective': 'two epochs of selected-action SFT on distinct replayed train issues',
        }, indent=2) + '\n')
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
                   PYTORCH_ALLOC_CONF='expandable_segments:True')
        env['PYTHONPATH'] = str(args.bundle)
        train_out = args.out / 'train-run'
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=4', '--max_restarts=0', '--monitor_interval=1',
                   '-m', 'training.train_ddp', '--config', str(cfg_path),
                   '--data', str(data_path), '--out', str(train_out)]
        train_start = time.monotonic()
        run_logged(command, args.out / 'ddp-training.log', args.bundle, args.timeout, env)
        training_seconds = time.monotonic() - train_start
        run_manifest = json.loads((train_out / 'training_manifest.json').read_text())
        # An independent serving package; keep optimizer checkpoints in train-run.
        export_dir = args.out / 'adapter'
        run_logged([sys.executable, '-m', 'training.export_peft',
                    '--adapter-dir', str(train_out / 'adapter'),
                    '--destination', str(export_dir), '--config', str(cfg_path),
                    '--training-manifest', str(train_out / 'training_manifest.json'),
                    '--out', str(args.out / 'export-manifest.json')],
                   args.out / 'export.log', args.bundle, 120, env)
        rank_results = [json.loads((train_out / f'rank-{rank:03d}-result.json').read_text())
                        for rank in range(4)]
        summary = {
            'status': 'completed', 'data_sha256': data_info['sha256'],
            'rows': len(rows), 'optimizer_steps': run_manifest['metrics']['optimizer_steps'],
            'world_size': 4, 'training_process_wall_seconds': training_seconds,
            'stage_wall_seconds': time.monotonic() - start,
            'metrics': run_manifest['metrics'],
            'peak_allocated_bytes_by_rank': [r['peak_cuda_allocated_bytes'] for r in rank_results],
            'adapter_dir': str(export_dir),
            'adapter_sha256': sha256(export_dir / 'adapter_model.safetensors'),
            'optimizer_checkpoint_retained': run_manifest['metrics']['last_checkpoint'],
            'repair_gain': None,
        }
        (args.out / 'stage-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    except BaseException:
        (args.out / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
