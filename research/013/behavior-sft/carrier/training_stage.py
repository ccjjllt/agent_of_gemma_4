"""Run eight epochs of four-L4 selected-action SFT on the frozen P013 behavior batch.

The P013 dataset manifest, not a hard-coded step cap, determines the run length:
34 independent train issues, global batch 4, 8 epochs -> 72 optimizer steps.
Uses the P007 immutable 12B NF4 cache and the existing DDP/checkpoint implementation.
New LoRA + optimizer; never resumes any older-protocol checkpoint.
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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def run_logged(command, log_path, cwd, timeout, env=None):
    """One bounded process group; keep its log and existing checkpoints on failure."""
    if timeout <= 0:
        raise TimeoutError('No remaining time for ' + str(command[0]))
    print(json.dumps({'command': command, 'log': str(log_path), 'timeout': timeout}), flush=True)
    with Path(log_path).open('a', buffering=1) as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            for sig, grace in ((signal.SIGTERM, 8), (signal.SIGKILL, 3)):
                try:
                    os.killpg(process.pid, sig)
                    process.wait(timeout=grace)
                    break
                except ProcessLookupError:
                    break
                except subprocess.TimeoutExpired:
                    pass
            raise
    print(Path(log_path).read_text(errors='replace')[-12000:], flush=True)
    if code:
        raise RuntimeError(f'Process exited {code}; retained evidence: {log_path}')


def load_frozen_data(bundle, manifest_path, minimum_issues=34, data_root=None):
    """Read only the explicitly supplied P013 behavior train file, never task datasets.

    The manifest and its encoded file travel together (same directory, e.g. a
    mounted read-only dataset). The executable bundle (training code) is passed
    separately for imports. data_root defaults to bundle (P011 layout).
    """
    from training.common import load_encoded_jsonl
    bundle = Path(bundle).resolve()
    manifest_path = Path(manifest_path).resolve()
    data_root = Path(data_root).resolve() if data_root else bundle
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'frozen' or manifest.get('split') != 'train':
        raise ValueError('Training needs an explicitly frozen train dataset manifest')
    rel = Path(manifest['encoded']['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Encoded path must be a plain relative file name')
    data_path = (manifest_path.parent / rel.name).resolve()
    if not data_path.is_relative_to(data_root):
        raise ValueError('Encoded data must sit next to its frozen manifest')
    if sha256(data_path) != manifest['encoded']['sha256']:
        raise ValueError('Frozen encoded data hash differs')
    ceiling = int(manifest['max_sequence_length'])
    if ceiling < 2 or ceiling > 8192:
        raise ValueError('This carrier admits complete sequences up to 8192; never truncate')
    rows, info = load_encoded_jsonl(data_path, ceiling)
    groups = [row['meta'].get('group', {}) for row in rows]
    if any(group.get('split') != 'train' for group in groups):
        raise ValueError('Every row must be explicitly train')
    issues = [group.get('issue') for group in groups]
    repos = [group.get('repo') for group in groups]
    if any(not isinstance(x, str) or not x for x in issues + repos):
        raise ValueError('Every row needs issue and repository identity')
    if len(set(issues)) != len(rows) or len(rows) < minimum_issues:
        raise ValueError('Need at least 34 independent issues, one complete transaction per issue')
    if info['provenance'].get('protocol_id') != 'p013-behavior-v1':
        raise ValueError('This stage trains the p013-behavior-v1 protocol only, never a mixed file')
    if any(row['meta'].get('truncated') or row['meta'].get('truncated_tool_block') for row in rows):
        raise ValueError('No truncated trajectories in the continuous SFT batch')
    for key, actual in [('rows', len(rows)), ('independent_issues', len(set(issues))),
                        ('independent_repos', len(set(repos)))]:
        if manifest.get(key) != actual:
            raise ValueError(f'Dataset manifest {key} differs from encoded records')
    for key, value in manifest['provenance'].items():
        if info['provenance'].get(key) != value:
            raise ValueError('Dataset provenance mismatch: ' + key)
    lengths = [len(row['input_ids']) for row in rows]
    summary = {
        'dataset_manifest_sha256': sha256(manifest_path), 'data': info,
        'rows': len(rows), 'independent_issues': len(set(issues)),
        'independent_repos': len(set(repos)),
        'repo_counts': {repo: repos.count(repo) for repo in sorted(set(repos))},
        'input_lengths': lengths, 'max_input_length': max(lengths),
        'epochs': 8, 'world_size': 4, 'micro_batch_size': 1,
        'gradient_accumulation_steps': 1, 'global_batch_size': 4,
        'planned_optimizer_steps': 8 * math.ceil(len(rows) / 4),
        'ddp_length_grouped_batches': True,
        'objective': 'Eight epochs, one frozen real behavior transaction per independent train issue',
        'no_repeated_padding_in_objective': True, 'no_history_truncation': True,
    }
    return data_path, rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--ct', type=Path)
    parser.add_argument('--data-root', type=Path,
                        help='Directory holding the frozen manifest + encoded file '
                             '(mounted dataset); defaults to --bundle (P011 layout)')
    parser.add_argument('--timeout', type=int, default=7200)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sys.path.insert(0, str(args.bundle))
    try:
        from training.common import load_config
        data_root = args.data_root.resolve() if args.data_root else args.bundle
        data_path, rows, summary = load_frozen_data(
            args.bundle, args.dataset_manifest, data_root=data_root)
        steps = summary['planned_optimizer_steps']
        cfg = load_config(args.bundle / 'training/configs/sft_p013_12b_behavior_ddp.yaml')
        cfg.update({
            'model_name_or_path': str(args.cache) if args.cache else 'P007_NF4_CACHE_NOT_LOADED',
            'tokenizer_name_or_path': str(args.ct) if args.ct else 'CT_V2_TOKENIZER_NOT_LOADED',
            'model_source_manifest': str(args.bundle / 'training/configs/gemma4_12b_sources.json'),
            'chat_template_path': str(args.bundle / 'training/configs/chat_templates/gemma4_12b_ct_v2.jinja'),
            'tool_schema_path': str(args.bundle / 'research/012/agent/repair_tools_v1.json'),
            'protocol_id': 'p013-behavior-v1',
            'device': 'cuda', 'micro_batch_size': 1, 'gradient_accumulation_steps': 1,
            'epochs': 8, 'learning_rate': 0.00002, 'warmup_ratio': 0.0,
            'max_sequence_length': 8192,
            'max_steps': steps, 'checkpoint_every_n_steps': min(32, max(8, steps // 4)),
            'seed': 20261020, 'ddp_length_grouped_batches': True,
        })
        if summary['max_input_length'] > cfg['max_sequence_length']:
            raise ValueError('Frozen rows exceed the configured sequence limit')
        write_json(args.out / 'data-summary.json', summary)
        write_json(args.out / 'sft-config.json', cfg)
        if args.prepare_only:
            write_json(args.out / 'stage-summary.json', {
                'status': 'prepared', **summary, 'model_weights_loaded': False,
                'optimizer_steps_executed': 0, 'diagnostic_runs': 0})
            print(json.dumps({'status': 'prepared', 'issues': len(rows), 'planned_steps': steps}), flush=True)
            return
        if args.resume_checkpoint:
            raise ValueError('P013 starts a new LoRA + optimizer; resuming any older-protocol checkpoint is rejected')
        if not args.cache or not args.ct:
            raise ValueError('Training requires the pinned cache and deployment tokenizer')
        names = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, timeout=10).splitlines()
        if len(names) != 4 or any('L4' not in name for name in names):
            raise RuntimeError('This run requires exactly four L4 GPUs: ' + repr(names))
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
                    '--force-reinstall', *wheels], args.out / 'training-dependencies.log', args.bundle, 180)
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
                   PYTORCH_ALLOC_CONF='expandable_segments:True', PYTHONPATH=str(args.bundle))
        train_out = args.out / 'train-run'
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node=4', '--max_restarts=0', '--monitor_interval=1',
                   '-m', 'training.train_ddp', '--config', str(args.out / 'sft-config.json'),
                   '--data', str(data_path), '--out', str(train_out)]
        train_started = time.monotonic()
        # Keep time for independent export and metadata even on long batches.
        available = args.timeout - (time.monotonic() - started) - 180
        run_logged(command, args.out / 'ddp-training.log', args.bundle, available, env)
        training_seconds = time.monotonic() - train_started
        run_manifest = json.loads((train_out / 'training_manifest.json').read_text())
        if run_manifest['metrics']['optimizer_steps'] != steps:
            raise RuntimeError('DDP returned before completing the manifest-derived eight epochs')
        checkpoint = Path(run_manifest['metrics']['last_checkpoint'])
        required = ['training_state.pt', 'progress.json'] + [f'rank-{r:03d}-rng.pt' for r in range(4)]
        if not all((checkpoint / name).is_file() for name in required):
            raise RuntimeError('Final resumable checkpoint is incomplete')
        export_dir = args.out / 'adapter'
        run_logged([sys.executable, '-m', 'training.export_peft', '--adapter-dir', str(train_out / 'adapter'),
                    '--destination', str(export_dir), '--config', str(args.out / 'sft-config.json'),
                    '--training-manifest', str(train_out / 'training_manifest.json'),
                    '--out', str(args.out / 'export-manifest.json')],
                   args.out / 'export.log', args.bundle, 120, env)
        rank_results = [json.loads((train_out / f'rank-{rank:03d}-result.json').read_text()) for rank in range(4)]
        write_json(args.out / 'stage-summary.json', {
            'status': 'completed', 'data_sha256': summary['data']['sha256'],
            'rows': len(rows), 'independent_issues': summary['independent_issues'],
            'independent_repos': summary['independent_repos'], 'optimizer_steps': steps, 'epochs': 8,
            'world_size': 4, 'training_process_wall_seconds': training_seconds,
            'stage_wall_seconds': time.monotonic() - started, 'metrics': run_manifest['metrics'],
            'peak_allocated_bytes_by_rank': [x['peak_cuda_allocated_bytes'] for x in rank_results],
            'adapter_dir': str(export_dir), 'adapter_sha256': sha256(export_dir / 'adapter_model.safetensors'),
            'full_optimizer_checkpoint': str(checkpoint), 'diagnostic_runs': 0, 'repair_gain': None})
        write_json(args.out / 'diagnostic-handoff.json', {
            'adapter_dir': str(export_dir), 'adapter_sha256': sha256(export_dir / 'adapter_model.safetensors'),
            'note': 'Training completed first. No model diagnosis or submission was started by this carrier. '
                    'Behavior-gain evaluation runs separately with a frozen eval plan.'})
        print(json.dumps({'status': 'completed', 'optimizer_steps': steps, 'adapter': str(export_dir)}), flush=True)
    except BaseException:
        (args.out / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
