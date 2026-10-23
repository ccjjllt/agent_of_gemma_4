#!/usr/bin/env python3
"""Explicitly launch the frozen 537-issue, 68-step short-corpus SFT stage.

Requires an approved GPU budget supplied as --wall-seconds. A wall limit stops
this invocation; it never rewrites the frozen optimizer schedule. Repeated
invocations share the user-approved total budget, accounted for by the operator.
No installation, download, calibration, or GPU work occurs on import.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


# Frozen receipts: generated/smith-pilot-calibration-v5/build-manifest.json and
# main-short-corpus-package.json. The pinned manifests bind their payload files.
BUNDLE_MANIFEST_SHA256 = '049d37bd6e0ed410308d307409733b237244bc626212867f5bf80fd48a37c5dc'
CORPUS_MANIFEST_SHA256 = '068514ff4557adc1de46d2c3e97ac391bbcb6976570b34998e0b54ef55b1a38f'
TOTAL_STEPS = 68


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def match_file(path, expected_hash, expected_bytes=None):
    if (not path.is_file()
            or (expected_bytes is not None and path.stat().st_size != expected_bytes)
            or sha256(path) != expected_hash):
        raise ValueError('Frozen file missing or changed: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'corpus', 'model', 'tokenizer', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after-checkpoint', type=int)
    args = parser.parse_args()
    if not 60 <= args.wall_seconds <= 28800:
        parser.error('--wall-seconds must be an explicit 60..28800 second limit')
    bundle, corpus, model, tokenizer, output = (
        getattr(args, name).expanduser().resolve()
        for name in ('bundle', 'corpus', 'model', 'tokenizer', 'out'))
    match_file(bundle / 'bundle-manifest.json', BUNDLE_MANIFEST_SHA256)
    manifest = read(bundle / 'bundle-manifest.json')
    for relative, facts in manifest['files'].items():
        match_file(bundle / relative, facts['sha256'], facts['bytes'])
    match_file(corpus / 'manifest.json', CORPUS_MANIFEST_SHA256)
    data_manifest = read(corpus / 'manifest.json')
    match_file(corpus / 'train.jsonl', data_manifest['data_sha256'], data_manifest['data_bytes'])
    for relative in ('proposed-config.json', 'native-tools.json'):
        facts = data_manifest['packaged_files'][relative]
        match_file(corpus / relative, facts['sha256'], facts['bytes'])
    if not model.is_dir() or not tokenizer.is_dir():
        raise ValueError('Bind existing local model and tokenizer directories')
    resume = args.resume.expanduser().resolve() if args.resume else None
    if resume:
        if resume.parent != output or not (resume / 'complete.json').is_file():
            raise ValueError('--resume must be a committed checkpoint inside --out')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Initial --out must be absent or empty; use --resume to continue')
    stop_step = args.stop_after_checkpoint
    if stop_step is not None:
        interval = read(corpus / 'proposed-config.json')['checkpoint_every_n_steps']
        if not 0 < stop_step < TOTAL_STEPS or stop_step % interval:
            raise ValueError('Controlled stop must name a scheduled checkpoint before step 68')
        if resume and stop_step <= read(resume / 'complete.json')['progress']['global_step']:
            raise ValueError('Controlled stop must follow the resumed checkpoint')

    # Reuse the frozen launcher's ownership tracking and cleanup implementation.
    spec = importlib.util.spec_from_file_location('p016_frozen_launcher', bundle / 'prepare_gpu_run.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    launcher.process_snapshot()
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc-per-node=4', '--max-restarts=0', '-m', 'training.train_fsdp',
        '--config', str(corpus / 'proposed-config.json'),
        '--data', str(corpus / 'train.jsonl'), '--out', str(output),
        '--model', str(model), '--tokenizer', str(tokenizer),
        '--tools', str(corpus / 'native-tools.json'),
        '--chat-template', str(tokenizer / 'chat_template.jinja')]
    if resume:
        command += ['--resume', str(resume)]
    env = dict(os.environ, **launcher.RUN_ENV, PYTHONPATH=str(bundle),
               NCCL_P2P_DISABLE='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=bundle, env=env, start_new_session=True)
    owned = launcher.OwnedProcesses(process)
    deadline = started + args.wall_seconds
    print(json.dumps({'event': 'main_sft_launch', 'pid': process.pid,
        'unix_time': time.time(), 'command': command, 'wall_seconds_limit': args.wall_seconds,
        'planned_total_steps': TOTAL_STEPS, 'independent_issues': 537,
        'resume_from': str(resume) if resume else None,
        'stop_after_checkpoint': stop_step,
        'runtime_environment': {**launcher.RUN_ENV, 'NCCL_P2P_DISABLE': '1',
                                'PYTORCH_ALLOC_CONF': 'expandable_segments:True'}}, indent=2), flush=True)
    reason = 'process_exit'
    try:
        while process.poll() is None:
            owned.refresh()
            if time.monotonic() >= deadline - 15:
                reason = 'wall_limit'
                break
            if stop_step is not None and (output / ('checkpoint-%06d' % stop_step) / 'complete.json').is_file():
                reason = 'explicit_committed_checkpoint_stop'
                break
            time.sleep(.5)
    except KeyboardInterrupt:
        reason = 'user_interrupt'
    finally:
        launcher.stop_group(process, owned, grace_seconds=max(0, min(15, deadline - time.monotonic())))
    print(json.dumps({'event': 'main_sft_termination', 'reason': reason,
        'child_returncode': process.returncode, 'wall_seconds': time.monotonic() - started,
        'unix_time': time.time()}), flush=True)
    return (process.returncode if reason == 'process_exit' else
            75 if reason == 'explicit_committed_checkpoint_stop' else
            124 if reason == 'wall_limit' else 130)


if __name__ == '__main__':
    sys.exit(main())
