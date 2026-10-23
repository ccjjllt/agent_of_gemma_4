#!/usr/bin/env python3
"""One-shot, CPU-only backup after the current P016 training invocation exits.

Run detached on the Pod. This does not stop training or the Pod, schedule another
job, download model weights, or transfer training files to the operator's Mac.
Only the explicitly supplied temporary rclone config is removed on termination.
"""
import argparse
import configparser
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid


RUN_NAME = 'main-sft-short-v1'
BUNDLE_NAME = 'smith-pilot-calibration-v5'
CHECKPOINT_FORMAT = 'p016-fsdp1-local-optimizer-v1'
ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors')
CHECKPOINT_FILES = tuple('adapter/' + name for name in ADAPTER_FILES) + tuple(
    'rank-%03d-state.pt' % rank for rank in range(4))


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def facts(path):
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            sha.update(block)
            md5.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def child(base, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe relative path in source manifest')
    path = base / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError('Missing or unsafe source file: ' + str(path))
    return path


def match(path, expected):
    actual = facts(path)
    if actual['bytes'] != expected['bytes'] or actual['sha256'] != expected['sha256']:
        raise ValueError('Source hash/size mismatch: ' + str(path))
    return actual


def process_info(pid):
    try:
        base = Path('/proc') / str(pid)
        fields = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'argv': (base / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def out_argument(argv):
    if '--out' in argv:
        index = argv.index('--out')
        return argv[index + 1] if index + 1 < len(argv) else None
    return next((arg[6:] for arg in argv if arg.startswith('--out=')), None)


def active_processes(run, launcher_pid, start_ticks):
    active = {}
    launcher = process_info(launcher_pid)
    if launcher and launcher['start_ticks'] == start_ticks:
        active[launcher_pid] = launcher
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        info = process_info(int(path.name))
        if not info:
            continue
        argv = info['argv']
        is_trainer = ('training.train_fsdp' in argv or 'torch.distributed.run' in argv
                      or any(arg.endswith('/run_main_short.py') for arg in argv))
        if is_trainer and out_argument(argv) == str(run):
            active[info['pid']] = info
    return [{'pid': value['pid'], 'start_ticks': value['start_ticks']}
            for value in active.values()]


def await_exit(args, launch, run, state):
    # Ten minutes after the launcher's own deadline is a bounded wait limit;
    # an unexpectedly active training process prevents any packaging/upload.
    latest = datetime.fromisoformat(launch['latest_end_utc']).timestamp() + 600
    while True:
        active = active_processes(run, args.launcher_pid, args.start_ticks)
        reason = 'training_processes_exited' if not active else None
        if active and time.time() >= latest:
            reason = 'wait_timeout_training_still_active'
        write(state / 'status.json', {'state': 'snapshot' if reason else 'waiting_for_training_exit',
              'updated_utc': now(), 'active_training_processes': active,
              'wait_deadline_utc': datetime.fromtimestamp(latest, timezone.utc).isoformat(),
              'reason': reason, 'pod_action': 'keep_running'})
        if reason == 'wait_timeout_training_still_active':
            raise TimeoutError('Training still active after backup wait deadline; no packaging or upload attempted')
        if reason:
            return reason, active
        time.sleep(30)


def build_backup(args, launch, reason, active):
    root, state = args.root, args.state_dir
    run = root / 'runs' / RUN_NAME
    snapshot = state / 'snapshot'
    snapshot.mkdir(exist_ok=True)
    sources, recorded, problems, checkpoints = {}, {}, [], []

    def add(path, archive_name, expected=None, mutable=False):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or nonregular backup source: ' + str(path))
        if path.resolve() == args.drive_config.resolve():
            raise ValueError('Credentials must never enter the archive')
        if mutable:
            # Copy only the current byte count to make a bounded local snapshot,
            # even if a separate telemetry writer is still appending to logs.
            target = snapshot / archive_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open('rb') as source, target.open('wb') as destination:
                remaining = os.fstat(source.fileno()).st_size
                while remaining:
                    block = source.read(min(4 * 1024 * 1024, remaining))
                    if not block:
                        break
                    destination.write(block)
                    remaining -= len(block)
            path = target
        value = match(path, expected) if expected else facts(path)
        sources[archive_name] = path
        recorded[archive_name] = value
        return value

    def run_name(path):
        return 'p016/runs/' + RUN_NAME + '/' + str(path.relative_to(run))

    signature = read(run / 'run_config.json')['signature'] if (run / 'run_config.json').is_file() else None
    for directory in sorted(run.glob('checkpoint-*')):
        if not re.fullmatch(r'checkpoint-\d{6}', directory.name) or not (directory / 'complete.json').is_file():
            continue
        try:
            manifest = read(directory / 'complete.json')
            step = int(directory.name.rsplit('-', 1)[1])
            if (manifest.get('format') != CHECKPOINT_FORMAT or manifest.get('world_size') != 4
                    or manifest.get('sampler') != 'length-grouped-batches-v1'
                    or manifest.get('signature') != signature
                    or manifest['progress']['global_step'] != step
                    or manifest['progress']['total_steps'] != args.planned_steps
                    or manifest['progress']['data_rows'] != 537
                    or not 0 < step <= args.planned_steps
                    or set(manifest['files']) != set(CHECKPOINT_FILES)):
                raise ValueError('Checkpoint structure/signature/progress mismatch')
            validated = {}
            for relative in CHECKPOINT_FILES:
                path = child(directory, relative)
                validated[run_name(path)] = (path, match(path, manifest['files'][relative]))
            for name, (path, value) in validated.items():
                sources[name], recorded[name] = path, value
            add(directory / 'complete.json', run_name(directory / 'complete.json'))
            checkpoints.append(step)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append('%s excluded: %s' % (directory.name, exc))

    completed = False
    try:
        if active:
            raise ValueError('Training processes still active at backup wait deadline')
        manifest = read(run / 'training_manifest.json')
        metrics = manifest['metrics']
        if (manifest.get('status') != 'training_completed' or manifest.get('mode') != 'fsdp_qlora'
                or metrics['optimizer_steps'] != args.planned_steps or metrics['world_size'] != 4
                or manifest['signature'] != signature or args.planned_steps not in checkpoints
                or Path(metrics['last_checkpoint']) != run / ('checkpoint-%06d' % args.planned_steps)):
            raise ValueError('Final training manifest does not describe the planned completed run')
        final_cp = read(run / ('checkpoint-%06d' % args.planned_steps) / 'complete.json')
        if (metrics['global_input_tokens'] != final_cp['progress']['global_input_tokens']
                or metrics['global_selected_tokens'] != final_cp['progress']['global_selected_tokens']):
            raise ValueError('Final token counts disagree with committed checkpoint')
        final_sources = []
        for name in ADAPTER_FILES:
            expected = manifest['adapter']['files'][name]
            path = child(run / 'adapter', name)
            match(path, expected)
            if expected != final_cp['files']['adapter/' + name]:
                raise ValueError('Final adapter differs from checkpoint adapter')
            final_sources.append((path, expected))
        for rank in range(4):
            result = read(run / ('rank-%03d-result.json' % rank))
            if result['rank'] != rank or any(result[key] != metrics[key] for key in (
                    'optimizer_steps', 'world_size', 'global_input_tokens',
                    'global_selected_tokens', 'last_checkpoint')):
                raise ValueError('Rank result disagrees with final training manifest')
        for path, expected in final_sources:
            add(path, run_name(path), expected)
        completed = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append('Final completion not established: ' + str(exc))

    # Explicit allowlist: no training-state temporary directories, environment,
    # model weights, rclone/OAuth files, or authorization logs are traversed.
    for pattern in ('run_config.json', 'training_manifest.json', 'rank-???-result.json',
                    'metrics*.jsonl', 'resume-config-*.json'):
        for path in sorted(run.glob(pattern)):
            add(path, run_name(path), mutable=True)
    for name in (RUN_NAME + '-launch.json', RUN_NAME + '.log', RUN_NAME + '-gpu.csv'):
        path = root / 'runs' / name
        if path.is_file():
            add(path, 'p016/runs/' + name, mutable=True)
    bundle = root / BUNDLE_NAME
    bundle_manifest = read(bundle / 'bundle-manifest.json')
    for relative, expected in bundle_manifest['files'].items():
        add(child(bundle, relative), 'p016/' + BUNDLE_NAME + '/' + relative, expected)
    add(bundle / 'bundle-manifest.json', 'p016/' + BUNDLE_NAME + '/bundle-manifest.json')
    corpus = root / RUN_NAME
    corpus_manifest = read(corpus / 'manifest.json')
    add(corpus / 'train.jsonl', 'p016/' + RUN_NAME + '/train.jsonl',
        {'bytes': corpus_manifest['data_bytes'], 'sha256': corpus_manifest['data_sha256']})
    for relative in ('native-tools.json', 'proposed-config.json'):
        add(child(corpus, relative), 'p016/' + RUN_NAME + '/' + relative,
            corpus_manifest['packaged_files'][relative])
    add(corpus / 'manifest.json', 'p016/' + RUN_NAME + '/manifest.json')
    add(root / 'run_main_short.py', 'p016/run_main_short.py')
    for name in ('pip-freeze.txt', 'runtime-imports.json', 'nvidia-topology.txt'):
        path = root / 'setup' / name
        if path.is_file():
            add(path, 'p016/setup/' + name, mutable=True)
        else:
            problems.append('Optional environment record absent: ' + name)
    model = root.parent / 'models/p005-cpu-nf4-cache/qat-nf4'
    for name in ('cache-manifest.json', 'config.json', 'generation_config.json', 'model.safetensors.index.json'):
        path = model / name
        if path.is_file():
            add(path, 'models/p005-cpu-nf4-cache/qat-nf4/' + name)
        elif name == 'cache-manifest.json':
            problems.append('NF4 cache manifest absent; recover base identity from run_config')

    kind = 'final' if completed else 'recovery'
    backup_name = RUN_NAME + '_' + kind + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    metadata = {'schema': 'p016-direct-drive-backup-v1', 'created_utc': now(),
                'backup_kind': kind, 'training_completion_verified': completed,
                'planned_optimizer_steps': args.planned_steps, 'committed_steps_included': checkpoints,
                'latest_recoverable_step': max(checkpoints) if checkpoints else None,
                'wait_reason': reason, 'active_processes_at_snapshot': active,
                'signature': signature, 'problems': problems, 'files': recorded,
                'base_weights_included': False, 'credentials_included': False,
                'pod_action': 'keep_running', 'source_launch': launch}
    manifest_path = state / 'backup-manifest.json'
    write(manifest_path, metadata)
    instructions = state / 'RESTORE.txt'
    instructions.write_text(
        'P016 direct Runpod to Google Drive backup\n'
        'Backup kind: %s. Latest recoverable optimizer step: %s.\n' % (kind, metadata['latest_recoverable_step'])
        + 'Extract under /workspace (archive contains p016/ and models/ metadata).\n'
        'The backup contains every hash-verified committed checkpoint available at snapshot time.\n'
        'Restore the separately cached official NF4 base from the recorded cache manifest; base weights are not included.\n'
        'Rebuild the recorded Python environment. Full optimizer/RNG resume requires the same four-GPU FSDP topology and package versions.\n'
        'Pass the latest included checkpoint to run_main_short.py --resume with a separately authorized GPU budget.\n'
        'A recovery backup does not establish training completion. A final backup does not establish an evaluation score gain.\n'
        'backup-manifest.json records SHA256/MD5/bytes for every original payload file; it does not hash itself or this explanatory text.\n')
    archive = state / (backup_name + '.tar')
    required = sum(value['bytes'] for value in recorded.values()) + 128 * 1024 * 1024
    if shutil.disk_usage(state).free < required:
        raise OSError('Insufficient free space for a complete backup archive')
    temporary = archive.with_suffix('.tar.partial')
    with tarfile.open(temporary, 'x') as output:
        for name, path in sorted(sources.items()):
            output.add(path, arcname=name, recursive=False)
        output.add(manifest_path, arcname='backup-manifest.json', recursive=False)
        output.add(instructions, arcname='RESTORE.txt', recursive=False)
    os.replace(temporary, archive)
    # Re-read the actual tar payload so the receipt covers bytes that will be
    # uploaded, including any unexpected source mutation during archive writing.
    with tarfile.open(archive, 'r') as packed:
        for member in packed:
            if member.name not in recorded:
                continue
            sha = hashlib.sha256()
            with packed.extractfile(member) as handle:
                for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                    sha.update(block)
            if member.size != recorded[member.name]['bytes'] or sha.hexdigest() != recorded[member.name]['sha256']:
                raise ValueError('Archived payload changed: ' + member.name)
    return backup_name, metadata, [archive, manifest_path, instructions]


def upload(args, folder_name, files):
    config = configparser.ConfigParser(interpolation=None)
    config.read(args.drive_config)
    remotes = [name for name in config.sections() if config[name].get('type') == 'drive']
    remote = args.drive_remote or (remotes[0] if len(remotes) == 1 else None)
    if remote not in remotes:
        raise ValueError('Supply a valid --drive-remote or a temporary config with exactly one Drive remote')
    base = [args.rclone, '--config', str(args.drive_config), '--drive-root-folder-id', args.drive_parent_id,
            '--retries', '1', '--low-level-retries', '1', '--contimeout', '20s', '--timeout', '90s',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--tpslimit-burst', '1',
            '--stats', '0', '--log-level', 'ERROR']
    upload_deadline = time.monotonic() + 7200

    def command(arguments, timeout=180):
        for attempt in range(1, 4):
            remaining = upload_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Two-hour backup upload deadline exhausted')
            try:
                result = subprocess.run(base + arguments, capture_output=True, text=True,
                                        timeout=min(timeout, remaining), check=False)
                if result.returncode == 0:
                    return result.stdout
                error = result.stderr[-4000:]
            except subprocess.TimeoutExpired:
                error = 'rclone command timed out'
            # Do not log the OAuth config or the full command line. Standard
            # rclone errors are retained separately and never enter the archive.
            error = re.sub(r'(?i)(access_token|refresh_token|client_secret)([\s"=:]+)[^\s",}]+',
                           r'\1\2[REDACTED]', error)
            with (args.state_dir / 'upload-errors.log').open('a') as handle:
                handle.write('%s attempt %d command %s: %s\n' % (now(), attempt, arguments[0], error))
            if attempt < 3:
                time.sleep(min(60, max(0, upload_deadline - time.monotonic())))
        raise RuntimeError('rclone %s failed after three bounded attempts; see upload-errors.log' % arguments[0])

    destination = remote + ':' + folder_name
    command(['mkdir', destination])
    verified = []
    for path in files:
        local = facts(path)
        target = destination + '/' + path.name
        command(['copyto', str(path), target, '--immutable', '--checksum'], timeout=1800)
        cloud = json.loads(command(['lsjson', target, '--stat', '--hash']))
        hashes = {key.lower().strip(): value.lower() for key, value in cloud.get('Hashes', {}).items()}
        if cloud.get('Size') != local['bytes'] or hashes.get('md5') != local['md5']:
            raise ValueError('Google Drive size/MD5 did not match uploaded file: ' + path.name)
        verified.append({'name': path.name, **local, 'drive_file_id': cloud.get('ID'),
                         'cloud_size_and_md5_verified': True})
    folder = json.loads(command(['lsjson', destination, '--stat']))
    if not folder.get('IsDir') or not folder.get('ID'):
        raise ValueError('Could not establish the uploaded Google Drive folder ID')
    return {'drive_folder_id': folder['ID'],
            'drive_folder_url': 'https://drive.google.com/drive/folders/' + folder['ID'],
            'verified_cloud_files': verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/p016'))
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--drive-config', type=Path, required=True)
    parser.add_argument('--drive-parent-id', required=True)
    parser.add_argument('--drive-remote')
    parser.add_argument('--rclone', default='rclone')
    parser.add_argument('--launcher-pid', type=int, required=True)
    parser.add_argument('--start-ticks', type=int, required=True)
    parser.add_argument('--planned-steps', type=int, default=68)
    args = parser.parse_args()
    os.umask(0o077)
    args.root = args.root.resolve()
    args.state_dir = args.state_dir.resolve()
    args.drive_config = args.drive_config.absolute()
    args.state_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    lock = (args.state_dir / 'backup.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another backup process owns this state directory; exiting without deleting its credentials.', flush=True)
        return 3
    if (args.state_dir / 'result.json').exists():
        previous = read(args.state_dir / 'result.json')
        print('This one-shot backup already ended; preserving its existing receipt.', flush=True)
        return 0 if previous.get('state') == 'uploaded_and_verified' else 1
    result = {'state': 'failed', 'started_utc': now(), 'pod_action': 'keep_running'}
    exit_code = 1
    try:
        if args.planned_steps != 68 or args.launcher_pid <= 0 or args.start_ticks <= 0:
            raise ValueError('This backup is bound to the current 68-step main training invocation')
        if not args.drive_config.is_file() or args.drive_config.is_symlink():
            raise ValueError('A regular temporary Drive config is required')
        os.chmod(args.drive_config, 0o600)
        launch = read(args.root / 'runs' / (RUN_NAME + '-launch.json'))
        run = args.root / 'runs' / RUN_NAME
        if (launch['pid'] != args.launcher_pid or launch['output'] != str(run)
                or launch['planned_optimizer_steps'] != args.planned_steps):
            raise ValueError('Launch receipt does not match the requested training invocation')
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
        reason, active = await_exit(args, launch, run, args.state_dir)
        name, metadata, files = build_backup(args, launch, reason, active)
        result.update({'backup_kind': metadata['backup_kind'], 'training_completion_verified': metadata['training_completion_verified'],
                       'latest_recoverable_step': metadata['latest_recoverable_step'],
                       'committed_steps_included': metadata['committed_steps_included'],
                       'local_server_files': [str(path) for path in files]})
        write(args.state_dir / 'status.json', {**result, 'state': 'uploading', 'updated_utc': now()})
        result.update(upload(args, name, files))
        result['state'] = 'uploaded_and_verified'
        exit_code = 0
    except BaseException as exc:
        result['error'] = '%s: %s' % (type(exc).__name__, exc)
    finally:
        try:
            args.drive_config.unlink(missing_ok=True)
            result['temporary_drive_config_removed'] = not args.drive_config.exists()
        except OSError as exc:
            result['temporary_drive_config_removed'] = False
            result['cleanup_error'] = str(exc)
            exit_code = 1
        result['finished_utc'] = now()
        write(args.state_dir / 'result.json', result)
        print(json.dumps(result, sort_keys=True), flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
