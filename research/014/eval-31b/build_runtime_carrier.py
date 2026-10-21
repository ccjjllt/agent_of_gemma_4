"""Freeze the P014 runtime repair into one uploaded Kaggle inference script.

Uses the already executed worker and coordinator, with only the frozen runtime-v2
arm on the original eight developer tasks. This does not launch compute.
"""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

from build_carrier import BOOTSTRAP


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    panel_path = ROOT / 'research/014/panel/material/agent-tasks.json'
    doc = json.loads(panel_path.read_text())
    tasks = doc if isinstance(doc, list) else doc['tasks']
    candidate = ROOT / 'research/014/candidate/runtime-v2/agent'
    arm = 'p014_single_v2_runtime'
    bundles = {arm: {
        str(path.relative_to(candidate)): path.read_text()
        for path in sorted(candidate.rglob('*')) if path.is_file()
    }}
    settings = {
        'study': '31b-runtime-v2',
        'model_alias': 'gemma-4-31b-it-qat-w4a16-ct',
        'model_path_pattern': 'gemma-4-31b-it-qat-w4a16-ct/2',
        'task_ids': [task['instance_id'] for task in tasks][:8],
        'task_count': 8,
        'budgets': {
            'timeout_seconds': 60, 'max_tool_calls': 28,
            'max_time_minutes': 5, 'max_turns': 40,
        },
        'concurrency': 4,
        'wall_seconds': 2700,
        'attempt_seconds': 480,
        'code_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'quota_estimate_upper_hours': 3000 * 2 / 3600,
        'purpose': 'runtime repair on the same frozen eight developer tasks; '
                   'single arm; no new independent repair-gain claim',
    }
    worker = (HERE / 'attempt_worker.py').read_text()
    coordinator = (HERE / 'coordinator.py').read_text()
    source = (
        '# Frozen P014 runtime repair inference; no competition submission.\n'
        + BOOTSTRAP
        + '\nSETTINGS = ' + repr(settings)
        + '\nBUNDLE = ' + repr(bundles)
        + '\nWORKER_SOURCE = ' + repr(worker) + '\n'
        + coordinator
    )
    ast.parse(source)
    kernel = HERE / 'kernel-runtime-v2'
    kernel.mkdir(exist_ok=True)
    (kernel / 'run.py').write_text(source)
    metadata = json.loads((HERE / 'kernel/kernel-metadata.json').read_text())
    metadata.update(
        id='zhuating/p014-31b-runtime-v2', title='P014 31B Runtime V2',
        code_file='run.py',
        kernel_sources=['zhuating/p014-frozen-eight-material-cpu'])
    (kernel / 'kernel-metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    manifest = {
        'settings': settings,
        'platform_timeout_seconds': 3000,
        'planned_attempts': 8,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'worker_sha256': hashlib.sha256(worker.encode()).hexdigest(),
        'coordinator_sha256': hashlib.sha256(coordinator.encode()).hexdigest(),
        'agent_task_sha256': hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        'members': {
            name: {rel: hashlib.sha256(text.encode()).hexdigest()
                   for rel, text in files.items()}
            for name, files in bundles.items()
        },
        'source_bytes': len(source.encode()),
        'launch_command': [
            'kaggle', 'kernels', 'push', '-p', str(kernel.relative_to(ROOT)),
            '-t', '3000', '--accelerator', 'NvidiaL4',
        ],
        'research_result': None,
    }
    (HERE / 'run-manifest-runtime-v2.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({
        'kernel': str(kernel), 'bytes': manifest['source_bytes'],
        'sha256': manifest['source_sha256'], 'task_ids': settings['task_ids'],
        'planned_attempts': 8, 'budgets': settings['budgets'],
        'coordinator_seconds': 2700, 'platform_seconds': 3000,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
