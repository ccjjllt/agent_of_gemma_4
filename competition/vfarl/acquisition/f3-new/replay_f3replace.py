"""Stage 4: replay all 40 with run_official.py --verify --verify-baseline.

Per-pool venv profile (copy mode). Parallel workers.
Usage: python3 replay_f3replace.py [workers]
Output: replay/artifacts/f3-replace/<key>-run-001 (+ .log)
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
OUT = ACQ / 'f3-replace'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f3-replace'

sys.path.insert(0, str(ACQ))
from f3replace_pool import TASKS, ACTIVE_TASKS  # noqa: E402

POOL_CONF = {
    'f3r-dvc': {'venv': '.venv-dvc/lib/python3.12/site-packages', 'python': None, 'layout': '.'},
    'f3r-conan': {'venv': '.venv-conan/lib/python3.12/site-packages', 'python': None, 'layout': '.'},
    'f3r-cp': {'venv': '.venv-r4-cloudpickle-task/lib/python3.11/site-packages',
               'python': '/Users/zhaohuaxiang/.local/share/uv/python/'
                         'cpython-3.11.15-macos-aarch64-none/bin/python3.11',
               'layout': '.'},
    'f3r-tg': {'venv': '.venv-typeguard/lib/python3.12/site-packages', 'python': None,
               'layout': 'src',
               'executables': {'mypy': 'tg-mypy-shim/mypy'}},
    'f3r-dask': {'venv': '.venv-dynamic-dask/lib/python3.12/site-packages', 'python': None,
                 'layout': '.'},
}

RUNNER = REPLAY / '.venv/bin/python'


def pool_of(key: str) -> str:
    return next(p for p in POOL_CONF if key.startswith(p))


def run_one(key: str):
    conf = POOL_CONF[pool_of(key)]
    adapted = ART / f'{key}-task.json'
    ws = ART / f'{key}-source'
    recipe = REPLAY / f'f3r-{key}-reviewed-recipe.json'
    out = ART / f'{key}-run-001'
    log = ART / f'{key}-run-001.log'
    if (out / 'run-summary.json').exists():
        return key, 'SKIP-done', 0
    cmd = [str(RUNNER), str(REPLAY / 'run_official.py'), '--task', str(adapted),
           '--workspace', str(ws), '--recipe', str(recipe), '--output', str(out),
           '--verify', '--verify-baseline', '--source-layout', conf['layout'],
           '--task-site-packages', str(REPLAY / conf['venv']),
           '--task-site-packages-mode', 'copy', '--time-minutes', '120',
           '--tool-calls', '100', '--command-timeout', '600', '--verification-timeout', '1200']
    if conf['python']:
        cmd += ['--task-python', conf['python']]
    for name, rel in conf.get('executables', {}).items():
        cmd += ['--sandbox-executable', f'{name}={REPLAY / rel}']
    t0 = time.time()
    with open(log, 'w') as s:
        r = subprocess.run(cmd, stdout=s, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = 'OK' if r.returncode == 0 else f'RC={r.returncode}'
    try:
        summary = json.loads((out / 'run-summary.json').read_text())
        status += (f" base={summary.get('baseline_resolved_by_local_official_verify_task')}"
                   f" fixed={summary.get('resolved_by_local_official_verify_task')}")
    except Exception as e:
        status += f' (no summary: {e})'
    print(f'{key}: {status} [{dt:.0f}s]', flush=True)
    return key, status, dt


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    extra = __import__('sys').argv[2:] if len(__import__('sys').argv) > 2 else []
    keys = extra or [t[0] for t in ACTIVE_TASKS()]
    print(f'replaying {len(keys)} with {workers} workers', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_one, keys))
    print('all done', flush=True)


if __name__ == '__main__':
    main()
