"""Stage 4b: replay 38 stage1c tasks (40 minus f3-replace2 2) to f3-replace2.

Inputs (read-only): replay/artifacts/f3-replace/<key>-task.json,
  replay/artifacts/f3-replace/<key>-source, replay/f3r-<key>-reviewed-recipe.json
Output (write-only): replay/artifacts/f3-replace2/<key>-run-001 (+ .log)
Pool mapping per acquisition/f3-new/replay_f3replace.py:19-29, copy mode.
Never touches f1/f2/f4/f5, S600, or f3-replace outputs.
Usage: python3 replay_f3replace2_38.py [workers]
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f3-replace'
ART2 = REPLAY / 'artifacts/f3-replace2'
ART2.mkdir(parents=True, exist_ok=True)

POOL_CONF = {
    'f3r-dvc': {'venv': '.venv-dvc/lib/python3.12/site-packages', 'python': None, 'layout': '.'},
    'f3r-conan': {'venv': '.venv-conan/lib/python3.12/site-packages', 'python': None, 'layout': '.'},
    'f3r-cp': {'venv': '.venv-r4-cloudpickle-task/lib/python3.11/site-packages',
               'python': '/Users/zhaohuaxiang/.local/share/uv/python/'
                         'cpython-3.11.15-macos-aarch64-none/bin/python3.11',
               'layout': '.'},
    'f3r-tg': {'venv': '.venv-typeguard/lib/python3.12/site-packages', 'python': None,
               'layout': 'src'},
    'f3r-dask': {'venv': '.venv-dynamic-dask/lib/python3.12/site-packages', 'python': None,
                 'layout': '.'},
}

RUNNER = REPLAY / '.venv/bin/python'

# 38 = stage1c 40 minus the 2 already fixed in f3-replace2
with open(ACQ / 'f3-replace/stage1c-manifest.json') as f:
    _stage = json.load(f)
KEYS38 = [t['key'] for t in _stage['tasks'] if t['key'] not in ('f3r-dvc-07', 'f3r-conan-05')]
assert len(KEYS38) == 38, len(KEYS38)


def pool_of(key: str) -> str:
    return next(p for p in POOL_CONF if key.startswith(p))


def run_one(key: str):
    conf = POOL_CONF[pool_of(key)]
    adapted = ART / f'{key}-task.json'
    ws = ART / f'{key}-source'
    recipe = REPLAY / f'f3r-{key}-reviewed-recipe.json'
    out = ART2 / f'{key}-run-001'
    log = ART2 / f'{key}-run-001.log'
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
    extra = sys.argv[2:] if len(sys.argv) > 2 else []
    keys = extra or KEYS38
    print(f'replaying {len(keys)} to f3-replace2 with {workers} workers', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_one, keys))
    print('all done', flush=True)


if __name__ == '__main__':
    main()
