"""Stage 4b-r2: second-round dual replay for stage1c replacement candidates.

Round 1 (workers, 18:29-18:55) wrote ART/<key>-run-001 with empty .log files.
This round writes ART/<key>-run-002 (+ .log) with identical pool config so the
batch has two independent dual (baseline+repaired) wheels for stability.

Scope: F3 replacement keys only. Never touches f1/f2/f4/f5, S600,
f3-replace2/, or any run-001 evidence.
Usage: python3 replay_f3replace_newbatch_r2.py [workers]
Batch log: acquisition/f3-new/f3-replace/replay-newbatch-r2.log
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f3-replace'

POOL_CONF = {
    'f3r-dvc': {'venv': '.venv-dvc/lib/python3.12/site-packages', 'python': None, 'layout': '.'},
    'f3r-conan': {'venv': '.venv-conan/lib/python3.12/site-packages', 'python': None, 'layout': '.'},
}

# Replacement candidates with round-1 dual evidence; tg trio excluded from
# rerun (round-1 dual fails on the same strict-matcher node x3, structural).
KEYS = ['f3r-dvc-10', 'f3r-dvc-11', 'f3r-dvc-12', 'f3r-dvc-13', 'f3r-dvc-14',
        'f3r-dvc-15', 'f3r-dvc-16', 'f3r-dvc-17',
        'f3r-conan-24', 'f3r-conan-25', 'f3r-conan-27']

RUNNER = REPLAY / '.venv/bin/python'


def pool_of(key: str) -> str:
    return next(p for p in POOL_CONF if key.startswith(p))


def run_one(key: str):
    import json
    conf = POOL_CONF[pool_of(key)]
    adapted = ART / f'{key}-task.json'
    ws = ART / f'{key}-source'
    recipe = REPLAY / f'f3r-{key}-reviewed-recipe.json'
    out = ART / f'{key}-run-002'
    log = ART / f'{key}-run-002.log'
    if (out / 'run-summary.json').exists():
        return key, 'SKIP-done', 0
    cmd = [str(RUNNER), str(REPLAY / 'run_official.py'), '--task', str(adapted),
           '--workspace', str(ws), '--recipe', str(recipe), '--output', str(out),
           '--verify', '--verify-baseline', '--source-layout', conf['layout'],
           '--task-site-packages', str(REPLAY / conf['venv']),
           '--task-site-packages-mode', 'copy', '--time-minutes', '120',
           '--tool-calls', '100', '--command-timeout', '600', '--verification-timeout', '1200']
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
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    extra = sys.argv[2:] if len(sys.argv) > 2 else []
    keys = extra or KEYS
    print(f'replaying round-2 {len(keys)} with {workers} workers', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_one, keys))
    print('all done', flush=True)


if __name__ == '__main__':
    main()
