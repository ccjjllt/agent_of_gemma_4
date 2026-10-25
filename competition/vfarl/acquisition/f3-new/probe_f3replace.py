"""Stage 3 probe: run each recipe observation against buggy (expect FAIL)
and fixed (expect PASS) trees using pool pythons. No replay harness.

Usage: python3 probe_f3replace.py [key ...]
Output: acquisition/f3-new/f3-replace/probe-results.json
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
OUT = ACQ / 'f3-replace'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f3-replace'

POOL_PY = {
    'f3r-dvc': REPLAY / '.venv-dvc/bin/python',
    'f3r-conan': REPLAY / '.venv-conan/bin/python',
    'f3r-cp': REPLAY / '.venv-r4-cloudpickle-task/bin/python',
    'f3r-tg': REPLAY / '.venv-typeguard/bin/python',
    'f3r-dask': REPLAY / '.venv-dynamic-dask/bin/python',
}

SCRATCH = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/f3probe')
SCRATCH.mkdir(parents=True, exist_ok=True)


def pool_of(key: str) -> str:
    return next(p for p in POOL_PY if key.startswith(p))


def run_obs(cmd: str, cwd: Path, py: Path, extra_env=None, timeout=240):
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['DVC_TEST'] = 'true'
    env['CONAN_COLOR_DARK'] = '0'
    if extra_env:
        env.update(extra_env)
    # run with the pool python as `python3` on PATH
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout,
                       env={**env, 'PATH': f'{py.parent}:{env["PATH"]}'})
    tail = (p.stdout + '\n' + p.stderr)[-1500:]
    return p.returncode, tail


def main():
    only = set(sys.argv[1:])
    stage1c = json.loads((OUT / 'stage1c-manifest.json').read_text())
    prev_path = OUT / 'probe-results.json'
    results = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    for t in stage1c['tasks']:
        key = t['key']
        if only and key not in only:
            continue
        print(f'=== {key} ===', flush=True)
        pool = pool_of(key)
        py = POOL_PY[pool]
        recipe = json.loads((REPLAY / f'f3r-{key}-reviewed-recipe.json').read_text())
        cmds = [a['arguments']['command'] for a in recipe['actions'] if a['tool'] == 'run_command']
        assert len(cmds) == 2 and cmds[0] == cmds[1]
        cmd = cmds[0]
        ws = ART / (key + '-source')
        extra = {}
        if pool == 'f3r-tg':
            extra['PYTHONPATH'] = str(ws / 'src')
        brc, bout = run_obs(cmd, ws, py, extra)
        print(f'  buggy rc={brc}', flush=True)
        # fixed copy
        fixed = SCRATCH / f'{key}-fixed'
        if fixed.exists():
            shutil.rmtree(fixed)
        shutil.copytree(ws, fixed, symlinks=True,
                        ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
        for e in t['recipe_edits']:
            fp = fixed / e['filepath']
            text = fp.read_text()
            assert text.count(e['old_string']) == 1, f"{key} {e['filepath']} count={text.count(e['old_string'])}"
            fp.write_text(text.replace(e['old_string'], e['new_string']))
        frc, fout = run_obs(cmd, fixed, py,
                            {**extra, 'PYTHONPATH': str(fixed / 'src')} if pool == 'f3r-tg' else extra)
        print(f'  fixed rc={frc}', flush=True)
        results[key] = {'buggy_rc': brc, 'buggy_tail': bout, 'fixed_rc': frc, 'fixed_tail': fout,
                        'want_buggy_fail': brc != 0, 'want_fixed_pass': frc == 0,
                        'verdict': 'GOOD' if (brc != 0 and frc == 0) else 'NEEDS_WORK'}
        print(f"  verdict={results[key]['verdict']}", flush=True)
    (OUT / 'probe-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    good = sum(1 for v in results.values() if v['verdict'] == 'GOOD')
    print(f'\nPROBE DONE good={good}/{len(results)}', flush=True)
    for k, v in results.items():
        if v['verdict'] != 'GOOD':
            print(f'  {k}: buggy_rc={v["buggy_rc"]} fixed_rc={v["fixed_rc"]}', flush=True)


if __name__ == '__main__':
    main()
