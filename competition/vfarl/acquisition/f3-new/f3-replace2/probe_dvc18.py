"""Probe f3r-dvc-18 obs on buggy (expect FAIL) and fixed (expect PASS). Additive only."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
NEW = ROOT / 'competition/vfarl/acquisition/f3-new/f3-replace2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART2 = REPLAY / 'artifacts/f3-replace2'
PY = REPLAY / '.venv-dvc/bin/python'
SCRATCH = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/f3probe2')
SCRATCH.mkdir(parents=True, exist_ok=True)


def run_obs(cmd, cwd, extra_env=None, timeout=240):
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['DVC_TEST'] = 'true'
    if extra_env:
        env.update(extra_env)
    p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, env={**env, 'PATH': f'{PY.parent}:{env["PATH"]}'})
    tail = (p.stdout + '\n' + p.stderr)[-1500:]
    return p.returncode, tail


def main():
    key = 'f3r-dvc-18'
    recipe = json.loads((REPLAY / f'f3r2-{key}-reviewed-recipe.json').read_text())
    cmds = [a['arguments']['command'] for a in recipe['actions'] if a['tool'] == 'run_command']
    assert len(cmds) == 2 and cmds[0] == cmds[1]
    cmd = cmds[0]
    ws = ART2 / f'{key}-source'
    brc, bout = run_obs(cmd, ws)
    print(f'buggy rc={brc}', flush=True)
    print(bout[-1200:], flush=True)
    st = json.loads((NEW / f'{key}-stage1c.json').read_text())
    fixed = SCRATCH / f'{key}-fixed'
    if fixed.exists():
        shutil.rmtree(fixed)
    shutil.copytree(ws, fixed, symlinks=True,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
    for e in st['recipe_edits']:
        fp = fixed / e['filepath']
        text = fp.read_text()
        assert text.count(e['old_string']) == 1, (e['filepath'], text.count(e['old_string']))
        fp.write_text(text.replace(e['old_string'], e['new_string']))
    frc, fout = run_obs(cmd, fixed)
    print(f'fixed rc={frc}', flush=True)
    print(fout[-1200:], flush=True)
    verdict = 'GOOD' if (brc != 0 and frc == 0) else 'NEEDS_WORK'
    print(f'verdict={verdict}', flush=True)
    prev = json.loads((NEW / 'probe2-results.json').read_text()) if (NEW / 'probe2-results.json').exists() else {}
    # additive: write separate probe file, do not overwrite prior probe2-results.json
    save = {key: {'buggy_rc': brc, 'buggy_tail': bout, 'fixed_rc': frc, 'fixed_tail': fout,
                  'want_buggy_fail': brc != 0, 'want_fixed_pass': frc == 0, 'verdict': verdict}}
    (NEW / 'probe-dvc18-results.json').write_text(json.dumps(save, ensure_ascii=False, indent=2) + '\n')
    if verdict != 'GOOD':
        sys.exit(1)


if __name__ == '__main__':
    main()
