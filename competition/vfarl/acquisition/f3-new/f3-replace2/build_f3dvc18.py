"""F3-replace2 gap-40 fix: mine 1 fresh spare f3r-dvc-18 (lowest compliant cost).

Candidate: iterative__dvc.1d6ea681.func_pm_remove_assign__nbjxqs89
- Same mirrors/upstreams as pool (swesmith/iterative__dvc.1d6ea681, iterative/dvc@1d6ea681):
  reuses cached upstream, zero new upstream downloads, zero overlap (iterative prefix
  absent from all 7154 common-ish ids).
- Small + clean: F2P=2/P2P=1, single test file tests/unit/stage/test_utils.py, no
  signal/xfail-prefix/stdio/install_test fragility.
- Production-only patch: dvc/stage/utils.py validate_kwargs (removes stage_name binding;
  buggy raises NameError, fixed validates). F2P tests create stages via validate_kwargs.
- Whitelist: non-empty Smith statement kept byte-identical (verbatim profile).
- Six tools only recipe; sequence probe GOOD expected via validate_kwargs gate obs.

Writes ONLY under acquisition/f3-new/f3-replace2/, replay/f3r2-*, replay/artifacts/f3-replace2/.
Never touches f1/f2/f4/f5, S600, f3-replace/, or prior f3-replace2 outputs (additive only).
"""
import difflib
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
OUT = ACQ / 'f3-replace'
NEW = ACQ / 'f3-replace2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f3-replace'
ART2 = REPLAY / 'artifacts/f3-replace2'
ART2.mkdir(parents=True, exist_ok=True)
REV = '77cab9055d42ab4a5c25c89a8f937096db13558e'

KEY = 'f3r-dvc-18'
IID = 'iterative__dvc.1d6ea681.func_pm_remove_assign__nbjxqs89'
REPO = 'swesmith/iterative__dvc.1d6ea681'
CANONICAL = 'iterative/dvc'
UPSTREAM_SHORT = '1d6ea681'
SYMBOL = 'validate_kwargs'
CONTRACT = 'stage_kwargs_name_gate_and_cache_scope'
MECH = ('Stage kwargs gate scope: set->named stage kwargs validate with equivalence '
        '(repeat validation of the same name yields the same binding) and isolation '
        '(other names validate independently), expire/invalidate->missing name binding '
        'is rejected (second observation differs: ok vs error), recompute->restored '
        'binding validates again; cacheable stages hash and saved entries reload.')

OBS = r"""
import os, tempfile
from types import SimpleNamespace
from dvc.stage.cache import StageCache, _can_hash
from dvc.stage.utils import validate_kwargs
def dep(protocol='local', def_path='data', h='abc'):
    d = SimpleNamespace()
    d.protocol = protocol
    d.def_path = def_path
    d.get_hash = lambda: h
    return d
def out():
    o = SimpleNamespace()
    o.protocol = 'local'
    o.def_path = 'out'
    o.get_hash = lambda: 'abc'
    o.use_cache = True
    o.persist = False
    o.is_in_repo = True
    return o
def stage(cmd='echo', deps=None, outs=None):
    s = SimpleNamespace()
    s.is_callback = False
    s.always_changed = False
    s.cmd = cmd
    s.deps = deps if deps is not None else [dep()]
    s.outs = outs if outs is not None else [out()]
    return s
assert _can_hash(stage()) is True
print("get_hit", True)
kw = validate_kwargs(cmd='echo', name='build')
assert kw['name'] == 'build'
print("equivalence", True)
kw2 = validate_kwargs(cmd='echo', name='other')
assert kw2['name'] == 'other'
print("isolation", True)
try:
    validate_kwargs(cmd='echo', name='build', fname='dvc.yaml')
except Exception:
    print("invalidate", True)
else:
    raise AssertionError("name+fname mix must be rejected")
sc = StageCache.__new__(StageCache)
tmp = tempfile.mkdtemp()
sc.cache_dir = tmp
key, val = 'e' * 32, 'v1'
p = os.path.join(tmp, key[:2], key, val)
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, 'w') as f:
    f.write('cmd: [echo]\nouts: []\n')
assert isinstance(sc._load_cache(key, val), dict)
print("recompute", True)
"""


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def split_files(patch: str):
    chunks, cur_path, cur_lines = {}, None, []
    for line in patch.splitlines(keepends=True):
        m = re.match(r'diff --git a/(\S+) b/\S+', line)
        if m:
            if cur_path:
                chunks[cur_path] = cur_lines
            cur_path, cur_lines = m.group(1), []
            continue
        if cur_path is not None:
            cur_lines.append(line)
    if cur_path:
        chunks[cur_path] = cur_lines
    return chunks


def file_edits(hunk_lines, direction: str):
    edits, old, new = [], [], []
    for line in hunk_lines:
        if line.startswith('@@ '):
            if old or new:
                edits.append((''.join(old), ''.join(new)))
            old, new = [], []
        elif line.startswith(('index ', '--- ', '+++ ', '\\', 'diff ', 'old mode', 'new mode')):
            continue
        elif line.startswith(' '):
            old.append(line[1:]); new.append(line[1:])
        elif line.startswith('+'):
            (old if direction == 'A' else new).append(line[1:])
        elif line.startswith('-'):
            (new if direction == 'A' else old).append(line[1:])
    if old or new:
        edits.append((''.join(old), ''.join(new)))
    return [e for e in edits if e[0] != e[1]]


def expand_until_unique(text: str, old: str, new: str, max_grow: int = 60):
    if text.count(old) == 1:
        return old, new, False
    if text.count(old) < 1:
        return None
    idx = text.find(old)
    start, end = idx, idx + len(old)
    for _ in range(max_grow):
        ls = text.rfind('\n', 0, start - 1)
        le = text.find('\n', end)
        nstart = 0 if ls < 0 else ls + 1
        nend = len(text) if le < 0 else le + 1
        if nstart == start and nend == end:
            break
        prefix, suffix = text[nstart:start], text[end:nend]
        start, end = nstart, nend
        old, new = prefix + old + suffix, prefix + new + suffix
        if text.count(old) == 1:
            return old, new, True
    return None


def acquire(repo: str, rev: str, dest: Path, label: str):
    dest.mkdir(parents=True, exist_ok=True)
    arch = dest.with_suffix('.tgz')
    url = f'https://codeload.github.com/{repo}/tar.gz/{rev}'
    if not arch.exists():
        print(f'  downloading {label} {repo}@{rev}', flush=True)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    arch.write_bytes(r.read())
                break
            except Exception as e:
                print(f'    attempt {attempt} failed: {e}', flush=True)
                if attempt == 4:
                    raise
    with tarfile.open(arch) as bundle:
        members = bundle.getmembers()
        top = members[0].name.split('/')[0]
        for m in members:
            parts = Path(m.name).parts
            if not parts or parts[0] != top or '..' in parts:
                continue
            rel = Path(*parts[1:])
            if not rel.parts:
                continue
            tgt = dest / rel
            if m.isdir():
                tgt.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                payload = bundle.extractfile(m).read()
                if tgt.exists() and tgt.read_bytes() != payload:
                    raise ValueError(f'altered {tgt}')
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_bytes(payload)
    return {'repo': repo, 'requested_revision': rev, 'resolved_revision': rev,
            'archive_path': str(arch.relative_to(ROOT)), 'archive_sha256': digest(arch.read_bytes()),
            'download_url': url, 'immutable_url': url, 'commit_resolution': None}


def binding(primary_file: str, extra_files=()) -> str:
    mod = primary_file.removesuffix('.py').replace('/', '.')
    lines = ['from pathlib import Path', 'import hashlib, importlib',
             f"expected=(Path.cwd() / '{primary_file}').resolve()",
             'try:', f" module = importlib.import_module('{mod}')",
             'except Exception as import_error:', ' traceback=import_error.__traceback__',
             ' while traceback is not None:', "  file=traceback.tb_frame.f_globals.get('__file__')",
             '  if file and Path(file).resolve()==expected:',
             "   print('source_binding', str(expected), 'sha256', hashlib.sha256(expected.read_bytes()).hexdigest(), 'actual_import_failure_frame')",
             '  traceback=traceback.tb_next', ' raise',
             'actual=Path(module.__file__).resolve()',
             'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())',
             'assert actual == expected, (actual, expected)']
    for f in extra_files:
        lines.append(f"print('edited_file', '{f}', 'sha256', "
                     f"hashlib.sha256(Path('{f}').read_bytes()).hexdigest())")
    return '\n'.join(lines)


def hunk_starts(patch: str):
    out = {}
    cur = None
    for line in patch.splitlines():
        m = re.match(r'diff --git a/(\S+)', line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)', line)
        if m and cur and cur not in out:
            out[cur] = int(m.group(1))
    return out


def main():
    # locate Smith row
    pages = [p for p in (ACQ / 'artifacts/pages').glob('*.json') if not p.name.endswith('.receipt.json')]
    row, pg = None, None
    for p in pages:
        d = json.load(open(p))
        for it in d['rows']:
            if it['row']['instance_id'] == IID:
                row, pg = it['row'], p
                break
        if row:
            break
    assert row, f'row missing {IID}'
    print(f'row {IID} F2P={len(row["FAIL_TO_PASS"])} P2P={len(row["PASS_TO_PASS"])} page={pg.name}', flush=True)

    # execution commit
    remote = subprocess.check_output(
        ['git', 'ls-remote', f'https://github.com/{REPO}.git', f'refs/heads/{IID}'],
        text=True).strip()
    commit, ref = remote.split()
    assert len(commit) == 40, remote
    print(f'exec={commit[:12]}', flush=True)

    # buggy source under ART2 (additive; never touches ART f3-replace)
    ws = ART2 / f'{KEY}-source'
    buggy = acquire(REPO, commit, ws, KEY)

    # upstream: reuse cached f3-replace upstream dir + meta (read-only)
    stage1 = json.loads((OUT / 'stage1-manifest.json').read_text())
    upstreams = stage1['upstreams']
    tag = CANONICAL.replace('/', '__') + '.' + UPSTREAM_SHORT
    up = upstreams[tag]
    up_dest = ROOT / up['dir']
    assert (up_dest / 'dvc/stage/utils.py').is_file()
    print(f'upstream {tag} full={up["full"][:12]}', flush=True)

    # direction detect (stage1c logic)
    files = split_files(row['patch'])
    print(f'patch files={list(files)}', flush=True)
    chosen, detail = None, {}
    from pathlib import Path as P
    for direction in ('A', 'B'):
        ok, info = True, {}
        cur_texts = {}
        for f, hunk_lines in files.items():
            cur = (ws / f).read_text()
            cur_texts[f] = cur
            anchors = []
            for o, n in file_edits(hunk_lines, direction):
                grown = expand_until_unique(cur, o, n)
                if grown is None and o.endswith('\n') and cur.count(o[:-1]) == 1:
                    grown = (o[:-1], n, False)
                if grown is None:
                    ok = False
                    anchors.append({'ok': False, 'count': cur.count(o)})
                    continue
                go, gn, expanded = grown
                cur = cur.replace(go, gn)
                anchors.append({'ok': True, 'old_string': go, 'new_string': gn, 'expanded': expanded})
                cur_texts[f] = cur
            up_text = (up_dest / f).read_text()
            cur = cur_texts[f]
            if cur != up_text:
                if cur + '\n' == up_text or cur == up_text + '\n':
                    ok = False
                else:
                    ok = False
            info[f] = {'matches': cur == up_text, 'anchors_ok': all(a['ok'] for a in anchors)}
        detail[direction] = {'ok': ok, 'files': info}
        if ok and chosen is None:
            # rebuild anchors for recipe
            recipe_edits = []
            for f, hunk_lines in files.items():
                cur = (ws / f).read_text()
                for o, n in file_edits(hunk_lines, direction):
                    grown = expand_until_unique(cur, o, n)
                    if grown is None and o.endswith('\n') and cur.count(o[:-1]) == 1:
                        grown = (o[:-1], n, False)
                    go, gn, expanded = grown
                    cur = cur.replace(go, gn)
                    recipe_edits.append({'filepath': f, 'old_string': go, 'new_string': gn})
            chosen = (direction, recipe_edits)
    print(json.dumps(detail, indent=1)[:2000], flush=True)
    assert chosen, f'no direction reconstructs upstream: {detail}'
    direction, recipe_edits = chosen
    print(f'OK dir={direction} edits={len(recipe_edits)}', flush=True)

    # verify full-tree reconstruction for patched files
    for e in recipe_edits:
        fp = ws / e['filepath']
        text = fp.read_text()
        assert text.count(e['old_string']) == 1, (e['filepath'], text.count(e['old_string']))

    # original task file under NEW (additive)
    idx = {json.loads(l)['instance_id']: json.loads(l)
           for l in open(ROOT / 'competition/vfarl/acquisition/general/smith-task-index.jsonl')}
    dataset_readme = ROOT / 'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'
    assert dataset_readme.is_file()
    dataset_license = {'license': 'MIT', 'path': str(dataset_readme.relative_to(ROOT)),
                       'sha256': sha(dataset_readme), 'revision': REV}
    orig = NEW / (IID + '.task.json')
    if not orig.exists():
        save(orig, {'row': row, 'source': idx.get(IID),
                    'saved_page': str(pg.relative_to(ROOT)), 'saved_page_sha256': sha(pg)})

    # adapted task (verbatim statement) under ART2
    required = list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])
    targets = sorted({n.split('::', 1)[0] for n in required})
    missing = []
    for t in targets:
        if not (ws / t).exists() and (up_dest / t).exists():
            text = (up_dest / t).read_text()
            missing.extend(difflib.unified_diff([], text.splitlines(keepends=True),
                                                fromfile='/dev/null', tofile='b/' + t))
    test_patch = ''.join(missing)
    task = {'instance_id': IID, 'repo': REPO, 'base_commit': commit,
            'problem_statement': row['problem_statement'], 'patch': '',
            'test_patch': test_patch, 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
            'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
    adapted = ART2 / f'{KEY}-task.json'
    save(adapted, task)

    # recipe (six tools only)
    ordered_files = []
    for e in recipe_edits:
        if e['filepath'] not in ordered_files:
            ordered_files.append(e['filepath'])
    starts = hunk_starts(row['patch'])
    primary = ordered_files[0]
    cmd = 'python3 - <<\'PYREPLAY\'\n' + binding(primary, ordered_files[1:]) + '\n' + OBS + '\nPYREPLAY'

    def act(tool, args, exp='ok'):
        v = {'tool': tool, 'arguments': args, 'expected_status': exp}
        if tool == 'run_command':
            v['reviewed_command_sha256'] = digest(args['command'].encode())
        return v

    actions = [act('get_status', {})]
    for f in ordered_files:
        s = max(1, starts.get(f, 140) - 12)
        actions.append(act('read_file', {'filepath': f, 'start_line': s, 'end_line': s + 35}))
    actions.append(act('run_command', {'command': cmd}, 'error'))
    for e in recipe_edits:
        actions.append(act('edit_file', {'filepath': e['filepath'], 'old_string': e['old_string'],
                                         'new_string': e['new_string']}))
    actions.extend([act('run_command', {'command': cmd}), act('get_status', {}),
                    act('submit_patch', {})])
    tools_used = {a['tool'] for a in actions}
    assert tools_used <= {'get_status', 'read_file', 'run_command', 'edit_file', 'submit_patch'}, tools_used
    review_notes = (f'{CANONICAL} {SYMBOL} must preserve {CONTRACT}. {MECH} '
                    'Restore the exact production logic without hardcoding values or '
                    'altering tests. All original F2P/P2P identifiers and complete target '
                    'files are retained verbatim. The Smith row introduces the defect; this '
                    'is an off-policy scripted demonstration, not a model sample or '
                    'preference pair. The inline observation fails before the repair '
                    '(first run expects error) and passes after (second run expects ok), '
                    'so the second observation differs.')
    recipe = {'evidence_kind': 'human_reviewed_off_policy_demonstration',
              'reviewed_by': 'Codex root source contract and actual command review 2026-10-26',
              'review_notes': review_notes, 'actions': actions,
              'final_response': f'Restored {SYMBOL} {CONTRACT}. The actual inline cache '
                                'observation completed after the production edit. Patch submitted.'}
    recipe_path = REPLAY / f'f3r2-{KEY}-reviewed-recipe.json'
    save(recipe_path, recipe)
    save(NEW / f'{KEY}-reviewed-recipe.json', recipe)

    # source preparation
    lic = up_dest / 'LICENSE'
    assert lic.is_file()
    fam = f'{CANONICAL}.{SYMBOL}::{CONTRACT}'
    snap = f'{CANONICAL}::{up["full"]}'
    venv = REPLAY / '.venv-dvc/lib/python3.12/site-packages'
    assert venv.is_dir()
    prep = {'instance_id': IID, 'key': KEY, 'original_task': str(orig.relative_to(ROOT)),
            'original_task_sha256': sha(orig), 'adapted_task': str(adapted.relative_to(ROOT)),
            'workspace': str(ws.relative_to(ROOT)), 'source_dataset': 'SWE-bench/SWE-smith-py',
            'source_train_revision': REV, 'physical_split': 'train',
            'course_membership': 'F600_F3_new', 'primary_adapter': 'F', 'submechanism': 'F3',
            'canonical_repo': CANONICAL, 'repair_family_id': fam, 'snapshot_group_id': snap,
            'execution_commit': commit, 'git_ls_remote': remote,
            'buggy_source': buggy,
            'upstream_source': {'repo': CANONICAL, 'requested_revision': UPSTREAM_SHORT,
                                'resolved_revision': up['full'],
                                'archive_path': up['meta']['archive_path'],
                                'archive_sha256': up['meta']['archive_sha256'],
                                'download_url': up['meta']['download_url'],
                                'immutable_url': up['meta']['immutable_url'],
                                'commit_resolution': None},
            'license_evidence': {
                'source': {'path': str(lic.relative_to(ROOT)), 'sha256': sha(lic),
                           'revision': up['full']},
                'dataset': dataset_license},
            'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
            'visible_prompt_profile': None,
            'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
            'mechanism_evidence': MECH, 'verification_target_files': targets,
            'source_page': str(pg.relative_to(ROOT)), 'source_page_sha256': sha(pg),
            'mirror_ref': {'url': f'https://github.com/{REPO}.git',
                           'ref': f'refs/heads/{IID}', 'actual_response': remote,
                           'response_sha256': digest(remote.encode())},
            'task_venv_site_packages': str(venv.relative_to(ROOT)),
            'task_venv_python': None,
            'task_source_layout': '.',
            'verification_assets': [
                {'path': t,
                 'sha256': sha(ws / t) if (ws / t).exists() else sha(up_dest / t),
                 'bytes': (ws / t).stat().st_size if (ws / t).exists() else (up_dest / t).stat().st_size,
                 'present_in_bug_source': (ws / t).exists()} for t in targets],
            'source_review_path': str((NEW / f'{KEY}-reviewed-recipe.json').relative_to(ROOT)),
            'use_partition': 'train_only'}
    save(REPLAY / f'f3r2-{KEY}-source-preparation.json', prep)
    save(NEW / f'{KEY}-source-preparation.json', prep)
    # stage1c-style record for probe/replay
    save(NEW / f'{KEY}-stage1c.json', {'key': KEY, 'instance_id': IID, 'direction': direction,
                                       'recipe_edits': recipe_edits, 'direction_detail': detail,
                                       'reconstruction_ok': True})
    print(f'DONE {KEY} dir={direction} targets={targets} F2P={len(row["FAIL_TO_PASS"])} P2P={len(row["PASS_TO_PASS"])} edits={len(recipe_edits)}', flush=True)


if __name__ == '__main__':
    main()
