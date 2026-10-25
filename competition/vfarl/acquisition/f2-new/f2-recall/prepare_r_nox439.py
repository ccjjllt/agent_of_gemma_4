"""Prepare f2-recall R1: nox-439 (SessionRunner.posargs slice-copy isolation)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-recall'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-recall'

IID = 'theacodes__nox-439'
KEY = 'f2-recall-nox-439'
REPO = 'theacodes/nox'
BASE = '41b9c798e93dae8dc27256a07622c32bc5b9f0fd'
FAMILY = 'theacodes/nox::nox.sessions.SessionRunner.posargs::slice_copy_isolation'
SNAP = f'{REPO}::{BASE}'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(b):
    return hashlib.sha256(b).hexdigest()


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def forward_edits(patch):
    edits = []
    old, new = [], []
    for line in patch.splitlines(keepends=True):
        if line.startswith('@@ '):
            if old or new:
                edits.append((''.join(old), ''.join(new)))
            old, new = [], []
        elif line.startswith(('diff --git ', 'index ', '--- ', '+++ ', '\\')):
            continue
        elif line.startswith(' '):
            old.append(line[1:])
            new.append(line[1:])
        elif line.startswith('+'):
            new.append(line[1:])
        elif line.startswith('-'):
            old.append(line[1:])
    if old or new:
        edits.append((''.join(old), ''.join(new)))
    return [(o, n) for o, n in edits if o != n]


row = json.loads((ACQ / 'pages-v2/theacodes__nox-439.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'nox-439-source'
arch = ART / 'nox-439-source.tgz'
ART.mkdir(parents=True, exist_ok=True)
url = f'https://codeload.github.com/{REPO}/tar.gz/{BASE}'
assert arch.exists(), f'missing {arch} (pre-downloaded)'
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
        tgt = WS / rel
        if m.isdir():
            tgt.mkdir(parents=True, exist_ok=True)
        elif m.isfile():
            payload = bundle.extractfile(m).read()
            if tgt.exists() and tgt.read_bytes() != payload:
                raise ValueError(f'altered {tgt}')
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_bytes(payload)
print('workspace ready', flush=True)

parts = row['patch'].split('diff --git a/')
prod_patch = ''
for part in parts:
    if part.startswith('nox/sessions.py'):
        prod_patch = 'diff --git a/' + part
        break
assert prod_patch, 'prod patch part missing'
edits = forward_edits(prod_patch)
assert len(edits) == 1, len(edits)
cur = (WS / 'nox/sessions.py').read_text()
o, n = edits[0]
assert cur.count(o) == 1, o[:200]
cur = cur.replace(o, n)
assert 'global_config.posargs[:]' in cur
print('edit reconstruction ready (1 prod hunk)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['tests/test__option_set.py', 'tests/test_manifest.py',
                   'tests/test_sessions.py', 'tests/test_tasks.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-recall-nox-439-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('nox.sessions')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('nox/sessions.py'), actual\n")
OBS = ("from unittest import mock\nimport nox.sessions, nox.manifest\nfrom nox import _options\n"
       'cfg = _options.options.namespace(posargs=["--orig"])\n'
       'runner = nox.sessions.SessionRunner(name="test", signatures=["test"], func=mock.sentinel.func, global_config=cfg, manifest=mock.create_autospec(nox.manifest.Manifest))\n'
       "print('copy_identity', runner.posargs is not cfg.posargs)\n"
       "assert runner.posargs is not cfg.posargs, 'posargs aliased global config'\n"
       'runner.posargs.append("--new")\n'
       'assert cfg.posargs == ["--orig"], cfg.posargs\n'
       "print('f2_posargs_isolation', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'nox/sessions.py', 'start_line': 500, 'end_line': 512}),
           act('run_command', {'command': CMD}, 'error'),
           act('edit_file', {'filepath': 'nox/sessions.py',
                             'old_string': o, 'new_string': n}),
           act('run_command', {'command': CMD}),
           act('get_status', {}),
           act('submit_patch', {})]
recipe_path = REPLAY / 'f2-recall-nox-439-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('root research agent: causal source read (SessionRunner aliased '
                    'global posargs list; slice copy isolates per-session mutation), pinned tests, command review 2026-10-25'),
    'review_notes': ('SessionRunner must own a copy of global posargs so extending one session '
                     'does not pollute others or the global config. Copy-time identity '
                     '(is not the global list) plus mutation isolation are asserted source-bound '
                     'before and after the repair. Original F2P/P2P unchanged. Baseline diagnostic '
                     'failure is expected task reproduction, not a policy error. Off-policy scripted '
                     'demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Copy global posargs with [:] so each session owns its list. '
                       'Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-recall/pages-v2/theacodes__nox-439.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/theacodes__nox-439.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-recall/nox-439-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('posargs slice-copy isolation: SessionRunner owns a copy '
                           '(is not the global list); captured at SessionRunner init timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-recall-nox-439-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
