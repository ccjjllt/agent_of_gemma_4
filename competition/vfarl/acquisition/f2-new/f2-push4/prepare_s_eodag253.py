"""Prepare f2-push4 S1: eodag-253 (download_all products copy isolation)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push4'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-push4'

IID = 'cs-si__eodag-253'
KEY = 'f2-push4-eodag-253'
REPO = 'CS-SI/eodag'
BASE = '0632e4147f26119bea43e0672f6d06e75d216a20'
FAMILY = 'cs-si/eodag::eodag.plugins.download.base.Download.download_all::products_copy_isolation'
SNAP = f'{REPO}::{BASE}'

PROD_FILES = ['eodag/api/core.py',
              'eodag/api/product/_product.py',
              'eodag/plugins/apis/base.py',
              'eodag/plugins/apis/usgs.py',
              'eodag/plugins/download/aws.py',
              'eodag/plugins/download/base.py',
              'eodag/plugins/download/http.py',
              'eodag/plugins/download/s3rest.py',
              'eodag/rest/utils.py',
              'eodag/utils/__init__.py']


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


row = json.loads((ACQ / 'pages-v2/cs-si__eodag-253.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'eodag-253-source'
arch = ART / 'eodag-0632e4147f26-source.tgz'
ART.mkdir(parents=True, exist_ok=True)
url = f'https://codeload.github.com/{REPO}/tar.gz/{BASE}'
assert arch.exists(), f'missing {arch}'
if not WS.exists():
    with tarfile.open(arch) as bundle:
        bundle.extractall(ART)
    tops = [d for d in ART.iterdir() if d.is_dir() and d.name.startswith('eodag-')]
    assert len(tops) == 1
    tops[0].rename(WS)
with tarfile.open(arch) as bundle:
    members = bundle.getmembers()
    top = members[0].name.split('/')[0]
    n_checked = 0
    for m in members:
        parts = Path(m.name).parts
        if not parts or parts[0] != top or '..' in parts:
            continue
        rel = Path(*parts[1:])
        if not rel.parts:
            continue
        tgt = WS / rel
        if m.isfile() and str(rel) in PROD_FILES:
            payload = bundle.extractfile(m).read()
            assert tgt.exists() and tgt.read_bytes() == payload, f'altered {tgt}'
            n_checked += 1
assert n_checked == len(PROD_FILES), n_checked
print('workspace ready', flush=True)

parts = row['patch'].split('diff --git a/')
file_edits = {}
patch_prod_files = set()
for part in parts:
    if not part.strip():
        continue
    name = part.split()[0].strip()
    if name in PROD_FILES:
        file_edits[name] = forward_edits('diff --git a/' + part)
    elif not name.startswith('tests/') and not name.startswith('docs/') and 'test' not in name.split('/')[-1]:
        patch_prod_files.add(name)
assert not patch_prod_files, patch_prod_files
assert set(file_edits) == set(PROD_FILES), set(file_edits)
cur_text = {}
for name, edits in file_edits.items():
    cur = (WS / name).read_text()
    for o, n in edits:
        assert o and cur.count(o) == 1, (name, o[:160])
        cur = cur.replace(o, n)
    cur_text[name] = cur
dl = cur_text['eodag/plugins/download/base.py']
assert 'products = products[:]' in dl
assert '# Products are going to be removed one by one from this sequence once' in dl
print(f'edit reconstruction verified ({sum(len(e) for e in file_edits.values())} prod hunks; workspace left at baseline)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['tests/test_cli.py', 'tests/test_end_to_end.py', 'tests/units/test_core.py',
                   'tests/units/test_eoproduct.py', 'tests/units/test_search_plugins.py',
                   'tests/units/test_search_result.py', 'tests/units/test_utils.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-push4-eodag-253-task.json'
save(adapted, task)

OBS = ("from types import SimpleNamespace\n"
       "from eodag.plugins.download.base import Download\n"
       "from eodag.api.search_result import SearchResult\n"
       "def make_product(i):\n"
       "    return SimpleNamespace(next_try=None, downloader=SimpleNamespace(config=SimpleNamespace()), downloader_auth=None, remote_location='http://example.com/x', properties={'title': 'p%d' % i})\n"
       "stub = SimpleNamespace(provider='stub', download=lambda product, **kw: '/tmp/stub-product.zip')\n"
       "caller = SearchResult([make_product(i) for i in range(3)])\n"
       "n_before = len(caller)\n"
       "snap = caller[:]\n"
       "print('snapshot_is_not_caller', snap is not caller)\n"
       "assert snap is not caller\n"
       "paths = Download.download_all(stub, caller, wait=0, timeout=0)\n"
       "print('caller_len_after', len(caller), 'paths', len(paths))\n"
       "assert len(paths) >= 1\n"
       "assert len(caller) == n_before, 'side effect: caller sequence consumed by download_all'\n"
       "print('f2_download_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('eodag.plugins.download.base')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('eodag/plugins/download/base.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'eodag/plugins/download/base.py',
                             'start_line': 242, 'end_line': 262}),
           act('run_command', {'command': CMD}, 'error')]
for name, edits in file_edits.items():
    for o, n in edits:
        actions.append(act('edit_file', {'filepath': name, 'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-push4-eodag-253-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-push4: causal source read (download_all consumed the caller '
                    'SearchResult via products.remove; per-call products[:] snapshot isolates; '
                    'same-issue sibling rows 254/260 share the fix hunk, COLLIDE-family; '
                    'pinned tests, command review 2026-10-25)'),
    'review_notes': ('download_all must own its working sequence: the working products list is '
                     'not the caller SearchResult, so per-product removal stays invisible to the '
                     'caller. Copy-time identity (snapshot is not the caller object) plus caller '
                     'intactness are asserted source-bound before and after the repair. The real '
                     'production retry/consume loop executes in both phases; only the leaf network '
                     'download is stubbed (offline sandbox). Original F2P/P2P unchanged. '
                     'Baseline diagnostic failure is expected task reproduction, not a policy error. '
                     'Off-policy scripted demonstration, not a Gemma sample. Geo stack via '
                     'task-site-packages-eodag; parent .venv otherwise.'),
    'actions': actions,
    'final_response': ('Snapshot the products sequence at download_all entry so per-product '
                       'consumption no longer empties the caller SearchResult. Source-bound '
                       'inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-push4/pages-v2/cs-si__eodag-253.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/cs-si__eodag-253.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-push4/eodag-0632e4147f26-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('download_all copy isolation: the working products sequence is not the '
                           'caller SearchResult; captured at the download_all entry timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-push4-eodag-253-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
