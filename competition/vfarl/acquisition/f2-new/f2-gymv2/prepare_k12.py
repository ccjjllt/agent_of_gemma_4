"""Prepare f2-gymv2 K12: asgiref-523 (_CVar copy-on-write context isolation)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2'

IID = 'django__asgiref-523'
KEY = 'f2-gymv2-asgiref-523'
REPO = 'django/asgiref'
BASE = 'f22bea26ff268c26253e827b127be76b61a612cf'
FAMILY = 'django/asgiref::_CVar::__setattr__/__delattr__::context_copy_on_write_isolation'
SNAP = f'{REPO}::{BASE}'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(b):
    return hashlib.sha256(b).hexdigest()


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


row = json.loads((ACQ / 'pages-v2/django__asgiref-523.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'asgiref-523-source'
arch = ART / 'asgiref-523-source.tgz'
ART.mkdir(parents=True, exist_ok=True)
url = f'https://codeload.github.com/{REPO}/tar.gz/{BASE}'
if not arch.exists():
    print(f'downloading {REPO}@{BASE}', flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'vfarl-f2-research'}),
                                timeout=300) as r:
        arch.write_bytes(r.read())
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

text = (WS / 'asgiref/local.py').read_text()
old_import = 'from typing import Any, Union'
new_import = 'from typing import Any, Dict, Union'
assert text.count(old_import) == 1
old_cls = '''class _CVar:
    """Storage utility for Local."""

    def __init__(self) -> None:
        self._data: dict[str, contextvars.ContextVar[Any]] = {}

    def __getattr__(self, key: str) -> Any:
        try:
            var = self._data[key]
        except KeyError:
            raise AttributeError(f"{self!r} object has no attribute {key!r}")

        try:
            return var.get()
        except LookupError:
            raise AttributeError(f"{self!r} object has no attribute {key!r}")

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            return super().__setattr__(key, value)

        var = self._data.get(key)
        if var is None:
            self._data[key] = var = contextvars.ContextVar(key)
        var.set(value)

    def __delattr__(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
        else:
            raise AttributeError(f"{self!r} object has no attribute {key!r}")
'''
new_cls = '''class _CVar:
    """Storage utility for Local."""

    def __init__(self) -> None:
        self._data: "contextvars.ContextVar[Dict[str, Any]]" = contextvars.ContextVar(
            "asgiref.local"
        )

    def __getattr__(self, key):
        storage_object = self._data.get({})
        try:
            return storage_object[key]
        except KeyError:
            raise AttributeError(f"{self!r} object has no attribute {key!r}")

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            return super().__setattr__(key, value)

        storage_object = self._data.get({}).copy()
        storage_object[key] = value
        self._data.set(storage_object)

    def __delattr__(self, key: str) -> None:
        storage_object = self._data.get({}).copy()
        if key in storage_object:
            del storage_object[key]
            self._data.set(storage_object)
        else:
            raise AttributeError(f"{self!r} object has no attribute {key!r}")
'''
assert text.count(old_cls) == 1, 'class anchor not unique'
assert '.copy()' in row['patch'] and 'contextvars.ContextVar(' in row['patch']
print('edit reconstruction ready', flush=True)

targets = sorted({n.split('::', 1)[0] for n in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['tests/test_local.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2-asgiref-523-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'asgiref/local.py').resolve()\n"
           "module = importlib.import_module('asgiref.local')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("import asyncio\nfrom asgiref.local import Local\n"
       "async def main():\n"
       "    test_local = Local()\n"
       "    test_local.value = 123\n"
       "    async def _test():\n"
       "        assert test_local.value == 123\n"
       "        del test_local.value\n"
       "        assert not hasattr(test_local, 'value')\n"
       "    await asyncio.create_task(_test())\n"
       "    print('parent_value', test_local.value)\n"
       "    assert test_local.value == 123, 'deletion leaked to parent'\n"
       "    first = test_local._data.get({})\n"
       "    test_local.other = 1\n"
       "    second = test_local._data.get({})\n"
       "    print('cow_identity', first is second)\n"
       "    assert first is not second, 'write did not copy-on-write'\n"
       "    assert first == {'value': 123}\n"
       "    print('f2_cow_isolation', True)\n"
       "asyncio.run(main())\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


nlines = len(text.splitlines())
actions = [act('get_status', {}),
           act('read_file', {'filepath': 'asgiref/local.py', 'start_line': 1, 'end_line': min(nlines, 60)}),
           act('run_command', {'command': CMD}, 'error'),
           act('edit_file', {'filepath': 'asgiref/local.py',
                             'old_string': old_import, 'new_string': new_import}),
           act('edit_file', {'filepath': 'asgiref/local.py',
                             'old_string': old_cls, 'new_string': new_cls}),
           act('run_command', {'command': CMD}),
           act('get_status', {}),
           act('submit_patch', {})]
recipe_path = REPLAY / 'f2-gymv2-asgiref-523-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (real-PR repair makes context-local writes copy-on-write), pinned tests, command review 2026-10-25',
    'review_notes': ('Context-local attribute writes/deletes must copy-on-write so concurrent tasks never observe '
                     'each other partial writes and deletion never leaks to the parent. Each write installs a new '
                     'storage dict (is-not the previous), earlier snapshots stay untouched, captured at set/delete '
                     'timepoint; asserted source-bound before and after the repair. Original F2P/P2P unchanged. '
                     'Baseline diagnostic failure is expected task reproduction, not a policy error. Off-policy '
                     'scripted demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Copy the storage dict on every context-local write and delete so tasks stay isolated. '
                       'Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2/pages-v2/django__asgiref-523.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/django__asgiref-523.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2/asgiref-523-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('_CVar copy-on-write isolation: every context-local write/delete installs a new storage '
                           'dict; deletion never leaks to the parent; captured at set/delete timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2-asgiref-523-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
