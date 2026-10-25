"""Prepare f2-recall R2: aiohttp-10672 (Response headers CIMultiDict copy isolation)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-recall'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-recall'

IID = 'aio-libs__aiohttp-10672'
KEY = 'f2-recall-aiohttp-10672'
REPO = 'aio-libs/aiohttp'
BASE = '3d11b5512aaaa53649e1c85b07d0145582bf4e0f'
FAMILY = 'aio-libs/aiohttp::aiohttp.web_response.Response.headers::copy_isolation'
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


row = json.loads((ACQ / 'pages-v2/aio-libs__aiohttp-10672.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'aiohttp-10672-source'
arch = ART / 'aiohttp-10672-source.tgz'
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
    if part.startswith('aiohttp/web_response.py'):
        prod_patch = 'diff --git a/' + part
        break
assert prod_patch, 'prod patch part missing'
edits = forward_edits(prod_patch)
assert len(edits) == 1, len(edits)
cur = (WS / 'aiohttp/web_response.py').read_text()
o, n = edits[0]
assert cur.count(o) == 1, o[:300]
cur = cur.replace(o, n)
assert 'real_headers = CIMultiDict(headers)' in cur
assert 'real_headers = headers' not in cur
print('edit reconstruction ready (1 prod hunk)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['tests/test_web_response.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-recall-aiohttp-10672-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('aiohttp.web_response')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('aiohttp/web_response.py'), actual\n")
OBS = ("from multidict import CIMultiDict\nfrom aiohttp import web\n"
       'h = CIMultiDict({"X-Base": "1"})\n'
       'r = web.Response(body=b"answer", headers=h)\n'
       "print('copy_identity', r.headers is not h)\n"
       "assert r.headers is not h, 'Response aliased caller headers'\n"
       'r.headers["X-New"] = "2"\n'
       'assert "X-New" not in h, dict(h)\n'
       "print('f2_headers_isolation', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'aiohttp/web_response.py', 'start_line': 528, 'end_line': 542}),
           act('run_command', {'command': CMD}, 'error'),
           act('edit_file', {'filepath': 'aiohttp/web_response.py',
                             'old_string': o, 'new_string': n}),
           act('run_command', {'command': CMD}),
           act('get_status', {}),
           act('submit_patch', {})]
recipe_path = REPLAY / 'f2-recall-aiohttp-10672-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('root research agent: causal source read (Response aliased CIMultiDict '
                    'headers; unconditional copy isolates caller mutation), pinned tests, command review 2026-10-25'),
    'review_notes': ('Response must own a copy of caller headers so preparing the response '
                     'does not pollute the shared input dict. Copy-time identity (is not the '
                     'input) plus mutation isolation are asserted source-bound before and after '
                     'the repair. Original F2P/P2P unchanged. Baseline diagnostic failure is '
                     'expected task reproduction, not a policy error. Off-policy scripted '
                     'demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Copy caller headers with CIMultiDict(headers) so Response owns its headers. '
                       'Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-recall/pages-v2/aio-libs__aiohttp-10672.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/aio-libs__aiohttp-10672.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-recall/aiohttp-10672-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('headers copy isolation: Response owns CIMultiDict(headers) '
                           '(is not the input); captured at Response(headers=...) timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-recall-aiohttp-10672-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
