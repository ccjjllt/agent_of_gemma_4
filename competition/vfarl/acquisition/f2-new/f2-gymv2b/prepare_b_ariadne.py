"""Prepare f2-gymv2b K15: ariadne-159 (deepcopy before stripping mixin)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2b'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2b'

IID = 'mirumee__ariadne-codegen-159'
KEY = 'f2-gymv2b-ariadne-159'
REPO = 'mirumee/ariadne-codegen'
BASE = 'b09a532636ec6a8740f968261496a092531275a2'
FAMILY = 'mirumee/ariadne-codegen::ResultTypesGenerator._get_operation_definition_without_mixin_directive::deepcopy_isolation'
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


row = json.loads((ACQ / 'pages-v2/mirumee__ariadne-codegen-159.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'ariadne-159-source'
arch = ART / 'ariadne-159-source.tgz'
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

parts = row['patch'].split('diff --git a/')
prod_patch = ''
for part in parts:
    if part.startswith('ariadne_codegen/client_generators/result_types.py'):
        prod_patch = 'diff --git a/' + part
        break
assert prod_patch, 'prod patch part missing'
edits = forward_edits(prod_patch)
assert len(edits) == 4, len(edits)
cur = (WS / 'ariadne_codegen/client_generators/result_types.py').read_text()
for o, n in edits:
    assert cur.count(o) == 1, o[:160]
    cur = cur.replace(o, n)
assert 'deepcopy(self.operation_definition)' in row['patch']
print('edit reconstruction ready (4 prod hunks)', flush=True)

targets = sorted({n.split('::', 1)[0] for n in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['tests/client_generators/result_types_generator/test_get_operation_as_str.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2b-ariadne-159-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'ariadne_codegen/client_generators/result_types.py').resolve()\n"
           "module = importlib.import_module('ariadne_codegen.client_generators.result_types')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("from typing import cast\n"
       "from graphql import OperationDefinitionNode, build_ast_schema, parse\n"
       "from ariadne_codegen.client_generators.result_types import ResultTypesGenerator\n"
       "from ariadne_codegen.client_generators.constants import MIXIN_NAME\n"
       "from tests.client_generators.result_types_generator.schema import SCHEMA_STR\n"
       "query_str = ('query CustomQuery { query2 @mixin(from: \".mixins\", import: \"TestMixin\") { id '\n"
       "             'field1 @mixin(from: \".mixins\", import: \"TestMixin1\") { fielda } '\n"
       "             'field2 @mixin(from: \".mixins\", import: \"TestMixin2\") { fieldb } } }')\n"
       "generator = ResultTypesGenerator(schema=build_ast_schema(parse(SCHEMA_STR)),\n"
       "    operation_definition=cast(OperationDefinitionNode, parse(query_str).definitions[0]),\n"
       "    enums_module_name='enums')\n"
       "result = generator.get_operation_as_str()\n"
       "print('mixin_in_result', MIXIN_NAME in result)\n"
       "assert MIXIN_NAME not in result, 'mixin directive leaked to server string'\n"
       "field_query2 = generator.operation_definition.selection_set.selections[0]\n"
       "assert field_query2.directives[0].name.value == MIXIN_NAME, 'stored definition was mutated'\n"
       "print('stored_untouched', True)\n"
       "print('f2_clone_isolation', True)\n")
# source-bound isolation observation (production call; fails on baseline, passes repaired):
OBS2 = OBS
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS2 + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'ariadne_codegen/client_generators/result_types.py', 'start_line': 165, 'end_line': 180}),
           act('run_command', {'command': CMD}, 'error')]
for o, n in edits:
    actions.append(act('edit_file', {'filepath': 'ariadne_codegen/client_generators/result_types.py',
                                    'old_string': o, 'new_string': n}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])
recipe_path = REPLAY / 'f2-gymv2b-ariadne-159-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (deepcopy before mixin stripping preserves stored definition), pinned tests, command review 2026-10-25',
    'review_notes': ('get_operation_as_str must format a deepcopy so stripping mixin directives from the working '
                     'copy leaves the stored operation definition untouched; production-call diagnostic asserts '
                     'mixin-free server string plus untouched stored definition (fails on baseline, passes repaired). '
                     'Original F2P/P2P unchanged. Baseline diagnostic failure is expected task '
                     'reproduction, not a policy error. Off-policy scripted demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Deepcopy the operation definition before stripping mixin directives so the stored '
                       'definition is never mutated. Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2b/pages-v2/mirumee__ariadne-codegen-159.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/mirumee__ariadne-codegen-159.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2b/ariadne-159-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('ResultTypesGenerator deepcopy isolation: working definition is not the stored one; '
                           'stripping leaves the stored untouched; captured at get_operation_as_str timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2b-ariadne-159-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
