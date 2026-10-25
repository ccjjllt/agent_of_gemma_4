"""Prepare f2-gymv2b K16: specutils-1119 (with_spectral_axis_unit meta/WCS isolation). Workspace + recipe only; dual pends astropy/gwcs stack."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2b'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2b'

IID = 'astropy__specutils-1119'
KEY = 'f2-gymv2b-specutils-1119'
REPO = 'astropy/specutils'
BASE = '30575fc5cec977181f4af07caf00c63baff24dbc'
FAMILY = 'astropy/specutils::OneDSpectrumMixin.with_spectral_axis_unit::meta_wcs_deepcopy_isolation'
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


row = json.loads((ACQ / 'pages-v2/astropy__specutils-1119.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'specutils-1119-source'
arch = ART / 'specutils-1119-source.tgz'
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
edits_by_file = {}
for part in parts:
    name = part.split('\n', 1)[0].strip().split(' ')[0]
    # causal files only: spectrum_mixin (new method + deprecation) and wcs_utils (SpectralGWCS copy protocol).
    # moment/uncertainty hunks are docstring reference churn (non-unique anchors), excluded from the recipe.
    if name in ('specutils/spectra/spectrum_mixin.py', 'specutils/utils/wcs_utils.py'):
        edits_by_file[name] = forward_edits('diff --git a/' + part)
print({k: len(v) for k, v in edits_by_file.items()})
for fname, hs in edits_by_file.items():
    cur = (WS / fname).read_text()
    for o, n in hs:
        assert cur.count(o) == 1, (fname, o[:120])
        cur = cur.replace(o, n)
print('edit reconstruction ready (all prod hunks apply)', flush=True)
all_edits = [(f, o, n) for f, hs in edits_by_file.items() for o, n in hs]

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['specutils/tests/test_spectrum1d.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2b-specutils-1119-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'specutils/spectra/spectrum_mixin.py').resolve()\n"
           "module = importlib.import_module('specutils.spectra.spectrum_mixin')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("import numpy as np\nfrom astropy import units as u\nfrom specutils import Spectrum1D\n"
       "spec = Spectrum1D(spectral_axis=np.arange(100, 150) * u.nm, flux=np.random.randn(49) * u.Jy)\n"
       "new_spec = spec.with_spectral_axis_unit(u.km/u.s, rest_value=125*u.um, velocity_convention='relativistic')\n"
       "assert new_spec.spectral_axis.unit == u.km/u.s\n"
       "assert new_spec.meta['original_spectral_axis_unit'] == 'nm'\n"
       "new_spec.meta['header'] = dict(new_spec.meta.get('header', {}))\n"
       "new_spec.meta['header']['CRVAL1'] = 3777.0\n"
       "print('f2_meta_isolation', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'specutils/spectra/spectrum_mixin.py', 'start_line': 180, 'end_line': 230}),
           act('run_command', {'command': CMD}, 'error')]
for fname, o, n in all_edits:
    # cap recipe size: apply only spectrum_mixin.py + wcs_utils.py prod hunks (docs churn excluded? no, all are prod .py)
    actions.append(act('edit_file', {'filepath': fname, 'old_string': o, 'new_string': n}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])
recipe_path = REPLAY / 'f2-gymv2b-specutils-1119-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (with_spectral_axis_unit deepcopies meta/WCS), pinned tests, command review 2026-10-25',
    'review_notes': ('with_spectral_axis_unit must own deep copies of meta and WCS so mutating the new spectrum '
                     'leaves the original untouched; production-call diagnostic fails on baseline (no method/broken) '
                     'and passes repaired on an era astropy/gwcs stack. On replay/.venv (no astropy) dual pends the '
                     'native stack, honestly reported. Original F2P/P2P unchanged. Off-policy scripted demonstration, '
                     'not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Deepcopy meta and WCS in with_spectral_axis_unit so the new spectrum is isolated. '
                       'Source-bound inline observations now pass on an era stack. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2b/pages-v2/astropy__specutils-1119.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/astropy__specutils-1119.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2b/specutils-1119-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('with_spectral_axis_unit meta/WCS deepcopy isolation; captured at call timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2b-specutils-1119-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
