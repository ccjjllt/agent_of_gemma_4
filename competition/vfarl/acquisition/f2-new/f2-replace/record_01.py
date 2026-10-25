"""Record f2-replace-01 replay evidence and run normalize (supplemental = f2-replace catalog)."""
import hashlib, importlib.util, json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
REPLAY = ROOT / 'competition/vfarl/replay'
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new'
OUT = ACQ / 'f2-replace'
ART = REPLAY / 'artifacts/f2-replace'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


KEY = 'f2-replace-deepdiff-01'
IID = 'seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__w1di5431'
RUN = ART / 'f2-replace-deepdiff-01-run-001'

summary = json.loads((RUN / 'run-summary.json').read_text())
prep = json.loads((REPLAY / 'f2-replace-deepdiff-01-source-preparation.json').read_text())
adapted = json.loads((ROOT / prep['adapted_task']).read_text())
source = json.loads((ROOT / prep['original_task']).read_text())['row']
assert all(source[n] == adapted[n] for n in ('instance_id', 'problem_statement', 'FAIL_TO_PASS', 'PASS_TO_PASS'))
for phase in ('baseline', 'repaired'):
    assert len(list(RUN.glob(phase + '-_swegemma_junit_*.xml'))) == 1, phase

report = dict(summary)
report.update({
    'original_task': prep['original_task'],
    'original_task_sha256': prep['original_task_sha256'],
    'adapted_task': prep['adapted_task'],
    'recipe': prep['recipe'],
    'replay_output': str(RUN.relative_to(ROOT)),
    'source_preparation': 'competition/vfarl/replay/f2-replace-deepdiff-01-source-preparation.json',
    'source_train_revision': prep['source_train_revision'],
    'source_revision': prep['source_train_revision'],
    'canonical_repo': prep['canonical_repo'],
    'source_dataset': prep['source_dataset'],
    'physical_split': 'train',
    'primary_adapter': 'F',
    'submechanism': 'F2',
    'execution_commit': prep['execution_commit'],
    'repair_family_id': prep['repair_family_id'],
    'snapshot_group_id': prep['snapshot_group_id'],
    'mechanism': {'adapter': 'F', 'submechanism': 'F2', 'evidence': prep['mechanism_evidence']},
    'patch_sha256': sha(RUN / 'submitted.patch'),
})
result_path = REPLAY / (KEY + '-replay-result.json')
save(result_path, report)
print('wrote', result_path.relative_to(ROOT), flush=True)

spec = importlib.util.spec_from_file_location('f2_norm', REPLAY / 'normalize_results.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
catalog = {t['instance_id']: t for t in json.loads((OUT / 'catalog.json').read_text())['tasks']}
approvals = {e['instance_id']: e for e in
             json.loads((REPLAY / 'approved-source-requirement-mappings.json').read_text())['approvals']}
res = mod.normalize(result_path, catalog, approvals)
save(OUT / 'normalize-01.json',
     {'instance_id': IID,
      'eligible_for_human_off_policy_SFT': res['eligible_for_human_off_policy_SFT'],
      'ineligibility_reasons': res['ineligibility_reasons'],
      'complete_official_replay': res['complete_official_replay'],
      'baseline_verified_original_F2P_failure': res['baseline_verified_original_F2P_failure'],
      'baseline_evidence_profile': res['baseline_evidence_profile'],
      'strict_verbatim_required_ids_passed': res['strict_verbatim_required_ids_passed'],
      'source_requirement_profile': res['source_requirement_profile'],
      'repair_family_id': res['repair_family_id'],
      'snapshot_group_id': res['snapshot_group_id']})
print(json.dumps({'eligible': res['eligible_for_human_off_policy_SFT'],
                  'reasons': res['ineligibility_reasons'],
                  'complete_replay': res['complete_official_replay'],
                  'baseline_F2P_fail': res['baseline_verified_original_F2P_failure']}, indent=2))
