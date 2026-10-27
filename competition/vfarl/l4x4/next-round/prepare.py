"""Freeze new VFARL collection tasks and independent train-pool research probes.

This is a data preparation entry point. It reads only the admitted S600 train
course, historical training identity records, and old experiment metadata.
It does not run a model or turn historical demonstrations into policy samples.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

from competition.vfarl.l4x4.prepare import (
    ROOT, SOURCE, load_candidates, sha_file, source_path, verified_file,
)


HERE = Path(__file__).resolve().parent
COLLECTION_REPOSITORIES = (
    'bottlepy/bottle', 'Suor/funcy', 'google/textfsm',
    'luozhouyang/python-string-similarity', 'msiemens/tinydb',
    'pyparsing/pyparsing', 'prettytable/prettytable', 'mewwts/addict',
    'marshmallow-code/apispec', 'mahmoud/glom', 'jd/tenacity', 'PyCQA/flake8',
)
VALIDATION_REPOSITORIES = (
    'kayak/pypika', 'gweis/isodate', 'hukkin/tomli', 'lincolnloop/python-qrcode',
)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def history_projection():
    """Project identity fields; no non-training task bodies are accessed."""
    ids, families, historical_snapshots, sources = set(), set(), set(), []
    selections = [ROOT / 'competition/vfarl/l4x4/selection.json']
    selections += sorted((ROOT / 'competition/vfarl/l4x4').glob('v*/selection.json'))
    for path in selections:
        data = json.loads(path.read_text())
        task_ids = [row['instance_id'] for row in data['tasks']]
        ids.update(task_ids)
        families.update(row['repair_family_id'] for row in data['tasks'])
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha_file(path),
                            scope='Entire prior frozen VFARL inventory conservatively excluded',
                            instance_ids=task_ids))
    path = ROOT / 'research/015/data-preparation/train-identity-freeze.json'
    data = json.loads(path.read_text())
    task_ids = [row['instance_id'] for row in data['selected']]
    ids.update(task_ids)
    historical_snapshots.update((row['repo'].lower(), row['base_commit'])
                                for row in data['selected'])
    sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha_file(path),
                        scope='P015 frozen train identities, including unsampled candidates',
                        instance_ids=task_ids))
    # P017 manifests contain run settings/identity metadata, not task bodies.
    for path in sorted((ROOT / 'research/017/inference').glob('**/experiment-manifest.json')):
        task_ids = json.loads(path.read_text()).get('settings', {}).get('task_ids', [])
        ids.update(task_ids)
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha_file(path),
                            scope='P017 inference metadata identities only', instance_ids=task_ids))
    return ids, families, historical_snapshots, sources


def prepare(output, seed):
    course, candidates, inputs = load_candidates()
    old_ids, old_families, old_snapshots, sources = history_projection()
    profiles_path = ROOT / 'competition/vfarl/l4x4/kaggle/environment-profiles.json'
    profiles = json.loads(profiles_path.read_text())['profiles']
    by_profile = {row['source_profile']: index for index, row in enumerate(profiles)}
    # Keep new tasks on an already prepared Linux stack. Each new task still
    # needs its own real Linux baseline/repaired run before GPU collection.
    wanted = set(COLLECTION_REPOSITORIES + VALIDATION_REPOSITORIES)
    eligible, unavailable_sources = [], []
    for item in candidates:
        q = item['question']
        if (q['canonical_repo'] not in wanted or q['instance_id'] in old_ids
                or q['repair_family_id'] in old_families
                or item['old_max_context_tokens'] > 6144):
            continue
        report_path = verified_file(q['actual_replay_result'], q['actual_replay_result_sha256'])
        report = json.loads(report_path.read_text())
        identity = (q['canonical_repo'].lower(), str(report.get('base_commit_metadata', '')))
        if identity in old_snapshots:
            continue
        profile = by_profile.get(report.get('task_site_packages_profile') or 'stock')
        if profile is None or profile == 0:
            continue
        required = [q['task_input'], q['source_task_path'], q['source_review_path']]
        missing = [value for value in required if not (ROOT / value).is_file()]
        if missing:
            unavailable_sources.append(dict(instance_id=q['instance_id'], missing_local_sources=missing))
            continue
        eligible.append(dict(item, report=report, linux_profile=profile))
    chosen = []
    used_families = set()
    for repo in COLLECTION_REPOSITORIES + VALIDATION_REPOSITORIES:
        options = [item for item in eligible if item['question']['canonical_repo'] == repo
                   and item['question']['repair_family_id'] not in used_families]
        if not options:
            raise ValueError('No new task with an existing Linux profile for ' + repo)
        def order(item):
            iid = item['question']['instance_id']
            return (abs(item['old_max_context_tokens'] - 4500),
                    hashlib.sha256(f'{seed}:{iid}'.encode()).hexdigest())
        item = min(options, key=order)
        chosen.append(item)
        used_families.add(item['question']['repair_family_id'])
    collection = chosen[:len(COLLECTION_REPOSITORIES)]
    validation = chosen[len(COLLECTION_REPOSITORIES):]
    for key in ('instance_id', 'repair_family_id', 'snapshot_group_id'):
        if {item['question'][key] for item in collection} & {item['question'][key] for item in validation}:
            raise ValueError('Collection/research validation identity overlap: ' + key)
    output.mkdir(parents=True, exist_ok=True)
    questions, tasks = [], []
    for index, item in enumerate(chosen):
        q, report = item['question'], item['report']
        if not report.get('resolved_by_local_official_verify_task') or not report.get('eligible_for_human_off_policy_SFT'):
            raise ValueError('Missing admitted local replay evidence: ' + q['instance_id'])
        for key in ('task_input', 'source_task', 'source_review'):
            path_key = key if key == 'task_input' else key + '_path'
            verified_file(q[path_key], q[key + '_sha256'])
        recipe = source_path(report['recipe'])
        questions.append({key: q[key] for key in (
            'instance_id', 'official_initial_user_prompt', 'official_initial_user_prompt_sha256')})
        record = {key: value for key, value in q.items()
                  if key not in ('official_initial_user_prompt', 'metadata_is_policy_prompt', 'policy_input_field')}
        record.update(question_row=index + 1, role=item['role'],
            research_role='collection' if index < len(collection) else 'validation',
            recipe=str(recipe.relative_to(ROOT)), recipe_sha256=sha_file(recipe),
            old_thinking_zero_max_context_tokens=item['old_max_context_tokens'],
            old_decision_examples=item['old_decision_examples'],
            old_assistant_loss_tokens=item['old_assistant_loss_tokens'],
            linux_environment_profile_id='profile-%02d' % item['linux_profile'],
            linux_environment_source=profiles[item['linux_profile']]['source_profile'],
            linux_prevalidation_required=True, metadata_is_policy_prompt=False)
        tasks.append(record)
    question_path = output / 'questions.jsonl'
    question_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in questions))
    def counts(rows):
        return dict(tasks=len(rows), repositories=len({r['canonical_repo'] for r in rows}),
                    repair_families=len({r['repair_family_id'] for r in rows}),
                    snapshot_groups=len({r['snapshot_group_id'] for r in rows}),
                    mechanisms=dict(sorted(Counter(r['submechanism'] for r in rows).items())))
    data = dict(schema_version='vfarl-new-task-research-v1', seed=seed,
        purpose='Small new-task VFARL collection and independent train-pool before/after research probes',
        physical_split='train', only_training_whitelist_read=True,
        source_course_manifest=str(SOURCE.relative_to(ROOT)), source_course_manifest_sha256=sha_file(SOURCE),
        source_inputs=inputs, excluded_competition_repositories=course['excluded_competition_repositories'],
        historical_exclusions=dict(sources=sources, unique_instance_ids=len(old_ids),
            vfarl_repair_families=len(old_families), p015_snapshot_identities=len(old_snapshots),
            limitation='Identity exclusions cover the named frozen VFARL/P015 inventories and available P017 run manifests; this is not proof that every historical research artifact has been enumerated.'),
        selection_policy=dict(
            repositories_declared_before_new_outcomes=True,
            collection_repositories=list(COLLECTION_REPOSITORIES),
            research_validation_repositories=list(VALIDATION_REPOSITORIES),
            rule='One new repair family per declared repository; reuse an existing nonstock Linux profile; historical thinking=0 maximum length <=6144 and closest to4500, then seeded SHA256 tie break. No new model rewards used.',
            prior_frozen_vfarl_task_and_repair_family_excluded=True,
            collection_validation_task_family_snapshot_disjoint=True,
            old_lengths_are_only_a_selection_heuristic=True,
            unavailable_local_source_exclusions=unavailable_sources),
        linux_environment_profiles=dict(path=str(profiles_path.relative_to(ROOT)), sha256=sha_file(profiles_path),
            reuse='Existing frozen profile directories only; no new packages requested',
            new_task_acceptance='Every selected task still needs real Linux baseline/repaired official verification'),
        research_split=dict(collection_task_ids=[r['instance_id'] for r in tasks if r['research_role']=='collection'],
            validation_task_ids=[r['instance_id'] for r in tasks if r['research_role']=='validation'],
            validation_use='Before/after original rollouts only; no focal sampling, rewards, gradients or adaptive task selection',
            limitation='All16 belong to the previously prepared S600 demonstration course. Research probes are new to this VFARL round, not an official heldout benchmark or proof of never-seen pretraining/SFT data.'),
        counts=dict(all=counts(tasks), collection=counts(tasks[:len(collection)]), validation=counts(tasks[len(collection):])),
        questions=dict(path=question_path.name, sha256=sha_file(question_path), records=len(questions),
            policy_input_field='official_initial_user_prompt', allowed_fields=list(questions[0])),
        metadata_boundary='Only official_initial_user_prompt enters the policy. Selection/source links/reference patches/verifier assets remain judge-side.',
        tasks=tasks)
    write_json(output / 'selection.json', data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE)
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    result = prepare(args.out.resolve(), args.seed)
    print(json.dumps(result['counts'], ensure_ascii=False, indent=2))
