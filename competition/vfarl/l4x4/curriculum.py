"""Build a train-only VFARL curriculum from frozen metadata and real outcomes.

This changes collection order, never task statements, assets, verifier labels or
sampled actions. Importing or running this module does not allocate a GPU.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ('cancelled-r1', 'cancelled-r2', 'cancelled-r3')


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def balanced_metadata_order(entries):
    """Use remaining role quotas, then repository diversity and weak cost priors.

    The remaining pool need not have the original inventory's 67/33 ratio.
    Historical demonstrations are only a scheduling prior, not policy lengths.
    """
    def cost(row):
        prior = row['historical_demo_length_prior']
        return (prior.get('decision_examples') or 10**9,
                prior.get('max_context_tokens') or 10**9,
                row['original_inventory_index'])

    roles = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in entries:
        roles[row['role']][row['submechanism']][row['canonical_repo']].append(row)
    queues = {}
    for role, mechanisms in roles.items():
        mechanism_queues = []
        for repositories in mechanisms.values():
            repos = [deque(sorted(values, key=cost)) for values in repositories.values()]
            repos.sort(key=lambda values: (cost(values[0]), values[0]['canonical_repo']))
            mechanism = deque()
            while any(repos):
                for values in repos:
                    if values:
                        mechanism.append(values.popleft())
            mechanism_queues.append(mechanism)
        mechanism_queues.sort(key=lambda values: (cost(values[0]), values[0]['submechanism']))
        ordered = deque()
        while any(mechanism_queues):
            for values in mechanism_queues:
                if values:
                    ordered.append(values.popleft()['instance_id'])
        queues[role] = ordered
    totals = {role: len(values) for role, values in queues.items()}
    emitted = Counter()
    result = []
    while any(queues.values()):
        role = min((role for role, values in queues.items() if values),
            key=lambda role: (emitted[role] / totals[role],
                              0 if role == 'specialist' else 1))
        result.append(queues[role].popleft())
        emitted[role] += 1
    return result


def build(selection_path, evidence_root, reserved_selection):
    selection_path, evidence_root = Path(selection_path), Path(evidence_root)
    sources = {}

    def load(path):
        path = Path(path)
        value = json.loads(path.read_text())
        sources[relative(path)] = file_sha(path)
        return value

    def source(path):
        return dict(path=relative(path), sha256=sources[relative(path)])

    selection = load(selection_path)
    selected = selection['tasks']
    ids = [row['instance_id'] for row in selected]
    if len(ids) != 100 or len(set(ids)) != len(ids):
        raise ValueError('This curriculum requires the frozen 100-task inventory')
    if any(row.get('physical_split') != 'train' for row in selected):
        raise ValueError('Only declared train metadata may enter this curriculum')
    if any(row.get('role') not in ('common', 'specialist') for row in selected):
        raise ValueError('Task role must be common or specialist, not inferred from adapter')
    observations = {key: {} for key in ids}

    def observation(key, run):
        return observations[key].setdefault(run, dict(run=run, claimed=False,
            original=None, branches={}, focal_groups=[]))

    def episode_record(value, provenance):
        return {**{key: value[key] for key in ('reward', 'verified', 'termination',
            'actions', 'final_context_tokens', 'policy_id') if key in value},
            'tool_failures_observed': value.get('ledger', {}).get('tool_failures_observed'),
            'source': provenance}

    for run in RUNS:
        directory = evidence_root/run
        summary_path = directory/'recovery-summary.json'
        if summary_path.exists():
            summary = load(summary_path)
            for key in summary.get('claimed_task_ids', []):
                if key in observations:
                    observation(key, run)['claimed'] = True
            for value in summary.get('original_episode_results', summary.get('original_episodes', [])):
                key = value['instance_id']
                if key in observations:
                    observation(key, run)['original'] = episode_record(value, source(summary_path))
        log_path = directory/'kernel.log'
        if log_path.exists():
            events = load(log_path)
            if not isinstance(events, list):
                raise ValueError('Expected official serialized event list: '+str(log_path))
            groups = {}
            for event in events:
                for line in str(event.get('data', '')).splitlines():
                    try:
                        value = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(value, dict):
                        continue
                    key, phase = value.get('instance_id'), value.get('phase')
                    if key not in observations:
                        continue
                    target = observation(key, run)
                    if phase == 'task_claimed':
                        target['claimed'] = True
                    elif phase == 'episode_result':
                        record = episode_record(value, {**source(log_path),
                            'session_seconds': event.get('time')})
                        if value.get('attempt') == 'original':
                            target['original'] = record
                        elif str(value.get('attempt', '')).startswith('branch-'):
                            target['branches'][value['attempt']] = record
                    elif phase == 'focal_group':
                        groups[(key, value['state_id'])] = dict(state_id=value['state_id'],
                            rewards=value.get('rewards', []), source=source(log_path),
                            checkpoint_turn=value.get('checkpoint_turn'),
                            actual_tool_failure_in_prefix=value.get('checkpoint_observed_tool_error'),
                            independent_rewards=value.get('rewards', [])[1:],
                            original_action_replay_reward=(value.get('rewards') or [None])[0],
                            protocol='historical-three-branch')
            for (key, _), group in groups.items():
                observation(key, run)['focal_groups'].append(group)
        # Episode files contain metadata, not issue text or reference patches.
        for path in sorted(directory.glob(
                'vfarl-train/resident/batches/batch-*/worker-*/*/draw-*/*/episode.json')):
            value = load(path)
            key = value.get('instance_id')
            if key not in observations:
                continue
            target = observation(key, run)
            target['claimed'] = True
            record = episode_record(value, source(path))
            if path.parent.name == 'original':
                target['original'] = record
            elif path.parent.name.startswith('branch-'):
                target['branches'][path.parent.name] = record

    pair_reviews = {}
    for run in RUNS:
        path = evidence_root/run/'tenacity-live-original-review.json'
        if path.exists():
            value = load(path)
            if value.get('instance_id') in observations:
                pair_reviews[value['instance_id']] = dict(source=source(path),
                    state_id=value.get('state_id'), comparisons=value.get('comparisons'),
                    reference_missing=value.get('reference_missing'),
                    eligible_for_training=value.get('eligible_for_training'),
                    interpretation='Source review only; missing reference is not replaced by a synthetic score')

    reserved = []
    if reserved_selection and Path(reserved_selection).exists():
        reserve = load(reserved_selection)
        reserved = list(reserve.get('research_split', {}).get('validation_task_ids', []))
    reserved_set = set(reserved)
    tasks, prior_claimed, paused, reuse = [], [], [], []
    for index, metadata in enumerate(selected):
        key = metadata['instance_id']
        history = list(observations[key].values())
        originals = [item['original'] for item in history if item['original'] is not None]
        latest = originals[-1] if originals else None
        claimed = any(item['claimed'] for item in history)
        candidate = any(item['original'] and item['original'].get('verified') is True
            and item['original'].get('reward') == 0
            and any(branch.get('verified') is True and branch.get('reward') == 1
                for name, branch in item['branches'].items() if name != 'branch-0')
            for item in history)
        groups = [group for item in history for group in item['focal_groups']]
        same_success = any(len(group['independent_rewards']) >= 2
            and all(reward == 1 for reward in group['independent_rewards']) for group in groups)
        scoring_unknown = bool(latest and (latest.get('verified') is not True
                                           or latest.get('reward') not in (0, 1)))
        if claimed:
            prior_claimed.append(key)
        if scoring_unknown:
            paused.append(key)
        if candidate:
            reuse.append(key)
        if key in reserved_set:
            bucket = 'reserved_validation'
        elif scoring_unknown:
            bucket = 'paused_scoring_unknown'
        elif candidate:
            bucket = 'reuse_source_review_candidate'
        elif not claimed:
            bucket = 'resident_unclaimed'
        elif latest and latest.get('verified') is True and latest.get('reward') == 0:
            bucket = 'valid_failure_observed'
        elif same_success:
            bucket = 'same_reward_state_observed'
        elif latest and latest.get('verified') is True and latest.get('reward') == 1:
            bucket = 'valid_success_observed'
        else:
            bucket = 'interrupted_without_terminal'
        tasks.append(dict(instance_id=key, original_inventory_index=index,
            role=metadata['role'], course=metadata['course'],
            primary_adapter=metadata['primary_adapter'], submechanism=metadata['submechanism'],
            canonical_repo=metadata['canonical_repo'], repair_family_id=metadata['repair_family_id'],
            task_input_sha256=metadata['task_input_sha256'], physical_split='train',
            historical_demo_length_prior=dict(
                max_context_tokens=metadata.get('old_thinking_zero_max_context_tokens'),
                decision_examples=metadata.get('old_decision_examples'),
                assistant_loss_tokens=metadata.get('old_assistant_loss_tokens'),
                interpretation='Weak historical demonstration cost prior; not current Gemma rollout length or difficulty'),
            resident_claimed=claimed, priority_bucket=bucket, observations=history,
            recovered_pair_review=pair_reviews.get(key),
            reuse_only_until_source_review=candidate,
            paused_scoring=scoring_unknown, excluded_from_training=key in reserved_set,
            focal_evidence_interpretation=(
                'Original0 and independent branch1 sources reviewed; actual reference completion and learner provenance acceptance remain required'
                if candidate and key in pair_reviews else
                'Original0 and independent branch1 observed1; exact source/restore/reference review still required; not a ready mixed group'
                if candidate else 'Same-reward groups describe only those checkpoints and policies; the task may be revisited with a new checkpoint or policy'
                if same_success else 'No inference of focal action preference from different task rewards')))
    eligible_new = [row for row in tasks if not row['resident_claimed']
                    and not row['paused_scoring'] and not row['excluded_from_training']]
    new_order = balanced_metadata_order(eligible_new)
    bucket_order = ('reuse_source_review_candidate', 'valid_failure_observed',
        'interrupted_without_terminal', 'valid_success_observed', 'same_reward_state_observed',
        'paused_scoring_unknown', 'reserved_validation')
    training_order = list(new_order)
    for bucket in bucket_order:
        training_order.extend(balanced_metadata_order([row for row in tasks
            if row['priority_bucket'] == bucket]))
    if len(training_order) != len(ids) or set(training_order) != set(ids):
        raise ValueError('Curriculum must preserve the original inventory exactly once')
    return dict(format='vfarl-evidence-course-v1',
        selection_path=relative(selection_path), selection_sha256=file_sha(selection_path),
        evidence_scope=dict(resident_runs=list(RUNS),
            claim_counts='Claims in these three resident runs only',
            older_sampling='Prior v4/16K sampling and prepared demonstrations are not counted as resident claims'),
        tasks=tasks, training_order=training_order, new_collection_order=new_order,
        prior_claimed_task_ids=prior_claimed, paused_scoring_ids=paused,
        reuse_candidate_ids=reuse, reserved_validation_task_ids=reserved,
        reserved_validation_overlap_ids=sorted(set(ids) & reserved_set),
        source_files=[dict(path=path, sha256=sha) for path, sha in sorted(sources.items())],
        counts=dict(inventory=len(ids), roles=dict(Counter(row['role'] for row in tasks)),
            resident_claimed=len(prior_claimed), new_collection=len(new_order),
            resident_claimed_roles=dict(Counter(row['role'] for row in tasks
                                                if row['resident_claimed'])),
            new_collection_roles=dict(Counter(row['role'] for row in eligible_new)),
            paused_scoring=len(paused), reuse_candidates=len(reuse),
            priority_buckets=dict(Counter(row['priority_bucket'] for row in tasks))),
        policy=dict(default_collection='new_collection_order only; no automatic replay of prior resident claims',
            exploration='All remaining resident-unclaimed tasks are retained; role quotas follow that remaining pool, mechanism/repository round-robin within each role',
            same_reward='Lower revisit priority for this observed state; not a permanent task exclusion',
            null_reward='Pause scoring-unknown tasks pending cause review; never convert null to model failure',
            preserved='Original 100 task identities, input hashes, statements and reference repairs unchanged',
            familiarity='Resident-unclaimed means unclaimed in cancelled R1/R2/R3 only; older v4/16K sampling and prepared demonstrations exist',
            limitation='Historical signals rank collection effort; this is not heldout evaluation, current-policy reward relabeling or proof of learning benefit'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=HERE/'v4/selection.json')
    parser.add_argument('--evidence-root', type=Path, default=HERE/'round100')
    parser.add_argument('--reserved-selection', type=Path, default=HERE/'next-round/selection.json')
    parser.add_argument('--out', type=Path, default=HERE/'round100/curriculum.json')
    args = parser.parse_args()
    result = build(args.selection, args.evidence_root, args.reserved_selection)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.part')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.out)
    print(json.dumps(dict(path=relative(args.out), sha256=file_sha(args.out), **result['counts']),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
