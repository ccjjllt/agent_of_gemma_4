"""Derive real focal samples from a cancelled collector; never generate rewards.

The CLI only reconstructs source-bound data. Reference completion requires an
explicit, already loaded policy supplied to complete_reference(); neither path
loads a model or submits a GPU job. Original recovered files are immutable.
"""
import argparse
import ast
import base64
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import zlib


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REVIEWED_COLLECTORS = {
    'ccaa9e2f1910dd6b05031047bb5965f5ad7628ca67cb91d916860450d75ab667',
}
DERIVED_RULE = 'source-reconstructed-first-real-tool-failure-with-recovery-budget-before-action-v1'
_FOCAL_RULE = """def focal_order(item):
    return (not item.get('has_recovery_budget', False),
            not item.get('observed_tool_error', False),
            len(item['action']['prompt_token_ids']))
"""


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode())


def source_path(name):
    path = Path(name)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT) or any('heldout' in part.lower() for part in path.parts):
        raise ValueError('Recovered sources must remain in the allowed research repository')
    return path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def collector_proof(text):
    """Bind reviewed source bytes and the concrete pre-sampling selection rule."""
    sha = digest(text.encode())
    if sha not in REVIEWED_COLLECTORS:
        raise ValueError('This frozen collector has not been reviewed for original-sample reuse')
    tree = ast.parse(text)
    focal = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == 'focal_order')
    if ast.dump(focal, include_attributes=False) != ast.dump(
            ast.parse(_FOCAL_RULE).body[0], include_attributes=False):
        raise ValueError('Frozen focal selection is not the reviewed prefix-only rule')
    episode = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
        and node.name == 'episode')
    calls = [node for node in ast.walk(episode) if isinstance(node, ast.Call)]
    captures = [node for node in calls if node.args and isinstance(node.args[0], ast.Attribute)
        and isinstance(node.args[0].value, ast.Name) and node.args[0].value.id == 'Snapshot'
        and node.args[0].attr == 'capture']
    samples = [node for node in calls if isinstance(node.func, ast.Attribute)
        and node.func.attr in ('sample', 'sample_async')]
    if len(captures) != 1 or not samples or captures[0].lineno >= min(n.lineno for n in samples):
        raise ValueError('Frozen source does not capture the state before sampling')
    capture_source = ast.get_source_segment(text, captures[0])
    return dict(source_frozen_collector_sha256=sha,
        source_focal_order_sha256=digest(ast.get_source_segment(text, focal).encode()),
        source_capture_before_sampling_sha256=digest(capture_source.encode()),
        source_focal_order=ast.get_source_segment(text, focal),
        source_capture_before_sampling=capture_source,
        source_frozen_collector_text=text)


def validate_source_proof(evidence):
    """Validate self-contained source evidence, including on a remote learner."""
    proof = collector_proof(evidence['source_frozen_collector_text'])
    if any(evidence.get(key) != value for key, value in proof.items()):
        raise ValueError('Derived collector source evidence was changed')
    if (evidence.get('selection_rule') != DERIVED_RULE
            or evidence.get('source_protocol') != 'historical-three-branch'
            or evidence.get('precommit_artifact_present') is not False
            or evidence.get('provenance_report_sha256') != evidence.get('selection_evidence_sha256')):
        raise ValueError('Derived evidence must explicitly describe legacy reconstruction')


@contextmanager
def reference_execution(policy):
    """Use one exclusive inference path, then restore the caller's mode."""
    previous = policy.info.get('execution_mode')
    if previous is None:
        previous = ('training' if policy.probability_profile.get('mode') ==
            'vfarl-focal-training' else 'inference')
    policy.switch_execution_mode('inference')
    try:
        yield
    finally:
        policy.switch_execution_mode(previous)


def build_pair(report_path, output_path, *, selection_path=None):
    """Create a pending pair from actual original/branch files and source proof."""
    from training.train_vfarl import live_focal_sample_sha256
    report_path, output_path = source_path(report_path), Path(output_path).resolve()
    report = json.loads(report_path.read_text())
    if report.get('format') != 'derived-original-pair-candidate':
        raise ValueError('Expected an explicit recovered-original candidate report')
    selection_path = source_path(selection_path or HERE/'v4/selection.json')
    selected = json.loads(selection_path.read_text())['tasks']
    task = next(row for row in selected if row['instance_id'] == report['instance_id'])
    if task.get('physical_split') != 'train':
        raise ValueError('Only the frozen train inventory can produce a recovered pair')
    inputs = {}
    for entry in report['source_files']:
        path = source_path(entry['path'])
        if digest(path.read_bytes()) != entry['sha256']:
            raise ValueError('Recovered source changed: '+str(path))
        inputs[str(path)] = entry['sha256']
    original_line, partner_line = report['source_lines']
    original_path, partner_path = source_path(original_line['path']), source_path(partner_line['path'])
    original_lines = original_path.read_bytes().splitlines()
    partner_lines = partner_path.read_bytes().splitlines()
    raw_original = original_lines[original_line['line']-1]
    raw_partner = partner_lines[partner_line['line']-1]
    original, partner = json.loads(raw_original), json.loads(raw_partner)
    for raw, row, entry in [(raw_original, original, original_line), (raw_partner, partner, partner_line)]:
        if (digest(raw) != entry['line_sha256']
                or live_focal_sample_sha256(row) != entry['sample_sha256']
                or row.get('original_action_replay') is not False):
            raise ValueError('A real sampled source line was changed or is a forced replay')
    episode_dir = original_path.parent.parent
    manifest_path = source_path(report['selection_reconstruction']['selected_snapshot'])
    snapshot = json.loads(manifest_path.read_text());state = snapshot['state']
    restore_path = partner_path.parent/'restore.json';restore = json.loads(restore_path.read_text())
    original_episode_path = original_path.parent/'episode.json'
    original_episode = json.loads(original_episode_path.read_text())
    partner_episode = json.loads((partner_path.parent/'episode.json').read_text())
    original_result = json.loads((episode_dir/'original-result.json').read_text())
    partner_result = json.loads((episode_dir/'branch-1-result.json').read_text())
    control_path = episode_dir/'focal-environment-control-result.json'
    control = json.loads(control_path.read_text())
    for summary, result in [(original_episode, original_result), (partner_episode, partner_result)]:
        if (summary.get('verified') is not True or result.get('verification_complete') is not True
                or summary.get('reward') not in (0, 1) or summary['reward'] != result.get('reward')):
            raise ValueError('A suffix has no complete consistent terminal reward')
    if control.get('verification_complete') is not True or control.get('reward') != 1:
        raise ValueError('The actual collection environment has no successful repair control')
    index = original_line['line']-1
    ledger = state['ledger']
    suffix_rows = [json.loads(line) for line in original_lines+partner_lines]
    if (state['context']['task_identity']['task_id'] != report['instance_id']
            or original_episode['instance_id'] != report['instance_id']
            or partner_episode['instance_id'] != report['instance_id']
            or canonical(state) != snapshot['state_id'] or snapshot['state_id'] != restore['state_id']
            or snapshot['state_id'] != report['state_id']
            or original['prompt_token_ids'] != partner['prompt_token_ids']
            or original['prompt_token_ids'] != state['prompt_token_ids']
            or original['prompt_token_ids'] != restore['prompt_token_ids']
            or ledger != restore['ledger'] or ledger['turns'] != index
            or ledger.get('tool_failures_observed') != 1 or ledger.get('last_tool_error') is not True
            or restore.get('filesystem_verified') is not True or restore.get('runtime_verified') is not True
            or restore.get('restoration_scope') != 'complete_declared_scope'
            or len({row['policy_id'] for row in suffix_rows}) != 1
            or state['policy_identity']['policy_id'] != original['policy_id']
            or restore['policy_identity']['policy_id'] != original['policy_id']
            or original['reference_id'] != partner['reference_id']
            or (original['completion_token_ids'], original['loss_mask']) ==
               (partner['completion_token_ids'], partner['loss_mask'])):
        raise ValueError('Recovered samples do not bind one real first-error state and policy')
    lengths = [len(json.loads(line)['prompt_token_ids']) for line in original_lines]
    if any(right <= left for left, right in zip(lengths, lengths[1:])):
        raise ValueError('The historical prefix ordering is not strictly increasing')
    carrier_manifest_path = next(source_path(entry['path']) for entry in report['source_files']
        if Path(entry['path']).name == 'carrier-manifest.json')
    carrier_path = carrier_manifest_path.with_name('run.py')
    packed = next(ast.literal_eval(node.value) for node in ast.parse(carrier_path.read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'SOURCE_BUNDLE' for target in node.targets))
    frozen = json.loads(zlib.decompress(base64.b64decode(packed)))['competition/vfarl/l4x4/run.py']
    proof = collector_proof(frozen)
    carrier_manifest = json.loads(carrier_manifest_path.read_text())
    if (carrier_manifest['source_files']['competition/vfarl/l4x4/run.py'] !=
            proof['source_frozen_collector_sha256']
            or report['selection_reconstruction']['frozen_source_sha256'] !=
            proof['source_frozen_collector_sha256']):
        raise ValueError('Frozen collector source is not bound to its actual carrier')
    branch_id = report['instance_id']+'-recovered-original'
    partner_id = report['instance_id']+'-recovered-branch-1'
    evidence = dict(format='vfarl-derived-live-capture-v1', **proof,
        source_protocol='historical-three-branch', precommit_artifact_present=False,
        selected_by_prefix_only=True, selection_rule=DERIVED_RULE,
        provenance_report_path=str(report_path), provenance_report_sha256=digest(report_path.read_bytes()),
        selection_evidence_path=str(report_path), selection_evidence_sha256=digest(report_path.read_bytes()),
        snapshot_manifest_path=str(manifest_path), snapshot_manifest_sha256=digest(manifest_path.read_bytes()),
        original_actions_path=str(original_path), original_action_index=index, original_action_line=index+1,
        original_action_line_sha256=digest(raw_original), original_action_sha256=live_focal_sample_sha256(original),
        original_episode_path=str(original_episode_path), original_episode_sha256=digest(original_episode_path.read_bytes()),
        original_action_replay=False, original_behavior_preserved=True,
        original_terminal_reward=original_episode['reward'], original_verified=True,
        state_id=snapshot['state_id'], restored_state_id=restore['state_id'],
        prompt_token_ids_sha256=canonical(original['prompt_token_ids']),
        restored_prompt_token_ids_sha256=canonical(restore['prompt_token_ids']),
        captured_ledger_sha256=canonical(ledger), restored_ledger_sha256=canonical(restore['ledger']),
        restored_policy_id=restore['policy_identity']['policy_id'],
        same_state_restore_evidence=str(restore_path), same_state_restore_sha256=digest(restore_path.read_bytes()),
        paired_branch_id=partner_id,
        reward_source='complete-verified-original-suffix; never a forced replay return')
    validate_source_proof(evidence)
    rows = []
    for source_row, episode, identity, kind in [
            (original, original_episode, branch_id, 'live_captured_original_action'),
            (partner, partner_episode, partner_id, 'sampled_focal_action')]:
        row = copy.deepcopy(source_row)
        row.update(state_id=snapshot['state_id'], instance_id=report['instance_id'],
            branch_id=identity, kind=kind, on_policy=True, verified=True, reward=episode['reward'],
            origin_episode_reward=original_episode['reward'],
            environment_control_verified=True, environment_control_result=str(control_path),
            restoration_scope=restore['restoration_scope'], restored_state_evidence=str(restore_path),
            checkpoint_turn=index, checkpoint_tool_observations=ledger['tool_observations'],
            checkpoint_tool_failures_observed=ledger['tool_failures_observed'],
            checkpoint_last_tool_feedback=ledger['last_tool_feedback'],
            collection_protocol='derived-live-original-pair',
            focal_selection_basis='reviewed historical prefix-only selection; original sample plus one independent restored sample')
        if kind == 'live_captured_original_action':
            row['live_capture_evidence'] = evidence
        if row.get('reference_forward_performed') is not True:
            # An uncomputed zero placeholder is not a reference probability.
            row['reference_logprobs'] = None
            row['reference_pending'] = True
        rows.append(row)
    if output_path.exists():
        raise ValueError('Refusing to overwrite a derived behavior file')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows))
    metadata = dict(format='vfarl-derived-pair-provenance-v1',
        input_report=str(report_path), input_report_sha256=digest(report_path.read_bytes()),
        selection_path=str(selection_path), selection_sha256=digest(selection_path.read_bytes()),
        task_input_sha256=task['task_input_sha256'], kernel=report['kernel'],
        state_id=snapshot['state_id'], source_files=report['source_files'],
        source_frozen_carrier_path=str(carrier_path), source_frozen_carrier_sha256=digest(carrier_path.read_bytes()),
        source_frozen_collector_sha256=proof['source_frozen_collector_sha256'],
        data_path=str(output_path), data_sha256=digest(output_path.read_bytes()),
        focal_sha256=digest(output_path.read_bytes()),
        samples=2, rewards=[row['reward'] for row in rows],
        reference_pending_branch_ids=[row['branch_id'] for row in rows if row.get('reference_pending')],
        eligible_for_training=False, new_rollout_performed=False, optimizer_steps=0,
        limitation='Pending data requires a real frozen-reference forward and strict learner acceptance; no old behavior probability was recomputed.')
    write_json(output_path.with_suffix('.manifest.json'), metadata)
    return metadata


def complete_reference(policy, pending_path, out_path):
    """Score missing references with an explicitly supplied, exclusively owned policy.

    The caller must close sampling services before this call: reference scoring
    temporarily zeroes LoRA scaling. This function never loads a model, samples
    an action, computes an actor replacement for old, or takes an optimizer step.
    """
    from training.train_vfarl import _live_capture_binding, live_focal_sample_sha256, prepare_rows
    from training.vfarl_policy import digest as identity_digest
    pending_path, out_path = Path(pending_path).resolve(), Path(out_path).resolve()
    if out_path.exists():
        raise ValueError('Refusing to overwrite scored behavior data')
    rows = [json.loads(line) for line in pending_path.read_text().splitlines() if line.strip()]
    before = copy.deepcopy(rows);scorings = []
    for row in rows:
        if row.get('kind') == 'live_captured_original_action':
            _live_capture_binding(row)
    missing = [row for row in rows if row.get('reference_forward_performed') is not True]
    execution_fields = {'probability_execution_sha256', 'probability_runtime_sha256'}
    core = lambda value: {key: item for key, item in value.items() if key not in execution_fields}
    if missing:
        with reference_execution(policy):
            for row in missing:
                source_reference = row['reference_identity']
                actual_reference = policy.identity['reference_identity']
                if (identity_digest(source_reference) != row['reference_id']
                        or core(source_reference) != core(actual_reference)):
                    raise ValueError('The loaded policy has different frozen reference weights or protocol')
                score = copy.deepcopy(policy.score_cached_reference(
                    row['prompt_token_ids'], row['completion_token_ids'], row['loss_mask']))
                if (score.get('frozen_reference') is not True
                        or score.get('reference_id') != policy.identity['reference_id']
                        or score.get('reference_identity') != actual_reference
                        or identity_digest(score['reference_identity']) != score['reference_id']
                        or core(score['reference_identity']) != core(source_reference)
                        or any(score.get(key) != row[key] for key in
                               ('prompt_token_ids', 'completion_token_ids', 'loss_mask'))):
                    raise ValueError('Reference scoring changed the source sequence or frozen reference core')
                values = score['logprobs']
                if (len(values) != len(row['completion_token_ids'])
                        or any(not math.isfinite(value) or value > 1e-6 for value in values)
                        or any(abs(value) > 1e-7 for value, keep in zip(values, row['loss_mask']) if not keep)):
                    raise ValueError('Reference forward returned invalid exact-token probabilities')
                row.update(reference_logprobs=values, reference_forward_performed=True,
                    reference_pending=False, reference_scoring='actual-posthoc-frozen-reference-forward',
                    posthoc_reference_evidence=dict(source_pending_sha256=digest(pending_path.read_bytes()),
                        scoring=score['scoring'], actual_reference_id=score['reference_id'],
                        actual_reference_identity=score['reference_identity'],
                        actual_execution=score['execution'],
                        actual_probability_execution_sha256=score.get('probability_execution_sha256'),
                        source_actor_reference_id=row['reference_id'],
                        identity_difference='Execution profile/runtime only; source actor identities stay unchanged',
                        seconds=score.get('seconds'), finite_logprobs_verified=True,
                        exact_tokens_and_mask_verified=True))
                scorings.append(dict(branch_id=row['branch_id'], result=score))
    allowed = {'reference_logprobs', 'reference_forward_performed', 'reference_pending',
               'reference_scoring', 'posthoc_reference_evidence'}
    for old, new in zip(before, rows):
        if ({key: value for key, value in old.items() if key not in allowed} !=
                {key: value for key, value in new.items() if key not in allowed}
                or live_focal_sample_sha256(old) != live_focal_sample_sha256(new)):
            raise ValueError('Reference completion changed immutable behavior or source evidence')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name+'.part')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows))
    try:
        prepared, groups = prepare_rows(temporary, policy.cfg['max_sequence_length'])
        if len(prepared) != len(rows) or groups['mixed_group_count'] != 1:
            raise ValueError('Scored recovered pair did not pass strict focal learner acceptance')
        scored_sha256 = digest(temporary.read_bytes())
        record = dict(format='vfarl-derived-reference-completion-v1',
            source_data_sha256=digest(pending_path.read_bytes()), data_sha256=scored_sha256,
            focal_sha256=scored_sha256,
            reference_forwards=len(scorings), scorings=scorings, groups=groups,
            immutable_behavior_preserved=True, eligible_for_training=True,
            new_rollout_performed=False, optimizer_steps=0)
        # Publish the validated data last. A failed sidecar write must not
        # strand a committed data file that cannot be retried or explained.
        write_json(out_path.with_suffix('.reference.json'), record)
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=HERE/'v4/selection.json')
    args = parser.parse_args()
    print(json.dumps(build_pair(args.report, args.out, selection_path=args.selection), ensure_ascii=False))


if __name__ == '__main__':
    main()
