"""Generate train-only transactions using real official tool executions.

No stored trajectory shell is executed. Each plan contains reviewed inline
behavior cases and production edits. Old observations/futures are not copied.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('p006_replay', ROOT / 'research/006/data-replay-20261013/replay.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)

def source(key):
    matches = [json.loads(p.read_text()) for p in (HERE / 'source-trajectories').glob('*.json')]
    return next(x for x in matches if x.get('asset_key', x['repo'].split('/')[-1]) == key)

def source_edit(row, index):
    msg = json.loads(row['messages_raw'])[index]
    arg = json.loads(msg['tool_calls'][0]['function']['arguments'])
    if arg['command'] != 'str_replace':
        raise ValueError('Only reviewed production replacement accepted')
    return {'filepath': arg['path'].removeprefix('/testbed/'), 'old_string': arg['old_str'],
            'new_string': arg['new_str'], 'allow_multiple': False}

def run_one(key, plans, metadata, usage):
    row = source(key)
    if row['repo'].lower() in usage['excluded_repositories']:
        raise ValueError('Reserved repo cannot enter training producer')
    meta = next(m for m in metadata if m['instance_id'] == row['instance_id'])
    plan = plans[key]
    dest_root = HERE / 'outputs' / key
    attempt = 1
    while (dest_root / ('attempt-%02d' % attempt)).exists():
        attempt += 1
    dest = dest_root / ('attempt-%02d' % attempt)
    dest.mkdir(parents=True)
    workspace = Path(tempfile.mkdtemp(prefix='p009v2-' + key + '-')) / 'workspace'
    shutil.copytree(ROOT / meta['source_root'], workspace)
    official.baseline(workspace)
    messages = json.loads(row['messages_raw'])
    body = messages[1]['content']
    body = '\n'.join(p['text'] for p in body) if isinstance(body, list) else body
    issue = body.split('<pr_description>', 1)[-1].split('</pr_description>', 1)[0].strip()
    user = issue + '\n\nCPU transaction scope: ' + plan['scope'] + ' The complete repository benchmark has not been run.'
    rec = official.Recorder(official.Context(official.LocalTransport(workspace), row['repo']),
                            official.official_tools(), user)
    edits = [source_edit(row, i) for i in plan.get('source_edit_indices', [])] + plan.get('edits', [])
    targets = []
    before = after = submitted = None
    try:
        for filepath in dict.fromkeys(e['filepath'] for e in edits):
            text = (workspace / filepath).read_text()
            first = next(e for e in edits if e['filepath'] == filepath)
            location = text.find(first['old_string'])
            start = max(1, text[:max(location, 0)].count('\n') - 5)
            rec.call('read_file', {'filepath': filepath, 'start_line': start,
                                  'end_line': min(start + 80, len(text.splitlines()))})
        before = rec.call('run_command', {'command': plan['command']},
                          'Observe the stated behavior and boundary cases before editing.')
        if before['status'] != 'error':
            raise ValueError('No failing baseline observed; not admitted as repair demonstration')
        for edit in edits:
            ordinal = len(rec.events)
            result = rec.call('edit_file', edit, plan.get('edit_reason', 'Apply the minimal production correction supported by the observation.'))
            if result['status'] != 'ok':
                raise ValueError('Official edit failed; retain attempt without copying old future')
            targets.append(ordinal)
        ordinal = len(rec.events)
        after = rec.call('run_command', {'command': plan['command']},
                         'Repeat the identical behavior contract after the production edits.')
        if after['status'] != 'ok':
            raise ValueError('Post-edit behavior did not pass; retain failed attempt')
        targets.append(ordinal)
        ordinal = len(rec.events)
        submitted = rec.call('submit_patch', {}, 'Capture the production changes after the observed local behavior passed.')
        if submitted['status'] != 'ok' or not submitted.get('files_changed'):
            raise ValueError('No official production patch captured')
        targets.append(ordinal)
        result = 'local_behavior_repair'
    except Exception as exc:
        result = 'attempt_not_admitted'
        failure = type(exc).__name__ + ': ' + str(exc)
    patch = official.git(workspace, 'diff', '--binary', '_swegemma_baseline')
    (dest / 'production.patch').write_text(patch)
    (dest / 'events.json').write_text(json.dumps(rec.events, ensure_ascii=False, indent=2) + '\n')
    (dest / 'commands.json').write_text(json.dumps(rec.ctx.docker.commands, ensure_ascii=False, indent=2) + '\n')
    fingerprint = {p: {'before_sha256': official.digest((ROOT / meta['source_root'] / p).read_bytes()),
                       'after_sha256': official.digest((workspace / p).read_bytes())}
                   for p in dict.fromkeys(e['filepath'] for e in edits)}
    tx = {'sample_id': 'p009v2-' + key, 'transaction_id': 'p009v2-' + key + '-tx01',
          'synthetic': False, 'origin_kind': 'cpu_replayed_expert_demonstration',
          'group': {'issue': row['instance_id'], 'repo': row['repo'], 'commit': meta['buggy_commit'], 'split': 'train'},
          'source': {'uri': 'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/' + usage['source_revision'],
                     'license': 'MIT dataset; ' + plan['code_license'] + ' touched code',
                     'sha256': row['messages_sha256'], 'teacher': row['teacher_source'], 'traj_id': row['traj_id']},
          'protocol_id': 'p005-harness-tools-v1', 'messages': rec.messages,
          'target_assistant_steps': [x for x in targets if x not in plan.get('masked_action_ordinals', [])],
          'supervision_note': plan.get('supervision_note'),
          'behavior': 'local_edit', 'needs_replay': False,
          'observed_outcome': {'action_succeeded': result == 'local_behavior_repair', 'task_resolved': None,
                               'issue_reproduction_before_failed': before is not None and before['status'] == 'error',
                               'issue_reproduction_after_passed': after is not None and after['status'] == 'ok',
                               'official_suite_ran': False, 'validation_scope': plan['scope']},
          'state_kind': 'fresh_pinned_buggy_production_source',
          'source_resolved_label': row['upstream_resolved'], 'source_label_not_revalidated': True,
          'replay_boundary': 'macOS CPU; unchanged official v31 tool functions, P006 local filesystem/process transport; no Docker or benchmark suite',
          'continuation_policy': 'All tool observations newly executed; no old future or source reasoning copied',
          'production_file_fingerprints': fingerprint, 'attempt_status': result,
          'failure': failure if result != 'local_behavior_repair' else None}
    (dest / 'transaction.json').write_text(json.dumps(tx, ensure_ascii=False, indent=2) + '\n')
    summary = {'repo': row['repo'], 'issue': row['instance_id'], 'status': result, 'attempt': attempt,
               'path': str((dest / 'transaction.json').relative_to(HERE)),
               'baseline_exit': before and before.get('exit_code'), 'after_exit': after and after.get('exit_code'),
               'task_resolved': None, 'failure': tx['failure'], 'workspace': str(workspace)}
    print(json.dumps(summary), flush=True)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repos', nargs='*')
    args = parser.parse_args()
    plans = json.loads((HERE / 'transaction-plans.json').read_text())
    metadata = json.loads((HERE / 'source-manifest.json').read_text())
    usage = json.loads((HERE / 'data-usage.json').read_text())
    summary_path = HERE / 'replay-summary.json'
    summaries = json.loads(summary_path.read_text()) if summary_path.exists() else []
    for key in args.repos or plans:
        summaries.append(run_one(key, plans, metadata, usage))
        summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__':
    main()

