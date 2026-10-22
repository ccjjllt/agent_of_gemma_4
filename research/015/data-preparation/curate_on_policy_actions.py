#!/usr/bin/env python3
"""Curate frozen-train, successful-draw causal action RFT (never full-task SFT).

Accept explicit train attempt directories only. Select the least-tool successful
draw per issue before examining window lengths. Existing raw history is retained
in full; unsupported finals and whole over-cap windows are never truncated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x]


def insist(ok, reason):
    if not ok:
        raise ValueError(reason)


def source_path(value):
    value = value.removeprefix('/workspace/').removeprefix('./')
    return value


def production(path):
    parts = Path(path).parts
    return bool(parts) and not any(x in ('tests', 'test', '.git') for x in parts) and not Path(path).name.startswith('test_') and not Path(path).stem.endswith('_test') and not re.search(r'(^|/)(repro|reproduce|reproduction|scratch|debug)([_.-]|$)', path, re.I)


def upstream_invocation(command):
    """Conservative real command recognition, not a substring in echo/code text."""
    if re.search(r'(\b(?:echo|printf|cat)\b|<<|\b(?:open|write_text|write_bytes)\s*\(|https?://)', command):
        return False
    try:
        words = shlex.split(re.split(r'&&|;', command)[-1].strip())
    except ValueError:
        return False
    while words and ('=' in words[0] or words[0] == 'env'):
        words.pop(0)
    if not words:
        return False
    exe = Path(words[0]).name
    return exe in ('pytest', 'py.test') or (re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', exe) is not None and len(words) > 2 and words[1] == '-m' and words[2] in ('pytest', 'unittest'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attempt', action='append', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prepared', action='append', type=Path, default=[])
    ap.add_argument('--capture-audit', action='append', type=Path, default=[])
    ap.add_argument('--material', action='append', type=Path, default=[])
    ap.add_argument('--max-targets', type=int, default=6)
    args = ap.parse_args()
    insist(1 <= args.max_targets <= 6, 'first-round target limit must be 1..6')
    frozen = read(HERE / 'train-identity-freeze.json')
    identities = {x['instance_id']: x for x in frozen['selected']}
    candidates, draw_log = {}, []
    for path in args.attempt:
        path = path.resolve()
        result, task = read(path / 'result.json'), read(path / 'agent-task.json')
        identity = identities.get(task['instance_id'])
        insist(result['split'] == 'train' and identity is not None, 'refusing non-frozen-train attempt before trace access')
        insist(task['repo'] == identity['repo'] and task['base_commit'] == identity['base_commit'], 'frozen repo/base mismatch')
        metrics = read(path / 'context_metrics.json')
        item = {'path': str(path), 'instance_id': task['instance_id'], 'tool_calls_used': metrics['tool_calls_used'], 'resolved': bool(result['resolved']), 'judge_valid': bool(result['judge_valid']), 'agent_error': result.get('agent_error')}
        draw_log.append(item)
        if item['resolved'] and item['judge_valid'] and not item['agent_error']:
            candidates.setdefault(task['instance_id'], []).append((item['tool_calls_used'], str(path)))
    audits = {read(p)['model_request_id']: p.resolve() for p in args.capture_audit}
    prepared = {read(p / 'manifest.json')['instance_id']: p.resolve() for p in args.prepared}
    materials = args.material or [ROOT / 'research/015/data/train-material', ROOT / 'research/015/data/train-next-material']
    args.output.mkdir(parents=True, exist_ok=True)
    encoded, selected_tx, issue_log, all_actions, common_tools = [], [], [], [], None
    for iid, draws in sorted(candidates.items()):
        _, chosen = min(draws)
        attempt = Path(chosen)
        material = next((p for p in materials if (p / 'snapshots' / (iid + '.tar.gz')).exists()), None)
        insist(material is not None, 'original-base snapshot unavailable for ' + iid)
        snapshot = material / 'snapshots' / (iid + '.tar.gz')
        with tarfile.open(snapshot) as archive:
            base_files = {x.name.removeprefix('repo/').removeprefix('./repo/') for x in archive.getmembers() if x.isfile()}
        data = prepared.get(iid)
        capture = read(attempt / 'trajectory-capture.json')
        audit_path = None
        if not capture['training_trace_usable']:
            reason_ids = {x.get('model_request_id') for x in capture['reasons']}
            audit_path = next((audits[x] for x in reason_ids if x in audits), None)
        if data is None:
            data = args.output / 'complete-action-windows' / iid
            cmd = [sys.executable, str(HERE / 'encode_on_policy_actions.py'), '--attempt', str(attempt), '--output', str(data), '--limits', '8192']
            if audit_path:
                cmd += ['--capture-audit', str(audit_path)]
            subprocess.run(cmd, check=True)
        manifest = read(data / 'manifest.json')
        insist(manifest['encoder_script_sha256'] == sha(HERE / 'encode_on_policy_actions.py'), 'prepared converter version mismatch')
        for name, expected in manifest['raw_source_hashes'].items():
            insist(sha(attempt / name) == expected, 'prepared source trace mismatch')
        for name, expected in manifest['outputs'].items():
            insist(sha(data / name) == expected['sha256'], 'prepared output hash mismatch')
        tools = read(data / 'actual-tools.json')
        insist(common_tools is None or common_tools == tools, 'mixed actual tool schemas require separate datasets')
        common_tools = tools
        transactions = {x['sample_id']: x for x in rows(data / 'causal-action-transactions.jsonl')}
        capped = {x['meta']['sample_id']: x for x in rows(data / 'encoded-8192-thought-mask.jsonl')}
        final_paths = {source_path(x) for x in re.findall(r'^\+\+\+ b/(.+)$', (attempt / 'agent.patch').read_text(), re.M)}
        production_final = {x for x in final_paths if x in base_files and production(x)}
        new_scripts = set()
        for tx in transactions.values():
            for call in tx['messages'][-2].get('tool_calls', []):
                fun = call['function']
                arg = json.loads(fun['arguments']) if isinstance(fun['arguments'], str) else fun['arguments']
                if fun['name'] == 'write_file':
                    path = source_path(arg.get('filepath', ''))
                    if path not in base_files or not production(path):
                        new_scripts.add(path)
        ranked, actions = [], []
        for item in manifest['census']:
            index = item['call_index']
            decision = {'instance_id': iid, 'attempt': chosen, 'call_index': index, 'selected': False}
            if 'excluded' in item:
                decision['reason'] = item['excluded']; actions.append(decision); continue
            sid = item['sample_id']; tx = transactions[sid]
            decision.update(sample_id=sid, model_request_id=item['model_request_id'], wire_id=item['wire_id'], tokens=item['tokens'], action_sha256=hashlib.sha256(json.dumps(tx, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest())
            calls = item['calls']
            decision['tool_call_ids'] = [c['id'] for c in calls]
            decision['tool_names'] = [c['function']['name'] for c in calls]
            if len(calls) != 1:
                decision['reason'] = 'multi-action target requires explicit curation'; actions.append(decision); continue
            fun = calls[0]['function']; name = fun['name']
            arg = json.loads(fun['arguments']) if isinstance(fun['arguments'], str) else fun['arguments']
            observation = json.loads(tx['messages'][-1]['content'])
            path = source_path(arg.get('filepath', ''))
            command = arg.get('command', '')
            category, reason, priority = None, 'not a necessary approved source action', None
            if name == 'write_file':
                reason = 'write_file/new repro scripts are never positive targets'
            elif name == 'run_command' and (re.search(r'\b(repro|reproduce|reproduction|scratch|debug)[\w.-]*\.py\b', command, re.I) or any(re.search(r'(?<![\w./-])(?:/workspace/|\./)?' + re.escape(s) + r'(?![\w./-])', command) or (Path(s).name and Path(s).name in command) for s in new_scripts)):
                reason = 'command invokes or handles a newly written script; context only'
            elif observation.get('status') != 'ok':
                reason = 'tool did not report successful execution; no blanket exit-code rule'
            elif name == 'edit_file' and path in production_final and arg.get('old_string') != arg.get('new_string'):
                category, reason, priority = 'production_edit', 'successful edit of an original production file retained in resolved final patch', 0
            elif name == 'run_command' and upstream_invocation(command):
                referenced = [x for x in base_files if ('tests' in Path(x).parts or Path(x).name.startswith('test_')) and (x in command or x.replace('/', '.').removesuffix('.py') in command)]
                if referenced and observation.get('exit_code') in (0, 1):
                    category, reason, priority = 'upstream_validation', 'actual invocation of original-base upstream test paths; nonzero exit is preserved as evidence, not automatically failure', 1
                    decision['upstream_paths'] = sorted(referenced)
                elif referenced:
                    reason = 'upstream command has interruption/usage/no-collection exit; not an assertion-result validation'
            elif name == 'submit_patch' and observation.get('patch_size', 0) > 0:
                category, reason, priority = 'submit', 'actual nonempty patch submission with real tool acknowledgment', 2
            elif name == 'read_file' and path in production_final:
                category, reason, priority = 'source_read', 'read of original production file actually changed by this resolved draw', 3
            elif name == 'get_status':
                category, reason, priority = 'status', 'actual budget/patch status observation', 4
            decision.update(category=category, reason=reason, exit_code=observation.get('exit_code'))
            if item['tokens'] > 8192:
                decision['reason'] += '; whole window exceeds 8192, excluded without truncation'
            elif category:
                ranked.append((priority, -index if category == 'source_read' else index, sid, decision))
            actions.append(decision)
        has_edit = any(x[3]['category'] == 'production_edit' for x in ranked)
        chosen_ids = set()
        if has_edit:
            for _, _, sid, decision in sorted(ranked)[:args.max_targets]:
                chosen_ids.add(sid); decision['selected'] = True
        else:
            for decision in actions:
                decision['reason'] += '; entire issue excluded: no in-cap successful production edit'
        for decision in actions:
            sid = decision.get('sample_id')
            if sid in chosen_ids:
                record = capped[sid]
                insist(record['meta']['thought_config']['thought_loss'] == 'mask' and record['meta']['group']['split'] == 'train', 'wrong loss/split')
                record['meta']['behavior']['curation'] = {'category': decision['category'], 'reason': decision['reason'], 'action_sha256': decision['action_sha256']}
                encoded.append(record); selected_tx.append(transactions[sid])
            elif decision.get('category') and decision.get('tokens', 8193) <= 8192 and has_edit:
                decision['reason'] += '; per-issue target budget priority'
        all_actions.extend(actions)
        issue_log.append({'instance_id': iid, 'chosen_attempt': chosen, 'draw_rule': 'minimum tool_calls_used among resolved+judge_valid+no-agent-error draws; stable path tie break; no alternate draw chosen based on length', 'original_snapshot_sha256': sha(snapshot), 'selected_targets': len(chosen_ids), 'admitted': has_edit, 'source_manifest_sha256': sha(data / 'manifest.json')})
    def emit(name, records):
        dest = args.output / name
        dest.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records))
        return sha(dest)
    encoded_sha = emit('encoded-data.jsonl', encoded)
    tx_sha = emit('selected-causal-actions.jsonl', selected_tx)
    tool_path = args.output / 'actual-tools.json'
    tool_path.write_text(json.dumps(common_tools or [], indent=2) + '\n')
    report = {'name': 'p015-on-policy-action-rft-curated', 'training_claim': 'on-policy action RFT, not complete task SFT; ordinary final reasoning unsupported and excluded', 'split': 'train', 'thought_loss': 'mask', 'max_seq_len': 8192, 'max_targets_per_issue': args.max_targets, 'encoded_data': str((args.output / 'encoded-data.jsonl').resolve()), 'encoded_data_sha256': encoded_sha, 'selected_transactions_sha256': tx_sha, 'actual_tools_path': str(tool_path.resolve()), 'actual_tools_file_sha256': sha(tool_path), 'samples': len(encoded), 'supervised_tokens': sum(x['meta']['supervised_tokens'] for x in encoded), 'curation_script_sha256': sha(Path(__file__)), 'input_draws': draw_log, 'issues': issue_log, 'actions': all_actions, 'context_policy': 'all actual causal history retained, even previous excluded/undesirable actions; no tool observations or previous assistant turns supervised; no clipping; no final reconstruction'}
    (args.output / 'dataset-manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ['encoded_data', 'encoded_data_sha256', 'samples', 'supervised_tokens']}, indent=2))


if __name__ == '__main__':
    main()
