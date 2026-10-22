#!/usr/bin/env python3
"""Recover curated six-tool actions from immutable P015 train windows on CPU.

This is a finite, manually reviewed production recipe for six frozen issues.
It never executes source commands, samples a model, or reconstructs final text.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from training.encoding.gemma_tool_chat import (
    ENCODING_VERSION, EncodingError, encode_transaction, load_tokenizer,
    normalize_messages, sha256_json, tokenizer_fingerprint,
)

P015 = ROOT / 'research/015'
OLD = P015 / 'data/on-policy-rft-first-v1'
TOKENIZER = P015 / 'tokenizer-assets/official-31b-ct-v2-tokenizer'

# All other actions stay in the genuine request history with loss masked.
# A repeated edit is retained only at the last effective occurrence, so its
# earlier duplicate and intervening revert are not extra positive examples.
SELECTION = {
    'dask__dask-9212': {
        'localization': [0], 'source_read': [1, 3, 6, 27],
        'production_edit': [23, 25], 'submit': [32]},
    'graphql-python__graphene-1478': {
        'localization': [1], 'source_read': [2, 3, 4],
        'production_edit': [16], 'patch_review': [20], 'submit': [22]},
    'john-kurkowski__tldextract-258': {
        'source_read': [0, 1, 21], 'production_edit': [22],
        'patch_review': [24], 'submit': [25]},
    'mahmoud__boltons-298': {
        'localization': [2, 7, 10, 12], 'source_read': [0, 1, 8, 9, 11],
        'production_edit': [13], 'patch_review': [20], 'submit': [21]},
    'marshmallow-code__marshmallow-2715': {
        'localization': [0, 3], 'source_read': [1, 4, 5, 6, 7, 8, 9, 17, 25],
        'production_edit': [10, 11, 12, 13, 18, 21, 23, 24, 26, 27],
        'submit': [29]},
    'oauthlib__oauthlib-771': {
        'localization': [1, 5, 8, 9, 15],
        'source_read': [0, 2, 4, 6, 10, 12, 13, 20],
        'production_edit': [21], 'patch_review': [26], 'submit': [27]},
}
RECOVERY = {
    ('dask__dask-9212', 23): 'successful shell edit after repeated missing-argument edit_file errors',
    ('dask__dask-9212', 25): 'successful inline production edit after missing-argument edit_file error',
    ('graphql-python__graphene-1478', 16): 'last retained edit after reverted trial; earlier equivalent edit masked',
    ('john-kurkowski__tldextract-258', 21): 'read actual source after accidental checkout removed the repair',
    ('john-kurkowski__tldextract-258', 22): 'restore final retained repair after checkout; duplicate earlier edit masked',
    ('marshmallow-code__marshmallow-2715', 17): 'inspect actual file after failed replacement',
    ('marshmallow-code__marshmallow-2715', 18): 'successful replacement following failed replacement and fresh read',
    ('oauthlib__oauthlib-771', 8): 'find original validator path after wrong-path command error',
    ('oauthlib__oauthlib-771', 9): 'correct-path lookup after wrong-path command error',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_jsonl(path, value):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in value))


def exclusion_reason(iid, index, tx, new_scripts):
    fun = tx['messages'][-2]['tool_calls'][0]['function']
    arg = json.loads(fun['arguments'])
    observation = json.loads(tx['messages'][-1]['content'])
    if fun['name'] == 'write_file':
        return 'new repro/helper script write; retained in history, not a target'
    if any(name in arg.get('command', '') for name in new_scripts):
        return 'execution or cleanup of new repro/helper script; not upstream validation'
    if observation.get('status') != 'ok':
        return 'failed or malformed tool action; retained in history, not a target'
    if (iid, index) in {('graphql-python__graphene-1478', 10), ('john-kurkowski__tldextract-258', 5)}:
        return 'earlier duplicate edit later undone; only last retained repair selected'
    if (iid, index) == ('graphql-python__graphene-1478', 14):
        return 'reverted candidate edit; not retained in final repair'
    if (iid, index) == ('john-kurkowski__tldextract-258', 9):
        return 'premature submit includes three unintended cache files; not selected'
    if (iid, index) == ('john-kurkowski__tldextract-258', 15):
        return 'checkout removed intended repair; destructive mistake stays masked'
    if (iid, index) == ('dask__dask-9212', 2):
        return 'misspelled start_line argument made unintended broad read; stays masked'
    return 'redundant or low-value exploration/cleanup; excluded by finite manual recipe'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'research/016/data/target-protocol')
    args = ap.parse_args()
    frozen_path = P015 / 'data-preparation/train-identity-freeze.json'
    frozen = {x['instance_id']: x for x in read(frozen_path)['selected']}
    old_manifest = read(OLD / 'dataset-manifest.json')
    chosen = {x['instance_id']: x for x in old_manifest['issues']}
    require(set(chosen) == set(SELECTION), 'frozen six-issue source set changed')
    identities = []
    # This entire pass precedes access to any prepared or raw trace contents.
    for iid, selected in sorted(chosen.items()):
        attempt = Path(selected['chosen_attempt'])
        result, task = read(attempt / 'result.json'), read(attempt / 'agent-task.json')
        identity = frozen.get(iid)
        require(identity is not None and task['instance_id'] == iid and result['split'] == 'train', 'not frozen train')
        require(task['repo'] == identity['repo'] and task['base_commit'] == identity['base_commit'], 'source identity drift')
        require(result['resolved'] is True and result['judge_valid'] is True and not result.get('agent_error'), 'not strict success')
        identities.append(dict(instance_id=iid, repo=task['repo'], base_commit=task['base_commit'], split='train',
            attempt=str(attempt), result_sha256=sha(attempt / 'result.json'), task_sha256=sha(attempt / 'agent-task.json')))
    tokenizer = load_tokenizer(str(TOKENIZER))
    template = (TOKENIZER / 'chat_template.jinja').read_text()
    tok_hash = tokenizer_fingerprint(tokenizer)
    tools = read(OLD / 'actual-tools.json')
    require(len(tools) == 6, 'unexpected target schema')
    def encode(tx):
        return encode_transaction(tx, tokenizer, tools, max_seq_len=32768, thought_loss='mask',
            chat_template=template, tokenizer_hash=tok_hash, tokenizer_revision='official-mounted-31b-ct-v2')
    decisions, selected_tx, single_records, merged_tx, records, merge_log = [], [], [], [], [], []
    provenance = []
    for identity in identities:
        iid, attempt = identity['instance_id'], Path(identity['attempt'])
        prepared = OLD / 'complete-action-windows' / iid
        manifest = read(prepared / 'manifest.json')
        require(sha(prepared / 'manifest.json') == chosen[iid]['source_manifest_sha256'], 'prepared manifest drift')
        for name, expected in manifest['raw_source_hashes'].items():
            require(sha(attempt / name) == expected, 'raw source changed: ' + name)
        tx_file = prepared / 'causal-action-transactions.jsonl'
        require(sha(tx_file) == manifest['outputs'][tx_file.name]['sha256'], 'prepared transactions changed')
        require(read(prepared / 'actual-tools.json') == tools, 'schema drift')
        snapshot = next((P015 / 'data' / kind / 'snapshots' / (iid + '.tar.gz')
            for kind in ('train-material', 'train-next-material')
            if (P015 / 'data' / kind / 'snapshots' / (iid + '.tar.gz')).exists()), None)
        require(snapshot and sha(snapshot) == chosen[iid]['original_snapshot_sha256'], 'base snapshot drift')
        with tarfile.open(snapshot) as archive:
            base_files = {x.name.removeprefix('./repo/').removeprefix('repo/') for x in archive.getmembers() if x.isfile()}
        txs = {x['behavior']['call_index']: x for x in rows(tx_file)}
        new_scripts = set()
        for tx in txs.values():
            for call in tx['messages'][-2]['tool_calls']:
                if call['function']['name'] == 'write_file':
                    new_scripts.add(json.loads(call['function']['arguments'])['filepath'])
        selected = {index: category for category, indexes in SELECTION[iid].items() for index in indexes}
        require(set(selected) <= set(txs), 'missing reviewed action')
        normalized, encoded, accepted = {}, {}, []
        for item in manifest['census']:
            index = item['call_index']
            decision = dict(instance_id=iid, call_index=index, selected=False)
            if 'excluded' in item:
                decision['reason'] = item['excluded']
            elif index not in selected:
                decision['reason'] = exclusion_reason(iid, index, txs[index], new_scripts)
            else:
                tx = copy.deepcopy(txs[index]); category = selected[index]
                require(tx['group']['issue'] == iid and tx['group']['split'] == 'train', 'prepared identity mismatch')
                call = tx['messages'][-2]['tool_calls']
                require(len(call) == 1, 'reviewed action must have one call')
                fun = call[0]['function']; arg = json.loads(fun['arguments']); observation = json.loads(tx['messages'][-1]['content'])
                require(observation.get('status') == 'ok' and fun['name'] != 'write_file', 'selected error or new script')
                if fun['name'] == 'read_file' or fun['name'] == 'edit_file':
                    require(arg['filepath'] in base_files, 'target is not original-base file')
                if fun['name'] == 'run_command':
                    require(observation.get('exit_code') == 0, 'selected command failure')
                if category == 'submit':
                    require(fun['name'] == 'submit_patch' and observation['files_changed'] in (1, 2), 'unexpected final patch')
                tx['behavior']['curation'] = {'category': category, 'recovery': RECOVERY.get((iid, index)),
                    'policy': 'manual-final-retained-actions-v1', 'source_transaction_sha256': sha256_json(txs[index])}
                try:
                    record = encode(tx)
                except EncodingError as error:
                    decision['reason'] = 'v4 whole-window rejection: ' + str(error)
                else:
                    decision.update(selected=True, category=category, recovery=RECOVERY.get((iid, index)),
                        sample_id=tx['sample_id'], sequence_length=len(record['input_ids']),
                        supervised_tokens=record['meta']['supervised_tokens'],
                        tool_name=fun['name'], tool_call_id=call[0]['id'],
                        reason='reviewed genuine action; complete visible history; errors and other targets stay masked')
                    normalized[index] = normalize_messages(tx['messages'], tools)[0]
                    encoded[index] = record
                    accepted.append(index); selected_tx.append(tx); single_records.append(record)
            decisions.append(decision)
        require(any(selected[x] == 'production_edit' for x in accepted), 'issue lacks admitted production edit')
        # Start with the longest later actual window. Absorb only actual earlier
        # message prefixes, never append synthetic history across compaction.
        remaining = set(accepted)
        while remaining:
            last = max(remaining)
            chain = sorted(i for i in remaining if normalized[last][:len(normalized[i])] == normalized[i])
            tx = copy.deepcopy(txs[last]); tx['sample_id'] = tx['transaction_id'] = iid + '-segment-%02d' % last
            tx['target_assistant_steps'] = [txs[i]['target_assistant_steps'][0] for i in chain]
            tx['behavior'] = dict(target_call_indexes=chain, categories=[selected[i] for i in chain],
                recoveries=[{'call_index': i, 'reason': RECOVERY[iid, i]} for i in chain if (iid, i) in RECOVERY],
                source_transactions_sha256=[sha256_json(txs[i]) for i in chain],
                claim='selected-action RFT; not complete autonomous-task SFT; no raw final available')
            try:
                record = encode(tx)
                # Token-for-token prefix agreement and identical supervised
                # subsequences prove the merge did not alter a target's state.
                for i in chain:
                    prior = encoded[i]
                    for pos, label in enumerate(prior['labels']):
                        if label != -100:
                            require(record['input_ids'][:pos+1] == prior['input_ids'][:pos+1], 'merged target prefix drift')
                            require(record['labels'][pos] == label, 'merged label drift')
                require(record['meta']['supervised_tokens'] == sum(encoded[i]['meta']['supervised_tokens'] for i in chain), 'merge changed target count')
            except (EncodingError, ValueError) as error:
                merge_log.append(dict(instance_id=iid, proposed=chain, reason=str(error), action='keep genuine individual windows'))
                for i in chain:
                    original = next(t for t in selected_tx if t['sample_id'] == txs[i]['sample_id'])
                    merged_tx.append(original); records.append(encoded[i])
            else:
                merged_tx.append(tx); records.append(record)
                merge_log.append(dict(instance_id=iid, accepted=chain, output=tx['sample_id'],
                    tokens_before=sum(len(encoded[i]['input_ids']) for i in chain), tokens_after=len(record['input_ids'])))
            remaining.difference_update(chain)
        provenance.append(dict(**identity, prepared_manifest_sha256=sha(prepared / 'manifest.json'),
            raw_hashes=manifest['raw_source_hashes'], prepared_transactions_sha256=sha(tx_file),
            snapshot_sha256=sha(snapshot), patch_sha256=sha(attempt / 'agent.patch'),
            original_capture=manifest['original_capture'], capture_audit=manifest.get('capture_audit')))
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'selected-causal-actions.jsonl', selected_tx)
    write_jsonl(output / 'merged-transactions.jsonl', merged_tx)
    write_jsonl(output / 'encoded-data.jsonl', records)
    write_json(output / 'actual-tools.json', tools)
    selected_decisions = [d for d in decisions if d['selected']]
    tokens_before = sum(len(r['input_ids']) for r in single_records)
    tokens_after = sum(len(r['input_ids']) for r in records)
    supervised = sum(r['meta']['supervised_tokens'] for r in records)
    summary = dict(issues=len(identities), selected_actions=len(selected_decisions), sequences=len(records),
        supervised_tokens=supervised, individual_window_input_tokens=tokens_before, merged_input_tokens=tokens_after,
        avoided_repeated_prefix_tokens=tokens_before-tokens_after,
        prefix_input_reduction_fraction=1-tokens_after/tokens_before,
        label_density_before=supervised/tokens_before, label_density_after=supervised/tokens_after,
        max_sequence_length=max(len(r['input_ids']) for r in records),
        categories=dict(Counter(d['category'] for d in selected_decisions)),
        tools=dict(Counter(d['tool_name'] for d in selected_decisions)),
        recovery_actions=sum(bool(d.get('recovery')) for d in selected_decisions),
        upstream_validation_actions=0, final_reasoning_targets=0,
        sequence_lengths=[len(r['input_ids']) for r in records])
    report = dict(name='p016-target-protocol-six-train-v1', date='2026-10-23',
        claim='selected-action RFT from six strict-success train issues; not complete task SFT or adaptation sufficiency',
        max_seq_len=32768, no_truncation=True, thought_loss='mask',
        identity_freeze_sha256=sha(frozen_path), prior_dataset_manifest_sha256=sha(OLD / 'dataset-manifest.json'),
        builder_sha256=sha(Path(__file__)), encoding_version=ENCODING_VERSION,
        encoder_sha256=sha(ROOT / 'training/encoding/gemma_tool_chat.py'),
        tokenizer_hash=tok_hash, template_sha256=sha(TOKENIZER / 'chat_template.jinja'), tools_hash=sha256_json(tools),
        sources=provenance, summary=summary, action_decisions=decisions, merging=merge_log,
        outputs={name: sha(output / name) for name in ('selected-causal-actions.jsonl', 'merged-transactions.jsonl', 'encoded-data.jsonl', 'actual-tools.json')},
        resources={'new_gpu_seconds': 0, 'new_model_calls': 0, 'new_submissions': 0, 'new_network_downloads': 0})
    write_json(output / 'dataset-manifest.json', report)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
