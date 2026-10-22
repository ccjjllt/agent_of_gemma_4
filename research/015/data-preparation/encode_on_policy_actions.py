#!/usr/bin/env python3
"""Encode complete real request/action/observation windows from frozen train traces.

No history clipping, fabricated observations, terminal reasoning rewrite or worker
changes. Only the current assistant action receives loss. Capture audit and task
resolution are independent prerequisites; original artifacts remain immutable.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from training.encoding.gemma_tool_chat import (encode_transaction, load_tokenizer,
    normalize_messages, sha256_json, tokenizer_fingerprint)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attempt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--capture-audit', type=Path)
    ap.add_argument('--limits', type=int, nargs='+', default=[6846, 8192])
    args = ap.parse_args()
    p = args.attempt.resolve()
    # Inspect train identity and split BEFORE opening any trajectory contents.
    result = read(p / 'result.json')
    task = read(p / 'agent-task.json')
    frozen = read(ROOT / 'research/015/data-preparation/train-identity-freeze.json')
    identity = next((x for x in frozen['selected'] if x['instance_id'] == task['instance_id']), None)
    require(result['split'] == 'train' and identity is not None, 'only frozen train identities allowed')
    require(identity['repo'] == task['repo'] and identity['base_commit'] == task['base_commit'], 'frozen source identity mismatch')
    require(result['resolved'] and result['judge_valid'], 'requires actual resolved and valid host judge')
    capture = read(p / 'trajectory-capture.json')
    audit = read(args.capture_audit) if args.capture_audit else None
    wire = jsonl(p / 'raw-wire.jsonl')
    model = jsonl(p / 'raw-model.jsonl')
    events = [x['event'] for x in jsonl(p / 'raw-adk-events.jsonl')]
    requests = [x for x in wire if x['kind'] == 'wire_request']
    require(len({x['wire_id'] for x in requests}) == len(requests), 'duplicate wire requests')
    responses = {x['wire_id']: x for x in wire if x['kind'] == 'wire_response'}
    ends = {x['wire_id']: x for x in wire if x['kind'] == 'wire_end'}
    model_responses = {x['model_request_id']: x for x in model if x['kind'] == 'model_response'}
    model_ends = {x['model_request_id']: x for x in model if x['kind'] == 'model_end'}
    if audit:
        require(audit.get('verified') is True, 'unverified capture audit')
        rid = audit['model_request_id']
        final = responses[audit['wire_id']]['response']['choices'][0]['message']
        require(requests[-1]['model_request_id'] == rid and ends[audit['wire_id']]['completed'], 'audit must concern complete terminal response')
        require(digest_text(final.get('content') or '') == audit['content_sha256'] and digest_text(final.get('reasoning_content') or '') == audit['reasoning_sha256'], 'audit content hashes disagree')
    tools = next((q['request'].get('tools') for q in requests if q['request'].get('tools')), [])
    tokenizer_dir = ROOT / 'research/015/tokenizer-assets/official-31b-ct-v2-tokenizer'
    tokenizer = load_tokenizer(str(tokenizer_dir))
    tok_hash = tokenizer_fingerprint(tokenizer)
    template_path = tokenizer_dir / 'chat_template.jinja'
    template = template_path.read_text()
    hashes = {name: digest(p / name) for name in ['raw-wire.jsonl', 'raw-model.jsonl', 'raw-adk-events.jsonl']}
    args.output.mkdir(parents=True, exist_ok=True)
    buffers = {(limit, policy): [] for limit in args.limits for policy in ['mask', 'include']}
    transactions, census = [], []
    for index, req in enumerate(requests):
        try:
            wid, rid = req['wire_id'], req['model_request_id']
            require(wid in responses and wid in ends and ends[wid]['completed'], 'missing or incomplete wire response/end')
            response = responses[wid]['response']['choices'][0]
            current = response['message']
            require(rid in model_responses and rid in model_ends, 'missing corresponding model response/end')
            if not model_ends[rid]['completed']:
                require(audit and audit['model_request_id'] == rid, 'unreviewed interrupted model response')
            calls = current.get('tool_calls') or []
            if not calls:
                census.append({'call_index': index, 'model_request_id': rid, 'excluded': 'auxiliary/non-tool response: no action target; ordinary final reasoning unsupported by v3'})
                continue
            require(req['request'].get('tools') == tools, 'unsupported changed action tool schema; window excluded only')
            require(response['finish_reason'] == 'tool_calls', 'nonterminal or truncated action response')
            raw_model = model_responses[rid]['response']
            require(not raw_model.get('partial') and not raw_model.get('error_code') and not raw_model.get('interrupted'), 'partial/error/interrupted model response')
            parts = (raw_model.get('content') or {}).get('parts', [])
            require(sum(bool(x.get('function_call')) for x in parts) == len(calls), 'wire/model tool-call count mismatch')
            require(''.join(x.get('text') or '' for x in parts if x.get('thought')) == (current.get('reasoning_content') or ''), 'wire/model reasoning mismatch')
            require(''.join(x.get('text') or '' for x in parts if not x.get('thought')) == (current.get('content') or ''), 'wire/model visible text mismatch')
            require(index + 1 < len(requests), 'action lacks next actual request containing its observation')
            history = req['request']['messages']
            following = requests[index + 1]['request']['messages']
            require(following[:len(history)] == history, 'next request rewrote/compacted history: use explicit separate segment design')
            # Compare supported fields semantically; provider-only metadata is not training text.
            left = normalize_messages(history + [current] + following[len(history)+1:], tools)[0]
            right = normalize_messages(following, tools)[0]
            require(left == right, 'actual response disagrees with next request assistant/history')
            observations = following[len(history) + 1:]
            require(len(observations) == len(calls) and all(x['role'] == 'tool' for x in observations), 'missing or extra tool observation')
            for call, observation in zip(calls, observations):
                model_calls = [x['function_call'] for x in parts if x.get('function_call') and x['function_call'].get('id') == call['id']]
                require(len(model_calls) == 1 and model_calls[0]['name'] == call['function']['name'] and model_calls[0]['args'] == json.loads(call['function']['arguments']), 'wire/model function call mismatch')
                require(observation['tool_call_id'] == call['id'], 'tool call ID mismatch')
                original = [part['function_response'] for event in events for part in (event.get('content') or {}).get('parts', []) if part.get('function_response') and part['function_response']['id'] == call['id']]
                require(len(original) == 1 and original[0]['name'] == call['function']['name'] and original[0]['response'] == json.loads(observation['content']), 'raw ADK tool response mismatch')
            ordinal = sum(x['role'] == 'assistant' for x in history)
            sample = task['instance_id'] + '-action-%02d' % index
            tx = {'sample_id': sample, 'transaction_id': sample,
                  'origin_kind': 'on-policy-action-rft', 'protocol_id': 'actual-adk-wire-' + sha256_json(tools),
                  'group': {'issue': task['instance_id'], 'repo': task['repo'], 'commit': task['base_commit'], 'split': 'train'},
                  'source': {'uri': str(p / 'raw-wire.jsonl'), 'raw_sha256': hashes['raw-wire.jsonl'], 'trace_id': rid, 'format': 'raw-wire-complete-causal-action-window'},
                  'messages': copy.deepcopy(history + [current] + observations),
                  'target_assistant_steps': [ordinal],
                  'behavior': {'call_index': index, 'wire_id': wid, 'tool_call_ids': [c['id'] for c in calls],
                               'task_resolved': True, 'original_capture_usable': capture['training_trace_usable'],
                               'capture_audit_sha256': digest(args.capture_audit) if audit else None,
                               'instruction_violation_repro_write': any(c['function']['name'] == 'write_file' and json.loads(c['function']['arguments']).get('filepath') == 'repro.py' for c in calls)}}
            row = {'call_index': index, 'sample_id': sample, 'model_request_id': rid, 'wire_id': wid, 'calls': calls, 'instruction_violation_repro_write': tx['behavior']['instruction_violation_repro_write']}
            encodings = {policy: encode_transaction(tx, tokenizer, tools, max_seq_len=131072, thought_loss=policy, chat_template=template, tokenizer_hash=tok_hash, tokenizer_revision='official-mounted-31b-ct-v2') for policy in ['mask', 'include']}
            transactions.append(tx)
            for policy, encoded in encodings.items():
                row['tokens'] = len(encoded['input_ids'])
                row[policy + '_selected_tokens'] = encoded['meta']['supervised_tokens']
                for limit in args.limits:
                    if row['tokens'] <= limit:
                        limited = copy.deepcopy(encoded)
                        limited['meta']['max_seq_len'] = limit
                        buffers[limit, policy].append(limited)
            census.append(row)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            census.append({'call_index': index, 'model_request_id': req.get('model_request_id'), 'excluded': type(error).__name__ + ': ' + str(error)})
    outputs = {}
    def write_jsonl(name, records):
        dest = args.output / name
        dest.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records))
        outputs[name] = {'sha256': digest(dest), 'records': len(records)}
    write_jsonl('causal-action-transactions.jsonl', transactions)
    for (limit, policy), records in buffers.items():
        name = 'encoded-%d-thought-%s.jsonl' % (limit, policy)
        write_jsonl(name, records)
        outputs[name]['supervised_tokens'] = sum(x['meta']['supervised_tokens'] for x in records)
        outputs[name]['instruction_violation_repro_write_records'] = sum(x['meta']['behavior']['instruction_violation_repro_write'] for x in records)
    (args.output / 'actual-tools.json').write_text(json.dumps(tools, indent=2) + '\n')
    manifest = {'scope': 'on-policy action RFT; complete visible causal state, current action only; no task-SFT/final support claim', 'instance_id': task['instance_id'], 'split': 'train', 'raw_source_hashes': hashes, 'original_capture': capture, 'capture_audit': audit, 'capture_policy': 'Per-window complete wire/model/end and real executed ADK observations; original episode flags retained independently. Compacted transitions/auxiliary requests only exclude affected windows; no previous history reassembly.', 'encoder_script_sha256': digest(Path(__file__)), 'tokenizer_hash': tok_hash, 'chat_template_sha256': digest(template_path), 'tools_sha256': sha256_json(tools), 'outputs': outputs, 'census': census, 'selection_policy': 'All complete source actions retained with policy-violation flags; no automatic endorsement of repro-file writes. Full historical context retained even if an earlier action violated instructions.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(outputs, indent=2))


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == '__main__':
    main()
